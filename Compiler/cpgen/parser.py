"""
parser.py — Recursive-descent parser for CPGen.

Consumes the token stream produced by Lexer and builds a Python AST directly.
Validation (undefined vars, bad bounds, etc.) happens inline during parsing —
there is no separate semantic analysis pass.

CPGen syntax → Python AST mapping
──────────────────────────────────
  x: int from [lo..hi]          →  _u_x = random.randint(lo, hi)
  x: float from [lo..hi]        →  _u_x = random.uniform(lo, hi)
  x: string(n) from [lo..hi]    →  _u_x = ''.join(random.choices(...range..., k=n))
  x: string(n) from <charset>   →  _u_x = ''.join(random.choices('<chars>', k=n))
  x: int = 42                   →  _u_x = 42
  x: float = 3.14               →  _u_x = 3.14
  x: string = "hi"              →  _u_x = 'hi'
  x: array(n) of <type>         →  _u_x = [<expr> for _ in range(n)]
  repeat <n>:                   →  for _ in range(n):
  @print(a, b)                  →  print(_u_a, _u_b, end='')
  @println(a, b)                →  print(_u_a, _u_b, end='\\n')

All user-declared variable names are prefixed with _u_ to prevent collisions
with Python builtins and runtime names (random, range, print, etc.).

Output: ast.Module containing a flat list of statements (import + assignments +
loops + print calls). ast.unparse() on the module gives a runnable Python script.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Tuple

from .tokens import Token, TokenType

_PREFIX = "_u_"

# Maps token types that are valid inside numeric/variable expressions.
_EXPR_TYPES = frozenset(
    {
        TokenType.NUMBER,
        TokenType.IDENTIFIER,
        TokenType.PLUS,
        TokenType.MINUS,
        TokenType.STAR,
        TokenType.SLASH,
        TokenType.LPAREN,
        TokenType.RPAREN,
    }
)

# Operator precedence and AST node mapping for binary expressions.
_PREC: Dict[TokenType, int] = {
    TokenType.PLUS: 1,
    TokenType.MINUS: 1,
    TokenType.STAR: 2,
    TokenType.SLASH: 2,
}
_BINOP: Dict[TokenType, ast.operator] = {
    TokenType.PLUS: ast.Add(),
    TokenType.MINUS: ast.Sub(),
    TokenType.STAR: ast.Mult(),
    TokenType.SLASH: ast.Div(),
}

# String value of types

_ALL_TYPES = frozenset({"int", "float", "string", "array"})
_NUMERIC_TYPES = frozenset({"int", "float"})
_STRING_TYPES = frozenset({"string"})


def _u(name: str) -> str:
    """Apply the user-variable prefix."""
    return _PREFIX + name


# ── AST construction helpers ──────────────────────────────────────────────────


def _name(id: str) -> ast.Name:
    return ast.Name(id=id, ctx=ast.Load())


def _attr(obj: str, attr: str) -> ast.Attribute:
    return ast.Attribute(value=_name(obj), attr=attr, ctx=ast.Load())


def _call(func: ast.expr, *args: ast.expr) -> ast.Call:
    return ast.Call(func=func, args=list(args), keywords=[])


def _const(value) -> ast.Constant:
    return ast.Constant(value=value)


def _assign(name: str, value: ast.expr) -> ast.Assign:
    target = ast.Name(id=name, ctx=ast.Store())
    return ast.Assign(targets=[target], value=value, lineno=0, col_offset=0)


def _static_numeric(node: ast.expr):
    """
    Return the numeric value if *node* is a statically-known number, else None.
    Handles ast.Constant and ast.UnaryOp(USub, Constant).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -node.operand.value
    return None


def _check_static(
    node: ast.expr, loc: Token, *, gt: float | None = None, ge: float | None = None
) -> None:
    """
    If *node* is a statically-known number, verify it satisfies the given
    constraint (gt = must be strictly greater than, ge = must be >= ).
    Raises ValueError with a location-tagged message on failure.
    """
    v = _static_numeric(node)
    if v is None:
        return
    if gt is not None and v <= gt:
        raise ValueError(
            f"Line {loc.line}, Col {loc.column}: " f"Expected value > {gt}, got {v}."
        )
    if ge is not None and v < ge:
        raise ValueError(
            f"Line {loc.line}, Col {loc.column}: " f"Expected value >= {ge}, got {v}."
        )


def _char_from_tok(tok: Token) -> str:
    """Extract a single character from a STRING_LIT or NUMBER token."""
    return tok.value[1:-1] if tok.type == TokenType.STRING_LIT else tok.value


# ── parser ────────────────────────────────────────────────────────────────────


class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        # name → 'array' | 'scalar'  (presence = declared)
        self._syms: Dict[str, str] = {}

    # ── token navigation ──────────────────────────────────────────────────

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def lookahead(self, offset: int) -> Token:
        """Token at current position + *offset*. lookahead(0) == peek()."""
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else self.tokens[-1]

    def consume(self, expected: TokenType) -> Token:
        tok = self.peek()
        if tok.type != expected:
            raise SyntaxError(
                f"Line {tok.line}, Col {tok.column}: "
                f"Expected {expected.name}, got {tok.type.name} ({tok.value!r})"
            )
        self.pos += 1
        return tok

    def match(self, expected: TokenType) -> bool:
        if self.peek().type == expected:
            self.pos += 1
            return True
        return False

    # ── entry point ───────────────────────────────────────────────────────

    def parse(self) -> ast.Module:
        stmts: List[ast.stmt] = [ast.Import(names=[ast.alias(name="random")])]
        while self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.NEWLINE:
                self.consume(TokenType.NEWLINE)
                continue
            stmts.extend(self._parse_statement())

        module = ast.Module(body=stmts, type_ignores=[])
        ast.fix_missing_locations(module)
        return module

    # ── statements ────────────────────────────────────────────────────────

    def _parse_statement(self) -> List[ast.stmt]:
        tok = self.peek()
        if tok.type == TokenType.REPEAT:
            return [self._parse_repeat()]
        if tok.type in (TokenType.PRINT, TokenType.PRINTLN):
            return [self._parse_print(tok.type)]
        if tok.type == TokenType.IDENTIFIER:
            if self.lookahead(1).type == TokenType.COLON:
                return [self._parse_var_decl()]
            else:
                return [self._parse_reassignment()]
        raise SyntaxError(
            f"Line {tok.line}, Col {tok.column}: "
            f"Unexpected token at start of statement: {tok.value!r}"
        )

    def _parse_repeat(self) -> ast.For:
        self.consume(TokenType.REPEAT)
        loc = self.peek()
        count_node = self._parse_expr_until(TokenType.COLON)

        _check_static(count_node, loc, ge=0)

        self.consume(TokenType.COLON)
        self.consume(TokenType.NEWLINE)
        self.consume(TokenType.INDENT)

        body: List[ast.stmt] = []
        while self.peek().type != TokenType.DEDENT:
            if self.peek().type == TokenType.NEWLINE:
                self.consume(TokenType.NEWLINE)
                continue
            body.extend(self._parse_statement())

        self.consume(TokenType.DEDENT)

        return ast.For(
            target=ast.Name(id="_", ctx=ast.Store()),
            iter=_call(_name("range"), count_node),
            body=body,
            orelse=[],
        )

    def _parse_print(self, tok_type: TokenType) -> ast.Expr:
        self.consume(tok_type)
        self.consume(TokenType.LPAREN)

        args: List[ast.expr] = []

        def _read_one() -> ast.expr:
            t = self.peek()
            if t.type == TokenType.IDENTIFIER:
                self.pos += 1
                if t.value not in self._syms:
                    raise NameError(
                        f"Line {t.line}, Col {t.column}: "
                        f"Cannot print undefined variable '{t.value}'"
                    )
                if self._syms[t.value] == "array":
                    return ast.Starred(value=_name(_u(t.value)), ctx=ast.Load())
                return _name(_u(t.value))
            if t.type == TokenType.NUMBER:
                self.pos += 1
                return self._token_to_number_const(t)
            if t.type == TokenType.STRING_LIT:
                self.pos += 1
                return _const(t.value[1:-1])
            raise SyntaxError(
                f"Line {t.line}, Col {t.column}: "
                f"Expected variable, number, or string literal, got {t.type.name}"
            )

        args.append(_read_one())
        while self.match(TokenType.COMMA):
            args.append(_read_one())

        self.consume(TokenType.RPAREN)
        self.consume(TokenType.NEWLINE)

        end = "\n" if tok_type == TokenType.PRINTLN else ""
        print_call = ast.Call(
            func=_name("print"),
            args=args,
            keywords=[ast.keyword(arg="end", value=_const(end))],
        )
        return ast.Expr(value=print_call)

    def _parse_var_decl(self) -> ast.Assign:
        name_tok = self.consume(TokenType.IDENTIFIER)
        name = name_tok.value
        if name in self._syms:
            raise NameError(
                f"Line {name_tok.line}, Col {name_tok.column}: "
                f"Variable '{name}' is already defined."
            )
        self.consume(TokenType.COLON)
        value_expr, kind = self._parse_type()
        self.consume(TokenType.NEWLINE)
        self._syms[name] = kind
        return _assign(_u(name), value_expr)

    def _parse_reassignment(self) -> ast.Assign:
        name_tok = self.consume(TokenType.IDENTIFIER)
        name = name_tok.value
        if name not in self._syms:
            raise NameError(
                f"Line {name_tok.line}, Col {name_tok.column}: "
                f"Undefined variable '{name}'"
            )
        dtype = self._syms[name]
        if dtype == "array":
            raise NotImplementedError(
                f"Line {name_tok.line}, Col {name_tok.column}: "
                f"TODO: implement array reassignment"
            )
        self.consume(TokenType.EQUAL)
        if dtype == "string":
            if (
                self.peek().type != TokenType.STRING_LIT
                and not self.peek().type in _ALL_TYPES
            ):
                raise TypeError(
                    f"Line {self.peek().line}, Col {self.peek().column}: "
                    f"Expected string literal, got {self.peek().type.name}"
                )
            value_expr = self._parse_string_literal()
        else:
            value_expr = self._parse_expr_until(
                TokenType.NEWLINE,
                force_float=(dtype == "float"),
                allowed_types=_NUMERIC_TYPES,
            )
            value_expr = _call(_name(dtype), value_expr)  # cast to correct type
        self.consume(TokenType.NEWLINE)
        return _assign(_u(name), value_expr)

    # ── type / value expressions ──────────────────────────────────────────

    def _parse_type(self) -> Tuple[ast.expr, str]:
        """Returns (ast.expr, kind) where kind is 'array' | 'int' | 'float' | 'string'."""
        tok = self.peek()

        if tok.type == TokenType.INT:
            if self.lookahead(1).type == TokenType.FROM:
                return self._parse_rand_range(TokenType.INT), "int"
            self.consume(TokenType.INT)
            self.consume(TokenType.EQUAL)
            expr = self._parse_expr_until(
                TokenType.NEWLINE, allowed_types=_NUMERIC_TYPES
            )
            return _call(_name("int"), expr), "int"

        if tok.type == TokenType.FLOAT:
            if self.lookahead(1).type == TokenType.FROM:
                return self._parse_rand_range(TokenType.FLOAT), "float"
            self.consume(TokenType.FLOAT)
            self.consume(TokenType.EQUAL)
            expr = self._parse_expr_until(
                TokenType.NEWLINE, force_float=True, allowed_types=_NUMERIC_TYPES
            )
            return _call(_name("float"), expr), "float"

        if tok.type == TokenType.STRING:
            if self._is_string_generator():
                return self._parse_string_rand_gen(), "string"

            self.consume(TokenType.STRING)
            self.consume(TokenType.EQUAL)
            if self.peek().type == TokenType.STRING_LIT:
                return self._parse_string_literal(), "string"
            return (
                self._parse_expr_until(TokenType.NEWLINE, allowed_types=_STRING_TYPES),
                "string",
            )

        if tok.type == TokenType.ARRAY:
            return self._parse_array_rand_gen(), "array"

        raise SyntaxError(
            f"Line {tok.line}, Col {tok.column}: "
            f"Invalid type definition: {tok.value!r}"
        )

    def _parse_rand_range(self, type_tok: TokenType) -> ast.expr:
        """
        int from [lo..hi]   → random.randint(lo, hi)
        float from [lo..hi] → random.uniform(lo, hi)
        """
        loc = self.peek()
        self.consume(type_tok)
        self.consume(TokenType.FROM)
        self.consume(TokenType.LBRACKET)

        if self.peek().type == TokenType.DOTDOT:
            raise SyntaxError(
                f"Line {loc.line}, Col {loc.column}: Missing expression for lower bound."
            )
        min_expr = self._parse_expr_until(TokenType.DOTDOT)
        self.consume(TokenType.DOTDOT)

        if self.peek().type == TokenType.RBRACKET:
            raise SyntaxError(
                f"Line {loc.line}, Col {loc.column}: Missing expression for upper bound."
            )
        max_expr = self._parse_expr_until(TokenType.RBRACKET)
        self.consume(TokenType.RBRACKET)

        # Static bound check.
        mn, mx = _static_numeric(min_expr), _static_numeric(max_expr)
        if mn is not None and mx is not None and mn > mx:
            raise ValueError(
                f"Line {loc.line}, Col {loc.column}: "
                f"Lower bound ({mn}) cannot exceed upper bound ({mx})."
            )

        fn = "randint" if type_tok == TokenType.INT else "uniform"
        return _call(_attr("random", fn), min_expr, max_expr)

    def _parse_string_rand_gen(self) -> ast.expr:
        """
        string(n) from ['a'..'z']  → ''.join(random.choices('abc...', k=n))
        string(n) from 'abc'       → ''.join(random.choices('abc', k=n))
        """
        loc = self.peek()
        self.consume(TokenType.STRING)
        self.consume(TokenType.LPAREN)
        length_node = self._parse_expr_until(TokenType.RPAREN)
        self.consume(TokenType.RPAREN)

        _check_static(length_node, loc, ge=0)

        self.consume(TokenType.FROM)
        charset = self._parse_charset_expr()
        deduped = "".join(sorted(set(charset)))

        choices_call = _call(_attr("random", "choices"), _const(deduped))
        choices_call.keywords = [ast.keyword(arg="k", value=length_node)]

        return _call(
            ast.Attribute(value=_const(""), attr="join", ctx=ast.Load()),
            choices_call,
        )

    def _parse_array_rand_gen(self) -> ast.expr:
        """array(n) of <type>  →  [<inner_expr> for _ in range(n)]"""
        loc = self.peek()
        self.consume(TokenType.ARRAY)
        self.consume(TokenType.LPAREN)
        size_node = self._parse_expr_until(TokenType.RPAREN)
        self.consume(TokenType.RPAREN)
        self.consume(TokenType.OF)

        _check_static(size_node, loc, gt=0)

        inner_expr, _ = self._parse_type()

        return ast.ListComp(
            elt=inner_expr,
            generators=[
                ast.comprehension(
                    target=ast.Name(id="_", ctx=ast.Store()),
                    iter=_call(_name("range"), size_node),
                    ifs=[],
                    is_async=0,
                )
            ],
        )

    def _parse_string_literal(self) -> ast.expr:
        """string = "hello"  →  ast.Constant('hello')"""
        tok = self.consume(TokenType.STRING_LIT)
        return _const(tok.value[1:-1])

    # ── expression parser ─────────────────────────────────────────────────

    def _parse_expr_until(
        self,
        stop: TokenType,
        force_float: bool = False,
        allowed_types: frozenset[str] = _ALL_TYPES,
    ) -> ast.expr:
        """
        Parse a CPGen arithmetic/variable expression directly into a Python AST
        node, consuming tokens up to (but not including) *stop*.

        Supports: numbers, declared user variables, +, -, *, /, parentheses.
        All identifier nodes are prefixed with _u_.
        """
        loc = self.peek()
        tokens: List[Token] = []
        depth = 0
        while True:
            t = self.peek()
            if t.type == TokenType.EOF:
                break
            if t.type == stop and depth == 0:
                break
            if t.type == TokenType.LPAREN:
                depth += 1
            elif t.type == TokenType.RPAREN:
                if depth == 0:
                    break
                depth -= 1
            if t.type not in _EXPR_TYPES:
                raise SyntaxError(
                    f"Line {t.line}, Col {t.column}: "
                    f"Unexpected token in expression: {t.value!r}"
                )
            tokens.append(t)
            self.pos += 1

        if not tokens:
            raise SyntaxError(
                f"Line {loc.line}, Col {loc.column}: Expected expression."
            )

        return self._tokens_to_ast(
            tokens, loc, force_float=force_float, allowed_types=allowed_types
        )

    def _tokens_to_ast(
        self,
        tokens: List[Token],
        loc: Token,
        force_float: bool = False,
        allowed_types: frozenset[str] = _ALL_TYPES,
    ) -> ast.expr:
        """
        Convert a flat token list into a Python AST expression.
        Implements a Pratt / precedence-climbing parser for +, -, *, /.
        """
        pos = [0]  # mutable so inner functions can advance it

        def peek_tok() -> Token | None:
            return tokens[pos[0]] if pos[0] < len(tokens) else None

        def consume_tok() -> Token:
            t = tokens[pos[0]]
            pos[0] += 1
            return t

        def _check_allowed_type(dtype: str, loc: Token) -> None:
            if dtype not in allowed_types:
                raise TypeError(
                    f"Line {loc.line}, Col {loc.column}: "
                    f"Expected one of {sorted(allowed_types)}, got '{dtype}'"
                )

        def parse_primary() -> ast.expr:
            t = peek_tok()
            if t is None:
                raise SyntaxError(
                    f"Line {loc.line}, Col {loc.column}: Unexpected end of expression."
                )
            # Unary minus
            if t.type == TokenType.MINUS:
                consume_tok()
                operand = parse_primary()
                sv = _static_numeric(operand)
                if sv is not None:
                    return _const(-sv)
                return ast.UnaryOp(op=ast.USub(), operand=operand)
            # Parenthesised sub-expression
            if t.type == TokenType.LPAREN:
                consume_tok()
                node = parse_expr(0)
                closing = peek_tok()
                if closing is None or closing.type != TokenType.RPAREN:
                    raise SyntaxError(
                        f"Line {loc.line}, Col {loc.column}: Expected ')'."
                    )
                consume_tok()
                return node
            # Number
            if t.type == TokenType.NUMBER:
                consume_tok()
                node = self._token_to_number_const(t, force_float=force_float)
                _check_allowed_type(
                    "float" if isinstance(node.value, float) else "int", t
                )
                return node
            # Identifier (user variable)
            if t.type == TokenType.IDENTIFIER:
                consume_tok()
                if t.value not in self._syms:
                    raise NameError(
                        f"Line {t.line}, Col {t.column}: "
                        f"Undefined variable '{t.value}'"
                    )
                dtype = self._syms[t.value]
                if dtype == "array":
                    raise NotImplementedError(
                        f"Line {t.line}, Col {t.column}: "
                        f"TODO: implement array arithmetic in expressions"
                    )
                _check_allowed_type(dtype, t)
                return _name(_u(t.value))
            raise SyntaxError(
                f"Line {t.line}, Col {t.column}: "
                f"Unexpected token in expression: {t.value!r}"
            )

        def parse_expr(min_prec: int) -> ast.expr:
            left = parse_primary()
            while True:
                t = peek_tok()
                if t is None or t.type not in _PREC:
                    break
                prec = _PREC[t.type]
                if prec <= min_prec:
                    break

                if allowed_types == _STRING_TYPES and t.type not in (TokenType.PLUS,):
                    raise SyntaxError(
                        f"Line {t.line}, Col {t.column}: "
                        f"Operator {t.value!r} not allowed in string expressions."
                    )

                consume_tok()
                right = parse_expr(prec)
                # Constant-fold pure numeric operations.
                lv, rv = _static_numeric(left), _static_numeric(right)
                if lv is not None and rv is not None:
                    result = {
                        TokenType.PLUS: lv + rv,
                        TokenType.MINUS: lv - rv,
                        TokenType.STAR: lv * rv,
                        TokenType.SLASH: lv / rv,
                    }[t.type]
                    left = _const(
                        int(result)
                        if isinstance(result, float) and result == int(result)
                        else result
                    )
                else:
                    left = ast.BinOp(left=left, op=_BINOP[t.type], right=right)
            return left

        result = parse_expr(0)
        if pos[0] != len(tokens):
            t = tokens[pos[0]]
            raise SyntaxError(
                f"Line {t.line}, Col {t.column}: "
                f"Unexpected token in expression: {t.value!r}"
            )
        return result

    # ── charset ───────────────────────────────────────────────────────────

    def _is_string_generator(self) -> bool:
        """
        Peek ahead to determine if 'string' is followed by (size) FROM.
        lookahead(0) is the STRING token itself.
        """
        offset = 1  # skip STRING
        if self.lookahead(offset).type != TokenType.LPAREN:
            return False
        offset += 1  # skip LPAREN
        has_size = False
        while self.lookahead(offset).type not in (TokenType.RPAREN, TokenType.EOF):
            has_size = True
            offset += 1
        if self.lookahead(offset).type != TokenType.RPAREN:
            t = self.lookahead(offset)
            raise SyntaxError(
                f"Line {t.line}, Col {t.column}: Expected ')' in string size, got {t.value!r}"
            )
        offset += 1  # skip RPAREN
        return has_size and self.lookahead(offset).type == TokenType.FROM

    def _parse_charset_expr(self) -> str:
        """Compile-time: returns the full charset string."""
        result = self._parse_charset_term()
        while self.match(TokenType.PLUS):
            result += self._parse_charset_term()
        return result

    def _parse_charset_term(self) -> str:
        if self.match(TokenType.LBRACKET):
            start_tok = self.peek()
            if start_tok.type not in (TokenType.STRING_LIT, TokenType.NUMBER):
                raise SyntaxError(
                    f"Line {start_tok.line}, Col {start_tok.column}: "
                    f"Expected STRING_LIT or NUMBER in charset range, "
                    f"got {start_tok.type.name}"
                )
            self.pos += 1
            self.consume(TokenType.DOTDOT)
            end_tok = self.peek()
            if end_tok.type not in (TokenType.STRING_LIT, TokenType.NUMBER):
                raise SyntaxError(
                    f"Line {end_tok.line}, Col {end_tok.column}: "
                    f"Expected STRING_LIT or NUMBER in charset range, "
                    f"got {end_tok.type.name}"
                )
            self.pos += 1
            self.consume(TokenType.RBRACKET)

            start = _char_from_tok(start_tok)
            end = _char_from_tok(end_tok)

            if len(start) != 1 or len(end) != 1:
                raise ValueError(
                    f"Line {start_tok.line}, Col {start_tok.column}: "
                    f"Charset range bounds must be single characters, got '{start}' to '{end}'"
                )
            return "".join(chr(c) for c in range(ord(start), ord(end) + 1))

        lit = self.consume(TokenType.STRING_LIT)
        return lit.value[1:-1]

    # ── number token → ast.Constant ───────────────────────────────────────

    @staticmethod
    def _token_to_number_const(tok: Token, force_float: bool = False) -> ast.Constant:
        raw = tok.value
        is_float_literal = "." in raw or "e" in raw.lower()

        if not force_float and not is_float_literal:
            return _const(int(raw))  # plain int literal, int context — fast path

        fval = float(raw)
        return _const(fval if force_float else int(fval))  # truncates in int context
