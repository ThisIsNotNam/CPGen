"""
lexer.py — Converts CPGen source text into a flat token stream.

Handles indentation-based block structure by emitting INDENT / DEDENT tokens,
so the parser never needs to inspect raw whitespace.
"""

import re
from typing import List

from .tokens import Token, TokenType


class Lexer:
    # Rules are tried in order; the first match wins.
    #fmt: off
    _TOKEN_RULES: List[tuple] = [
        ("PRINT",      r"@print\b"),
        ("PRINTLN",    r"@println\b"),
        ("REPEAT",     r"repeat\b"),
        ("INT",        r"int\b"),
        ("FLOAT",      r"float\b"),
        ("STRING",     r"string\b"),
        ("ARRAY",      r"array\b"),
        ("OF",         r"of\b"),
        ("FROM",       r"from\b"),
        ("DOTDOT",     r"\.\."),
        ("COLON",      r":"),
        ("COMMA",      r","),
        ("LPAREN",     r"\("),
        ("RPAREN",     r"\)"),
        ("LBRACKET",   r"\["),
        ("RBRACKET",   r"\]"),
        ("PLUS",       r"\+"),
        ("MINUS",      r"-"),
        ("STAR",       r"\*"),
        ("SLASH",      r"/"),
        ("EQUAL",      r"(?<!=)=(?!=)"),   
        # Floats, scientific notation, and plain integers — order matters.
        ("NUMBER",     r"\d+\.\d+|\d+[eE][+-]?\d+|\d+"),
        ("IDENTIFIER", r"[A-Za-z_][A-Za-z0-9_]*"),
        # Both single- and double-quoted strings.
        ("STRING_LIT", r'"[^"]*"|\'[^\']*\''),
        ("COMMENT",    r"#.*"),
        # Spaces, tabs, and HTML non-breaking spaces (\xa0).
        ("SKIP",       r"[ \t\xa0]+"),
    ]
    #fmt: on

    def __init__(self, source_code: str) -> None:
        self.source_code = source_code
        self._compiled_rules = [
            (name, re.compile(pattern)) for name, pattern in self._TOKEN_RULES
        ]

    # ── public entry point ────────────────────────────────────────────────

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        indent_stack: List[int] = [0]
        lines = self.source_code.splitlines()

        for line_num, line in enumerate(lines, start=1):
            # Skip blank lines and comment-only lines.
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip())
            self._emit_indent_tokens(indent, indent_stack, tokens, line_num)
            self._scan_line(line.lstrip().rstrip(), indent, line_num, tokens)
            tokens.append(Token(TokenType.NEWLINE, "\n", line_num, len(line) + 1))

        # Close any remaining open indent levels at end-of-file.
        eof_line = len(lines) + 1
        while len(indent_stack) > 1:
            indent_stack.pop()
            tokens.append(Token(TokenType.DEDENT, "", eof_line, 1))
        tokens.append(Token(TokenType.EOF, "", eof_line, 1))
        return tokens

    # ── private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _emit_indent_tokens(
        indent: int,
        indent_stack: List[int],
        tokens: List[Token],
        line_num: int,
    ) -> None:
        """Push INDENT or pop DEDENT(s) to reflect the new indentation level."""
        if indent > indent_stack[-1]:
            indent_stack.append(indent)
            tokens.append(Token(TokenType.INDENT, "    ", line_num, indent + 1))
        elif indent < indent_stack[-1]:
            while indent < indent_stack[-1]:
                indent_stack.pop()
                tokens.append(Token(TokenType.DEDENT, "", line_num, indent + 1))
            if indent != indent_stack[-1]:
                raise SyntaxError(f"Line {line_num}, Col 1: Indentation mismatch.")

    def _scan_line(
        self,
        text: str,
        indent: int,
        line_num: int,
        tokens: List[Token],
    ) -> None:
        """Tokenise a single source line and append results to *tokens*."""
        pos = 0
        while pos < len(text):
            matched = False
            for name, regex in self._compiled_rules:
                mo = regex.match(text, pos)
                if not mo:
                    continue

                kind = name
                value = mo.group(0)
                column = indent + pos + 1

                if kind == "COMMENT":
                    return  # Rest of line is a comment; stop scanning.
                if kind != "SKIP":
                    tokens.append(Token(TokenType[kind], value, line_num, column))
                pos = mo.end()
                matched = True
                break

            if not matched:
                column = indent + pos + 1
                raise SyntaxError(
                    f"Line {line_num}, Col {column}: "
                    f"Unexpected character {text[pos]!r}"
                )
