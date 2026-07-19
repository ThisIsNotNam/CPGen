import ast
from .lexer import Lexer
from .parser import Parser


def _parse(source: str) -> ast.Module:
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def build(source: str) -> str:
    return ast.unparse(_parse(source))
