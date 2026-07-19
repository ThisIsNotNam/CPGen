"""
tokens.py — Token kinds and the Token data type.

Every other module in the compiler imports from here; this file imports nothing
from within the package.
"""

import enum
from dataclasses import dataclass


class TokenType(enum.Enum):
    # keywords
    REPEAT = "REPEAT"
    PRINT = "PRINT"
    PRINTLN = "PRINTLN"
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    ARRAY = "ARRAY"
    OF = "OF"
    FROM = "FROM"

    # literals & identifiers
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    STRING_LIT = "STRING_LIT"

    # operators & punctuation
    COLON = "COLON"
    COMMA = "COMMA"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    DOTDOT = "DOTDOT"
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    EQUAL = "EQUAL"

    # structural / layout
    INDENT = "INDENT"
    DEDENT = "DEDENT"
    NEWLINE = "NEWLINE"
    EOF = "EOF"


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return (
            f"Token({self.type.name}, {self.value!r}, "
            f"Line={self.line}, Col={self.column})"
        )
