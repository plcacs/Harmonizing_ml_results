from typing import Dict, Any

ENDMARKER: int = 0
NAME: int = 1
NUMBER: int = 2
STRING: int = 3
NEWLINE: int = 4
INDENT: int = 5
DEDENT: int = 6
LPAR: int = 7
RPAR: int = 8
LSQB: int = 9
RSQB: int = 10
COLON: int = 11
COMMA: int = 12
SEMI: int = 13
PLUS: int = 14
MINUS: int = 15
STAR: int = 16
SLASH: int = 17
VBAR: int = 18
AMPER: int = 19
LESS: int = 20
GREATER: int = 21
EQUAL: int = 22
DOT: int = 23
PERCENT: int = 24
BACKQUOTE: int = 25
LBRACE: int = 26
RBRACE: int = 27
EQEQUAL: int = 28
NOTEQUAL: int = 29
LESSEQUAL: int = 30
GREATEREQUAL: int = 31
TILDE: int = 32
CIRCUMFLEX: int = 33
LEFTSHIFT: int = 34
RIGHTSHIFT: int = 35
DOUBLESTAR: int = 36
PLUSEQUAL: int = 37
MINEQUAL: int = 38
STAREQUAL: int = 39
SLASHEQUAL: int = 40
PERCENTEQUAL: int = 41
AMPEREQUAL: int = 42
VBAREQUAL: int = 43
CIRCUMFLEXEQUAL: int = 44
LEFTSHIFTEQUAL: int = 45
RIGHTSHIFTEQUAL: int = 46
DOUBLESTAREQUAL: int = 47
DOUBLESLASH: int = 48
DOUBLESLASHEQUAL: int = 49
AT: int = 50
ATEQUAL: int = 51
OP: int = 52
COMMENT: int = 53
NL: int = 54
RARROW: int = 55
AWAIT: int = 56
ASYNC: int = 57
ERRORTOKEN: int = 58
COLONEQUAL: int = 59
N_TOKENS: int = 60
NT_OFFSET: int = 256

tok_name: Dict[int, str] = {}
for (_name, _value) in list(globals().items()):
    if isinstance(_value, int):
        tok_name[_value] = _name

def ISTERMINAL(x: int) -> bool:
    return (x < NT_OFFSET)

def ISNONTERMINAL(x: int) -> bool:
    return (x >= NT_OFFSET)

def ISEOF(x: int) -> bool:
    return (x == ENDMARKER)