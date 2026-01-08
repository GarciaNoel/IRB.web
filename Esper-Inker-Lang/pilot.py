import sys
import os
import re

# ------------------------
# TOKEN DEFINITIONS
# ------------------------

TOKEN_TYPES = [
    ("WHITESPACE", r"[ \t\n]+"),
    ("COMMENT", r"#.*"),
    ("KEYWORD", r"\b(let|print|if|while|file_open|file_read|file_write|file_close|true|false)\b"),
    ("NUMBER", r"\d+(\.\d+)?"),
    ("STRING", r'"[^"]*"'),
    ("IDENT", r"[a-zA-Z_]\w*"),
    ("OPERATOR", r"[+\-*/=<>!]"),
    ("SYMBOL", r"[{}();]"),
]

class Token:
    def __init__(self, t, v):
        self.type = t
        self.value = v

    def __repr__(self):
        return f"{self.type}:{self.value}"

# ------------------------
# LEXER
# ------------------------

class Lexer:
    def __init__(self, code):
        self.code = code
        self.pos = 0
        self.tokens = []
        self.lex()

    def lex(self):
        while self.pos < len(self.code):
            for ttype, pattern in TOKEN_TYPES:
                regex = re.compile(pattern)
                match = regex.match(self.code, self.pos)
                if match:
                    text = match.group(0)
                    self.pos = match.end()
                    if ttype not in ("WHITESPACE", "COMMENT"):
                        self.tokens.append(Token(ttype, text))
                    break
            else:
                raise SyntaxError(f"Unexpected character: {self.code[self.pos]}")

# ------------------------
# PARSER
# ------------------------

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.ast = []
        self.parse()

    def cur(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def eat(self, t=None, v=None):
        tok = self.cur()
        if tok and (t is None or tok.type == t) and (v is None or tok.value == v):
            self.pos += 1
            return tok
        raise SyntaxError(f"Expected {t} {v}, got {tok}")

    def parse(self):
        while self.cur():
            self.ast.append(self.statement())

    def statement(self):
        tok = self.cur()
        if tok.value == "let":
            return self.let_stmt()
        if tok.value == "print":
            return self.print_stmt()
        if tok.value == "if":
            return self.if_stmt()
        if tok.value == "while":
            return self.while_stmt()
        raise SyntaxError(f"Unknown statement: {tok.value}")

    def let_stmt(self):
        self.eat("KEYWORD", "let")
        name = self.eat("IDENT").value
        self.eat("OPERATOR", "=")
        expr = self.expression()
        self.eat("SYMBOL", ";")
        return ("LET", name, expr)

    def print_stmt(self):
        self.eat("KEYWORD", "print")
        expr = self.expression()
        self.eat("SYMBOL", ";")
        return ("PRINT", expr)

    def if_stmt(self):
        self.eat("KEYWORD", "if")
        self.eat("SYMBOL", "(")
        cond = self.expression()
        self.eat("SYMBOL", ")")
        body = self.block()
        return ("IF", cond, body)

    def while_stmt(self):
        self.eat("KEYWORD", "while")
        self.eat("SYMBOL", "(")
        cond = self.expression()
        self.eat("SYMBOL", ")")
        body = self.block()
        return ("WHILE", cond, body)

    def block(self):
        self.eat("SYMBOL", "{")
        body = []
        while self.cur().value != "}":
            body.append(self.statement())
        self.eat("SYMBOL", "}")
        return body

    def expression(self):
        parts = []
        while self.cur() and self.cur().value not in (";", ")", "}"):
            parts.append(self.eat().value)
        return " ".join(parts)

# ------------------------
# INTERPRETER
# ------------------------

class Interpreter:
    def __init__(self, ast):
        self.ast = ast
        self.env = {}

    def eval_expr(self, expr):
        local_env = dict(self.env)
        local_env["true"] = True
        local_env["false"] = False
        return eval(expr, {"__builtins__": {}}, local_env)

    def run(self):
        for stmt in self.ast:
            self.exec(stmt)

    def exec(self, node):
        kind = node[0]

        if kind == "LET":
            _, name, expr = node
            self.env[name] = self.eval_expr(expr)

        elif kind == "PRINT":
            print(self.eval_expr(node[1]))

        elif kind == "IF":
            _, cond, body = node
            if self.eval_expr(cond):
                for s in body:
                    self.exec(s)

        elif kind == "WHILE":
            _, cond, body = node
            while self.eval_expr(cond):
                for s in body:
                    self.exec(s)

# ------------------------
# ENTRY POINT
# ------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python esper.py <file.esper>")
        return

    with open(sys.argv[1]) as f:
        code = f.read()

    lexer = Lexer(code)
    parser = Parser(lexer.tokens)
    Interpreter(parser.ast).run()

if __name__ == "__main__":
    main()
