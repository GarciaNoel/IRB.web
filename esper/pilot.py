import sys
import os
import re

# Token types for lexer
TOKEN_TYPES = {
    'NUMBER': r'\d+(\.\d+)?',
    'STRING': r'"[^"]*"',
    'IDENT': r'[a-zA-Z_]\w*',
    'OPERATOR': r'[+\-*/=<>!]',
    'KEYWORD': r'(let|print|if|while|file_open|file_read|file_write|file_close|true|false)',
    'SYMBOL': r'[{};()]',
}

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

class Lexer:
    def __init__(self, code):
        self.code = code
        self.pos = 0
        self.tokens = []
        self.lex()

    def lex(self):
        while self.pos < len(self.code):
            match = None
            for token_type, pattern in TOKEN_TYPES.items():
                regex = re.compile(pattern)
                match = regex.match(self.code, self.pos)
                if match:
                    value = match.group(0)
                    if token_type != 'WHITESPACE':  # Ignore whitespace
                        self.tokens.append(Token(token_type, value))
                    self.pos = match.end()
                    break
            if not match:
                # Handle symbols and skip comments/whitespace
                char = self.code[self.pos]
                if char in ' \t\n':
                    self.pos += 1
                elif char == '#':
                    # Skip comments until end of line
                    while self.pos < len(self.code) and self.code[self.pos] != '\n':
                        self.pos += 1
                elif char in '{}();+-*/=<>!':
                    self.tokens.append(Token('SYMBOL' if char in '{}();' else 'OPERATOR', char))
                    self.pos += 1
                else:
                    raise SyntaxError(f"Unexpected character: {char}")

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.ast = []
        self.parse()

    def current_token(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def eat(self, expected_type=None, expected_value=None):
        token = self.current_token()
        if token and (expected_type is None or token.type == expected_type) and (expected_value is None or token.value == expected_value):
            self.pos += 1
            return token
        raise SyntaxError(f"Expected {expected_type} {expected_value}, got {token}")

    def parse(self):
        while self.current_token():
            self.ast.append(self.parse_statement())

    def parse_statement(self):
        token = self.current_token()
        if token.value == 'let':
            return self.parse_let()
        elif token.value == 'print':
            return self.parse_print()
        elif token.value in ('file_open', 'file_read', 'file_write', 'file_close'):
            return self.parse_file_op(token.value)
        elif token.value == 'if':
            return self.parse_if()
        elif token.value == 'while':
            return self.parse_while()
        else:
            raise SyntaxError(f"Unexpected statement: {token.value}")

    def parse_let(self):
        self.eat('KEYWORD', 'let')
        var_name = self.eat('IDENT').value
        self.eat('OPERATOR', '=')
        expr = self.parse_expression()
        self.eat('SYMBOL', ';')
        return {'type': 'LET', 'var': var_name, 'expr': expr}

    def parse_print(self):
        self.eat('KEYWORD', 'print')
        expr = self.parse_expression()
        self.eat('SYMBOL', ';')
        return {'type': 'PRINT', 'expr': expr}

    def parse_file_op(self, op_type):
        self.eat('KEYWORD', op_type)
        if op_type == 'file_open':
            var_name = self.eat('IDENT').value
            filename = self.parse_expression()
            mode = self.parse_expression()
            self.eat('SYMBOL', ';')
            return {'type': 'FILE_OPEN', 'var': var_name, 'filename': filename, 'mode': mode}
        elif op_type == 'file_read':
            var_name = self.eat('IDENT').value
            file_var = self.eat('IDENT').value
            self.eat('SYMBOL', ';')
            return {'type': 'FILE_READ', 'var': var_name, 'file_var': file_var}
        elif op_type == 'file_write':
            file_var = self.eat('IDENT').value
            content = self.parse_expression()
            self.eat('SYMBOL', ';')
            return {'type': 'FILE_WRITE', 'file_var': file_var, 'content': content}
        elif op_type == 'file_close':
            file_var = self.eat('IDENT').value
            self.eat('SYMBOL', ';')
            return {'type': 'FILE_CLOSE', 'file_var': file_var}

    def parse_if(self):
        self.eat('KEYWORD', 'if')
        self.eat('SYMBOL', '(')
        condition = self.parse_expression()
        self.eat('SYMBOL', ')')
        self.eat('SYMBOL', '{')
        body = []
        while self.current_token() and self.current_token().value != '}':
            body.append(self.parse_statement())
        self.eat('SYMBOL', '}')
        return {'type': 'IF', 'condition': condition, 'body': body}

    def parse_while(self):
        self.eat('KEYWORD', 'while')
        self.eat('SYMBOL', '(')
        condition = self.parse_expression()
        self.eat('SYMBOL', ')')
        self.eat('SYMBOL', '{')
        body = []
        while self.current_token() and self.current_token().value != '}':
            body.append(self.parse_statement())
        self.eat('SYMBOL', '}')
        return {'type': 'WHILE', 'condition': condition, 'body': body}

    def parse_expression(self):
        # Simple expression parser using Python's eval for math/strings
        expr_tokens = []
        while self.current_token() and self.current_token().value not in (';', ')', '}'):
            token = self.eat()
            expr_tokens.append(token.value if token.type in ('NUMBER', 'STRING', 'IDENT', 'OPERATOR') else token.value)
        expr_str = ''.join(expr_tokens).replace('"', '')  # Basic handling
        return {'type': 'EXPR', 'value': expr_str}

class Interpreter:
    def __init__(self, ast):
        self.ast = ast
        self.env = {}  # Runtime environment for variables
        self.file_handles = {}  # Runtime storage for file objects

    def evaluate_expr(self, expr):
        # Safely evaluate expressions using Python's eval with limited scope
        try:
            # Replace variables with their values
            expr_value = expr['value']
            for var, val in self.env.items():
                if isinstance(val, (int, float, str, bool)):
                    expr_value = re.sub(r'\b' + var + r'\b', str(val), expr_value)
            return eval(expr_value, {"__builtins__": {}}, {})  # Safe eval, no builtins
        except Exception as e:
            raise RuntimeError(f"Expression error: {e}")

    def interpret(self):
        for node in self.ast:
            self.execute(node)

    def execute(self, node):
        if node['type'] == 'LET':
            value = self.evaluate_expr(node['expr'])
            self.env[node['var']] = value
        elif node['type'] == 'PRINT':
            value = self.evaluate_expr(node['expr'])
            print(value)
        elif node['type'] == 'FILE_OPEN':
            filename = self.evaluate_expr(node['filename'])
            mode = self.evaluate_expr(node['mode'])
            try:
                file_handle = open(filename, mode)
                self.file_handles[node['var']] = file_handle
                self.env[node['var']] = f"<File: {filename}>"  # Store a placeholder in env
            except Exception as e:
                raise RuntimeError(f"File open error: {e}")
        elif node['type'] == 'FILE_READ':
            file_var = node['file_var']
            if file_var not in self.file_handles:
                raise RuntimeError(f"File not open: {file_var}")
            try:
                content = self.file_handles[file_var].read()
                self.env[node['var']] = content
            except Exception as e:
                raise RuntimeError(f"File read error: {e}")
        elif node['type'] == 'FILE_WRITE':
            file_var = node['file_var']
            if file_var not in self.file_handles:
                raise RuntimeError(f"File not open: {file_var}")
            content = self.evaluate_expr(node['content'])
            try:
                self.file_handles[file_var].write(content)
            except Exception as e:
                raise RuntimeError(f"File write error: {e}")
        elif node['type'] == 'FILE_CLOSE':
            file_var = node['file_var']
            if file_var in self.file_handles:
                self.file_handles[file_var].close()
                del self.file_handles[file_var]
                del self.env[file_var]
        elif node['type'] == 'IF':
            if self.evaluate_expr(node['condition']):
                for stmt in node['body']:
                    self.execute(stmt)
        elif node['type'] == 'WHILE':
            while self.evaluate_expr(node['condition']):
                for stmt in node['body']:
                    self.execute(stmt)

def main():
    if len(sys.argv) < 2:
        print("Usage: python pytoylang_interpreter.py <filename.toy>")
        return
    filename = sys.argv[1]
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    
    with open(filename, 'r') as f:
        code = f.read()
    
    try:
        lexer = Lexer(code)
        parser = Parser(lexer.tokens)
        interpreter = Interpreter(parser.ast)
        interpreter.interpret()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()