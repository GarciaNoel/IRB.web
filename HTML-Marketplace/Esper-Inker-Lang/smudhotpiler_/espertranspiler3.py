class RitualPlusPlusTranspiler:
    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.output = []

    def transpile(self, code):
        lines = code.strip().splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("let "):
                self._assign(line)
            elif line.startswith("define "):
                i = self._define_function(lines, i)
            elif line.endswith("()"):
                self._call_function(line[:-2])
            elif line.startswith("if "):
                self._handle_conditional(line)
            elif line.startswith("repeat "):
                self._handle_loop(line)
            else:
                self._translate(line)
            i += 1
        return "\n".join(self.output)

    def _assign(self, line):
        name, val = line[4:].split("=", 1)
        self.variables[name.strip()] = val.strip().strip('"')

    def _define_function(self, lines, i):
        name = lines[i].split()[1][:-2]
        body = []
        i += 1
        while i < len(lines) and lines[i].strip() != "}":
            body.append(lines[i].strip())
            i += 1
        self.functions[name] = body
        return i

    def _call_function(self, name):
        for line in self.functions.get(name, []):
            self._translate(line)

    def _handle_conditional(self, line):
        condition, action = line[3:].split("then", 1)
        var, val = condition.strip().split("==")
        if self.variables.get(var.strip()) == val.strip().strip('"'):
            self._translate(action.strip())

    def _handle_loop(self, line):
        parts = line.split()
        count = int(parts[1])
        action = " ".join(parts[3:])
        for _ in range(count):
            self._translate(action)

    def _translate(self, line):
        if line.startswith("echo("):
            msg = line[5:-1].strip('"')
            self.output.append(f'echo "{msg}"')
        elif line.startswith("write("):
            file, msg = self._extract_args(line)
            self.output.append(f'echo "{msg}" > {file}')
        elif line.startswith("read("):
            file = self._extract_single_arg(line)
            self.output.append(f'cat {file}')
        else:
            self.output.append(f'# Unknown command: {line}')

    def _extract_args(self, line):
        args = line[line.find("(")+1:line.find(")")].split(",")
        return args[0].strip().strip('"'), args[1].strip().strip('"')

    def _extract_single_arg(self, line):
        return line[line.find("(")+1:line.find(")")].strip().strip('"')


# Example usage
if __name__ == "__main__":
    ritual_code = """
    let mood = "ambient-chaos"

    define greet() {
      echo("Welcome to the ritual.")
      if mood == "ambient-chaos" then echo("Mood confirmed.")
    }

    repeat 2 times echo("Looping...")

    greet()

    write("ritual.log", "Loop complete.")
    read("ritual.log")
    """

    transpiler = RitualPlusPlusTranspiler()
    bash_output = transpiler.transpile(ritual_code)
    print("#!/bin/bash")
    print(bash_output)
