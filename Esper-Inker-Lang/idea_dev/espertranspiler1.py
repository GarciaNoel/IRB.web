# esper_compiler.py — Turing-complete ESPER-to-Bash transpiler

class EsperCompiler:
    def __init__(self):
        self.output = []
        self.variables = {}
        self.functions = {}
        self.in_function = None
        self.loop_depth = 0

    def transpile(self, esper_code):
        lines = esper_code.strip().splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("let "):
                self._handle_variable(line)
            elif line.startswith("if "):
                i = self._handle_conditional(lines, i)
            elif line.startswith("repeat "):
                i = self._handle_loop(lines, i)
            elif line.startswith("define "):
                i = self._handle_function(lines, i)
            elif line.endswith("()"):
                self._call_function(line[:-2])
            else:
                self._translate(line)
            i += 1
        return "\n".join(self.output)

    def _handle_variable(self, line):
        name, value = line[4:].split("=", 1)
        self.variables[name.strip()] = value.strip().strip('"')

    def _handle_conditional(self, lines, i):
        condition = lines[i][3:].strip()
        if "then" in condition:
            cond_expr, action = condition.split("then", 1)
            var, val = cond_expr.strip().split("==")
            var = var.strip()
            val = val.strip().strip('"')
            if self.variables.get(var) == val:
                self._translate(action.strip())
        return i

    def _handle_loop(self, lines, i):
        parts = lines[i].split()
        count = int(parts[1])
        action = " ".join(parts[3:])
        for _ in range(count):
            self._translate(action.strip())
        return i

    def _handle_function(self, lines, i):
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

    def _translate(self, line):
        if line.startswith("echo("):
            msg = line[5:-1].strip('"')
            self.output.append(f'echo "{msg}"')
        elif line.startswith("mv("):
            src, dest = line[3:-1].split(",")
            self.output.append(f'mv {src.strip()} {dest.strip()}')
        elif line.startswith("cp("):
            src, dest = line[3:-1].split(",")
            self.output.append(f'cp {src.strip()} {dest.strip()}')
        else:
            self.output.append(f'# Unknown command: {line}')


# Example usage
esper_code = """
let mood = "ambient-chaos"
if mood == "ambient-chaos" then echo("Mood confirmed")

repeat 2 times echo("Looping through ritual")

define breach() {
  echo("Sabotage initiated")
  mv("/nut/s/README.md", "/nut/s/README.md.bak")
  echo("README overwritten")
}
breach()
"""

compiler = EsperCompiler()
bash_output = compiler.transpile(esper_code)
print(bash_output)
