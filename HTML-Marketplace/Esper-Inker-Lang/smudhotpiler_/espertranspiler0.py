# YO_TRANSPILER.v1
# Transpiles ESPER ritual syntax into executable Bash sabotage

class YOTranspiler:
    def __init__(self):
        self.output = []

    def transpile(self, esper_code):
        for line in esper_code.strip().splitlines():
            self._translate(line.strip())
        return "\n".join(self.output)

    def _translate(self, line):
        if line == "rotate_suffix":
            self.output.append('echo "Suffix rotated."')
        elif line == "flick_trace":
            self.output.append('echo "Trace flicked."')
        elif line.startswith("overwrite_readme"):
            path, msg = self._extract_args(line)
            self.output.append(f'mv {path}README.md {path}README.md.bak')
            self.output.append(f'echo "{msg}" > {path}README.md')
        elif line.startswith("mute"):
            file = self._extract_single_arg(line)
            self.output.append(f'mv /nut/s/{file} /nut/s/{file.replace(".zine", ".muted")}')
        elif line.startswith("fork"):
            src, dest = self._extract_args(line)
            self.output.append(f'cp /nut/s/{src} {dest}')
        else:
            self.output.append(f'echo "Unknown ritual: {line}"')

    def _extract_args(self, line):
        args = line[line.find("(")+1:line.find(")")].split(",")
        return args[0].strip().strip('"'), args[1].strip().strip('"')

    def _extract_single_arg(self, line):
        return line[line.find("(")+1:line.find(")")].strip().strip('"')


# Example usage
esper_code = """
rotate_suffix
flick_trace
overwrite_readme("/nut/s/", "This folder now exists because we rotated it.")
mute("SIGH_LOOP.zine")
fork("SMOKE_LOOP.remix", "/nuts/SMOKE_LOOP.reboot")
"""

transpiler = YOTranspiler()
bash_output = transpiler.transpile(esper_code)
print(bash_output)
