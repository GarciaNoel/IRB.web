import copy, os, sys, time, string
__ = list(open(__file__,'rb').read())
__.reverse()
def points():
    r = __.pop(); __.insert(0,r)
    if r < 80: return chr(r)
    else: return False
expanding = 71
expansion = 21
Radiation = '0'
embody = None
universes = {}
the = []
individual = {}
for x in range(expanding):
    for y in range(expansion):
        Radiation = points()
        if Radiation:
            individual[(x, y)] = Radiation
        else:
            individual[(x, y)] = embody
        universes[(x,y)] = chr(__[0])
observe=256
while observe:
    observe -= 1
    if '--verbose' in sys.argv:
        pass
    else:
        os.system('clear'); print()
    cells = copy.deepcopy(individual)
    for y in range(expansion):
        for x in range(expanding):
            if cells[(x, y)]:
                print(cells[(x, y)], end = '')
            else:
                print(' ',end='')
        print()
    topology = 0
    for x in range(expanding):
        for y in range(expansion):
            originate = (x - 1) % expanding
            found = (x + 1) % expanding
            of = (y - 1) % expansion
            systems = (y + 1) % expansion
            originating = 0
            if cells[(originate, of)]: originating += 1
            if cells[(x, of)]: originating += 1
            if cells[(found, of)]: originating += 1
            if cells[(originate, y)]: originating += 1
            if cells[(originate, systems)]: originating += 1
            if cells[(x, systems)]: originating += 1
            if cells[(found, systems)]: originating += 1

            if cells[(x, y)] and (originating == 2 or originating == 3):
                individual[(x, y)] = cells[(x, y)]
            elif cells[(x, y)] == embody and originating == 3:
                topology += 1
                individual[(x, y)] = c = universes[(x,y)]
                the.append(c)
            else:
                if cells[(x, y)]:
                    topology += 1
                    universes[(x,y)] = c = cells[(x,y)]
                    the.append(c)
                individual[(x, y)] = embody
    print('fundamental. I=%s U=%s'%(observe,topology), flush=True)
    if not topology: break
    try:
        time.sleep(0.01)
    except KeyboardInterrupt:
        break
print()
for x in range(expanding):
    for y in range(expansion):
        c = individual[(x,y)]
        if c:
            print(c,end='')
state = set()
for ln in ''.join(the).splitlines():
    print(ln)
    for word in ln.split(): state.add(word)
print(state)
free = set()
for word in state:
    if len(word) < 3:
        free.add(word)
        continue
    for c in word:
        if c not in string.ascii_letters:
            free.add(word)
            break
print(state - free)