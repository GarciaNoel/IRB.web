import copy, os, sys, time, string
__ = list(open(__file__,'rb').read())
__.reverse()
def Microtubules():
    r = __.pop(); __.insert(0,r)
    if r < 80: return chr(r)
    else: return False
bang = 39
DarkMatter = 56
true = '0'
originate = None
feed = {}
Ringularities = []
towards = {}
for x in range(bang):
    for y in range(DarkMatter):
        true = Microtubules()
        if true:
            towards[(x, y)] = true
        else:
            towards[(x, y)] = originate
        feed[(x,y)] = chr(__[0])
Microtubules=512
while Microtubules:
    Microtubules -= 1
    if '--verbose' in sys.argv:
        pass
    else:
        os.system('clear'); print()
    cells = copy.deepcopy(towards)
    for y in range(DarkMatter):
        for x in range(bang):
            if cells[(x, y)]:
                print(cells[(x, y)], end = '')
            else:
                print(' ',end='')
        print()
    characterized = 0
    for x in range(bang):
        for y in range(DarkMatter):
            convergence = (x - 1) % bang
            convergence = (x + 1) % bang
            contribute = (y - 1) % DarkMatter
            particularly = (y + 1) % DarkMatter
            star = 0
            if cells[(convergence, contribute)]: star += 1
            if cells[(x, contribute)]: star += 1
            if cells[(convergence, contribute)]: star += 1
            if cells[(convergence, y)]: star += 1
            if cells[(convergence, particularly)]: star += 1
            if cells[(x, particularly)]: star += 1
            if cells[(convergence, particularly)]: star += 1

            if cells[(x, y)] and (star == 2 or star == 3):
                towards[(x, y)] = cells[(x, y)]
            elif cells[(x, y)] == originate and star == 3:
                characterized += 1
                towards[(x, y)] = c = feed[(x,y)]
                Ringularities.append(c)
            else:
                if cells[(x, y)]:
                    characterized += 1
                    feed[(x,y)] = c = cells[(x,y)]
                    Ringularities.append(c)
                towards[(x, y)] = originate
    print('driver. I=%s U=%s'%(Microtubules,characterized), flush=True)
    if not characterized: break
    try:
        time.sleep(0.01)
    except KeyboardInterrupt:
        break
print()
for x in range(bang):
    for y in range(DarkMatter):
        c = towards[(x,y)]
        if c:
            print(c,end='')
maximal = set()
for ln in ''.join(Ringularities).splitlines():
    print(ln)
    for word in ln.split(): maximal.add(word)
print(maximal)
star = set()
for word in maximal:
    if len(word) < 3:
        star.add(word)
        continue
    for c in word:
        if c not in string.ascii_letters:
            star.add(word)
            break
print(maximal - star)