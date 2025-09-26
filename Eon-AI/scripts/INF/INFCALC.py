from infmath import *

def calculator():
    inp = ''
    z = 0
    inp = input("infn number +>x (q to quit): ")
    if inp == 'q':
        return -1
    z = int(inp)
    start_x = z
    inp = input("infn number +>y (q to quit): ")
    if inp == 'q':
        ret = infn_set(start_x)
        print(f"+>{start_x} = {sorted(ret)}")
        return ret
    z = int(inp)
    start_y = z
    
    final_result = infn_add(start_x, start_y)
    print(f"+>{start_x} + +>{start_y} = {sorted(final_result)}")

    while True:
        inp = input("Enter +>z (type -x to subtract) (q to quit): ")
        if inp == 'q':
            break
        z = int(inp)
        if z < 0:
            mult_scalar = -z
            final_result = infn_subt_append(final_result, mult_scalar)
            print(f"... - +>{-z} = {sorted(final_result)}")
            continue
        final_result = infn_append(final_result, z)
        print(f"... + +>{z} = {sorted(final_result)}")
    
    return final_result

if __name__ == "__main__":
    calculator()
