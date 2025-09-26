def infn_set(x):
    return set([-x - 1, -x, -x + 1, 0, x - 1, x, x + 1])

def infn_subt(x, y):
    set_x = infn_set(x)
    set_y = infn_set(y)

    result = set()
    for num_x in set_x:
        for num_y in set_y:
            result.add(num_x - num_y)

    return sorted(result)

def infn_subt_append(x, y):
    set_x = x
    set_y = infn_set(y)

    result = set()
    for num_x in set_x:
        for num_y in set_y:
            result.add(num_x - num_y)

    return sorted(result)

def infn_subt_extend(informal_set_x, informal_set_y):
    result = set()

    for num_x in informal_set_x:
        for num_y in informal_set_y:
            result.add(num_x - num_y)

    return sorted(result)

def infn_add(x, y):
    set_x = infn_set(x)
    set_y = infn_set(y)

    result = set()
    for num_x in set_x:
        for num_y in set_y:
            result.add(num_x + num_y)

    return sorted(result)

def infn_append(x, y):
    set_x = x
    set_y = infn_set(y)

    result = set()
    for num_x in set_x:
        for num_y in set_y:
            result.add(num_x + num_y)

    return sorted(result)


def infn_add_extend(informal_set_x, informal_set_y):
    result = set()

    for num_x in informal_set_x:
        for num_y in informal_set_y:
            result.add(num_x + num_y)

    return sorted(result)