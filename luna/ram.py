"""
The RAM system is used to conveniently create globally temporary values in
any place of a code.

The values to store in a RAM have the below features:
    - Users do not want to **declare it explicitly** in the program, which
        makes the code rather dirty.
    - Users want to **share** it across functions, or even files.
    - Users use it **temporarily**, such as for debugging
    - Users want to **reuse** a group of values several times, while **reset** each
        value in the group before reusing them will add a great overhead to the code.
"""

__global_ram = {}

def ram_write(k, v):
    __global_ram[k] = v


def ram_pop(k):
    return __global_ram.pop(k)


def ram_append(k, v):
    if k not in __global_ram:
        __global_ram[k] = []
    __global_ram[k].append(v)


def ram_read(k):
    return __global_ram[k]


def ram_has(k):
    return k in __global_ram


# def ram_linear_analyze(k):
    # y = __global_ram[k]
    # x = list(range(len(y)))
    # reg = LinearRegression().fit(np.array(x).reshape(-1, 1),
    #                              np.array(y).reshape(-1, 1))
    # return "y={:4.2f}*x+{:4.2f}".format(cast_item(reg.coef_), cast_item(reg.intercept_))


def ram_reset(prefix=None):
    if prefix is not None:
        for key in __global_ram:
            if key.startswith(prefix):
                __global_ram.pop(prefix)
    else:
        __global_ram.clear()
        