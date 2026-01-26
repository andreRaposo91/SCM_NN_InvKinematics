
# class Builder():
#     def __init__(self, data, models):
#         print([fun(data, *args, **kwargs) for fun, args, kwargs in models])

# def do_this(data, pos_arg, kw_arg1=True, kw_arg2=5):
#     print('pos_arg', pos_arg)
#     print('kw_arg1', kw_arg1)
#     print('kw_arg2', kw_arg2)


# Builder([i for i in range(10)], [(do_this, [[10]], {})])

import time

def test():
    i = 0
    start_time = time.time()
    while i < 1000:
        i += 1
    t = time.time() - start_time
    print(i)
    print(f"{t / 1000:6f}")

for _ in range(20):
 test()
