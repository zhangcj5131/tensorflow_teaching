def generate_num(n=1000000000):
    re = []
    for i in range(n):
        re.append(i)
    return re

def yield_num(n = 1000000000):
    for i in range(n):
        yield i


fun = yield_num(10000000)
print(next(fun))
print(next(fun))
print(next(fun))
print(next(fun))

# print(next(yield_num(10000000)))
# print(next(yield_num(10000000)))
# print(next(yield_num(10000000)))
# print(next(yield_num(10000000)))
# for i in yield_num():
#     print(i)
# for i in generate_num():
#     print(i)
