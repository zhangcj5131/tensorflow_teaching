
a = 1.2
b = 3.4

def f(x1, x2):
    return (x1 - a) ** 2 * (x2 - b) ** 2

def df_dx1(x1, x2):
    return 2 * (x1 - a) * (x2 - b) ** 2

def df_dx2(x1, x2):
    return (x1 - a) ** 2 * 2 * (x2 - b)

def dx1(x1, x2, lr = 0.01):
    return - lr * df_dx1(x1, x2)

def dx2(x1, x2, lr = 0.01):
    return - lr * df_dx2(x1, x2)

x1 = 1.4
x2 = 3
for _ in range(3000):
    x1 += dx1(x1, x2)
    x2 += dx2(x1, x2)

print(x1)
print(x2)

