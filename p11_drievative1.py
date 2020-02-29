

def f(x):
    return (x - 2) ** 2 + 100

def df_dx(x):
    return 2 * (x - 2)

def dx(x, lr = 0.01):
    return - lr * df_dx(x)

x = 4
for _ in range(10000):
    x += dx(x)
print(x)