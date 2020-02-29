import p13_auto_derivative as auto
import math

def train(y, x, epoches = 2000, lr = 0.01):
    dy_dx = y.deriv(x)

    x0 = 1
    for _ in range(epoches):
        x0 -= lr * dy_dx.eval(x=x0)
    return x0

if __name__ == '__main__':
    x = auto.Variable('x')
    for a in range(2,10):
        y = (x*x - a) * (x*x - a)
        y_predict = train(y, x)
        print(a, math.sqrt(a), y_predict)