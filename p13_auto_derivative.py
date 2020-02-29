import math

class Exp:
    def eval(self, **env):
        pass

    def deriv(self, x):
        if not isinstance(x, Variable):
            raise Exception('cannot get derivative by a none variable!')

    def __add__(self, other):
        other = to_exp(other)
        return Add(self, other)

    def __radd__(self, other):
        other = to_exp(other)
        return Add(other, self)


    def __sub__(self, other):
        other = to_exp(other)
        return Sub(self, other)

    def __rsub__(self, other):
        other = to_exp(other)
        return Sub(other, self)

    def __mul__(self, other):
        other = to_exp(other)
        return Mul(self, other)

    def __rmul__(self, other):
        other = to_exp(other)
        return Mul(other, self)

    def __truediv__(self, other):
        other = to_exp(other)
        return Truediv(self, other)

    def __rtruediv__(self, other):
        other = to_exp(other)
        return Truediv(other, self)

    def __abs__(self):
        return Abs(self)

    @staticmethod
    def sin(other):
        other = to_exp(other)
        return Sin(other)

    @staticmethod
    def cos(other):
        other = to_exp(other)
        return Cos(other)


class Sin(Exp):
    def __init__(self, value):
        self.value = value

    def eval(self, **env):
        return math.sin(self.value.eval(**env))

    def deriv(self, x):
        return Cos(self.value)*self.value.deriv(x)

    def __repr__(self):
        return 'sin(%s)' % self.value

class Cos(Exp):
    def __init__(self, value):
        self.value = value

    def eval(self, **env):
        return math.cos(self.value.eval(**env))

    def deriv(self, x):
        return -1 *Sin(self.value) * self.value.deriv(x)

    def __repr__(self):
        return 'cos(%s)' % self.value

class Abs(Exp):
    def __init__(self,value):
        self.value = value

    def eval(self, **env):
        self.result = self.value.eval(**env)
        return abs(self.result)

    def deriv(self, x):
        return (1 if self.result > 0 else -1) * self.value.deriv(x)

    def __repr__(self):
        return '|%s|' % self.value







class Truediv(Exp):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, **env):
        if self.right.eval(**env) == 0:
            raise Exception('divided by zero!')

        if self.left.eval(**env) == 0:
            return 0
        return self.left.eval(**env) / self.right.eval(**env)

    def deriv(self, x):
        if isinstance(self.right, Const) and self.right.eval() == 0:
            raise Exception('divided by zero!')
        return (self.left.deriv(x) * self.right - self.left * self.right.deriv(x)) / (self.right * self.right)

    def __repr__(self):
        return '(%s / %s)' % (self.left, self.right)


class Mul(Exp):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, **env):
        return self.left.eval(**env) * self.right.eval(**env)

    def deriv(self, x):
        return self.left.deriv(x) * self.right + self.left * self.right.deriv(x)

    def __repr__(self):
        return '(%s * %s)' % (self.left, self.right)

class Sub(Exp):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, **env):
        return self.left.eval(**env) - self.right.eval(**env)

    def deriv(self, x):
        return self.left.deriv(x) - self.right.deriv(x)

    def __repr__(self):
        return '(%s - %s)' % (self.left, self.right)

class Add(Exp):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, **env):
        return self.left.eval(**env) + self.right.eval(**env)

    def deriv(self, x):
        return self.left.deriv(x) + self.right.deriv(x)

    def __repr__(self):
        return '(%s + %s)' % (self.left, self.right)





def to_exp(other):
    if isinstance(other, Exp):
        return other
    if type(other) == str:
        return Variable(other)
    if type(other) in (int, float):
        return Const(other)
    raise Exception('cannot convert %s to Exp' % other)

class Variable(Exp):
    def __init__(self, name):
        self.name = name

    def eval(self, **env):
        if self.name in env:
            return env[self.name]
        raise Exception('%s not found' % self.name)

    def deriv(self, x):
        super().deriv(x)
        return 1 if x.name == self.name else 0

    def __repr__(self):
        return self.name

class Const(Exp):
    def __init__(self,value):
        self.value = value

    def eval(self, **env):
        return self.value

    def deriv(self, x):
        super().deriv(x)
        return 0

    def __repr__(self):
        return str(self.value)





if __name__ == '__main__':
    x = Variable('x')
    y =  Variable('y')
    c = Const(10)

    print('-'*100)
    value = x.eval(x=2)
    print(value)
    print(x.deriv(x))

    print('-'*100)
    value = c.eval(x=2)
    print(value)
    print(c.deriv(x))

    print('-' * 100)
    y = 3 + x
    print(y.eval(x=3))
    print(y.deriv(x))

    print('-' * 100)
    y = x - 3
    print(y.eval(x=3))
    print(y.deriv(x))


    print('-' * 100)
    y = 3-x*x
    print(y.eval(x=3))
    print(y.deriv(x))

    print('-' * 100)
    y = 1/(x*x)
    print(y.eval(x=3))
    print(y.deriv(x))

    print('-' * 100)
    y = abs(5-x)
    print(y.eval(x=3))
    print(y.deriv(x))


    print('-' * 100)
    y = x*x + 1
    y = Exp.sin(y)
    print(y.eval(x=3))
    print(y.deriv(x))

