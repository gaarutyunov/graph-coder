from graph_coder.ast import code_to_graph
from tests.utils import assert_equals_snapshot


def test_function():
    source = """import typing

def say_hello(a: str, b: typing.Dict[str, int] = {}, *arg, **args):
    c = a
    return f'Hello {c}!'"""

    ctx = code_to_graph(source)

    assert_equals_snapshot(ctx.g, 0)


def test_operators():
    source = """class A:
    a: int = 0
    def __init__(self, a: int):
        self.a = a

    def __str__(self):
        return f'A({self.a})'

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.a)

    def __add__(self, other):
        return A(self.a + other.a)

    def __sub__(self, other):
        return A(self.a - other.a)

    def __mul__(self, other):
        return A(self.a * other.a)

    def __truediv__(self, other):
        return A(self.a / other.a)

    def __floordiv__(self, other):
        return A(self.a // other.a)

    def __mod__(self, other):
        return A(self.a % other.a)

    def __pow__(self, other):
        return A(self.a ** other.a)

    def __lshift__(self, other):
        return A(self.a << other.a)

    def __rshift__(self, other):
        return A(self.a >> other.a)

    def __and__(self, other):
        return A(self.a & other.a)

    def __xor__(self, other):
        return A(self.a ^ other.a)

    def __or__(self, other):
        return A(self.a | other.a)

    def __neg__(self):
        return A(-self.a)

    def __pos__(self):
        return A(+self.a)

    def __abs__(self):
        return A(abs(self.a))

    def __invert__(self):
        return A(~self.a)

    def __complex__(self):
        return complex(self.a)

    def __int__(self):
        return int(self.a)

    def __float__(self):
        return float(self.a)

    def __round__(self, n=None):
        return round(self.a, n)

    def __lt__(self, other):
        return self.a < other.a

    def __le__(self, other):
        return self.a <= other.a

    def __ne__(self, other):
        return self.a != other.a

    def __gt__(self, other):
        return self.a > other.a

    def __ge__(self, other):
        return self.a >= other.a"""

    ctx = code_to_graph(source)
    assert_equals_snapshot(ctx.g, 1)
