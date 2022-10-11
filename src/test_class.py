class TestClass:
    def __init__(self, foo, bar):
        self.foo = foo
        self.bar = bar

    def __repr__(self):
        return repr(f"TestClass foo= {self.foo} bar={self.bar}")
