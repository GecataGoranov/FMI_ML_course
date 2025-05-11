class Value:
    def __init__(self, data, prevs=None, op=None):
        self.data = data
        self._prev = prevs
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        return Value(self.data + other.data, {self, other}, "+")
    
    def __mul__(self, other):
        return Value(self.data * other.data, {self, other}, "*")
    

def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._op)

if __name__ == "__main__":
    main()