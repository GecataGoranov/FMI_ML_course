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
    

def trace(value):
    nodes = {value}
    edges = set()

    if value._prev is None:
        return nodes, edges
    
    for prev in value._prev:
        nodes.add(prev)
        edges.add((prev, value))
        n, e = trace(prev)
        nodes = nodes.union(n)
        edges = edges.union(e)

    return nodes, edges


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    
    nodes, edges = trace(x)
    print('x')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(y)
    print('y')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(z)
    print('z')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(result)
    print('result')
    print(f'{nodes=}')
    print(f'{edges=}')


if __name__ == "__main__":
    main()