import graphviz


val_letter = 'a'

def get_letter():
    global val_letter
    return val_letter

def increment_letter():
    global val_letter
    val_letter = chr(ord(val_letter) + 1)


class Value:
    def __init__(self, data, prevs=None, op=None, grad=0):
        self.data = data
        self._prev = prevs
        self._op = op
        self.letter = get_letter()
        self.grad = grad

        increment_letter()

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        return Value(self.data + other.data, {self, other}, "+")
    
    def __mul__(self, other):
        return Value(self.data * other.data, {self, other}, "*")


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={
                           'rankdir': 'LR'})

    nodes, edges = trace(root)
    letter = 'a'
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        if n == root:
            label=f"L | data: {n.data} | grad: {n.grad}"
        else:
            label=f"{n.letter} | data: {n.data} | grad: {n.grad}"
        dot.node(name=uid, label=label, shape='record')
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

        letter = chr(ord(letter) + 1)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


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
    t=  Value(-2.0)
    u = x * y + z
    result = u * t
    
    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(result).render(directory='./graphviz_output', view=True)


if __name__ == "__main__":
    main()