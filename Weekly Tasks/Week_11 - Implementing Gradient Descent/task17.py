import graphviz
import numpy as np


val_letter = 'a'

def get_letter():
    global val_letter
    return val_letter

def increment_letter():
    global val_letter
    val_letter = chr(ord(val_letter) + 1)


class Value:
    def __init__(self, data, prevs=None, op=None, grad=np.nan, name=None):
        self.data = data
        self._prev = prevs
        self._op = op
        if name is None:
            self.name = get_letter()
            increment_letter()
        else:
            self.name = name
        self.grad = grad


    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        return Value(self.data + other.data, [self, other], "+")
    
    def __mul__(self, other):
        return Value(self.data * other.data, [self, other], "*")
    
    def tanh(self):
        result = (np.exp(2 * self.data) - 1) / (np.exp(2 * self.data) + 1)
        return Value(result, [self], "tanh")
    
    def backward(self):
        if self.grad is np.nan:
            self.grad = 1.0
        if self._prev is None:
            return
        else:
            if self._op == "+":
                for prev in self._prev:
                    if prev.grad is np.nan:
                        prev.grad = self.grad
                    else:
                        prev.grad = self.grad + prev.grad
            if self._op == "*":
                self._prev[0].grad = self.grad * self._prev[1].data
                self._prev[1].grad = self.grad * self._prev[0].data
            if self._op == "tanh":
                self._prev[0].grad = 1 - ((np.exp(2 * self.data) - 1) / (np.exp(2 * self.data) + 1)) ** 2
            for prev in self._prev:
                prev.backward()


def escape_html_label(label: str) -> str:
    """Escape any special characters in the label for Graphviz"""
    return label.replace('<', '&lt;').replace('>', '&gt;')


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    letter = 'a'
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        label=f"{n.name} | data: {n.data} | grad: {n.grad}"
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
    x = Value(10, name="x")

    y = x + x
    y.backward()

    draw_dot(y).render(directory='./graphviz_output', view=True)


if __name__ == "__main__":
    main()