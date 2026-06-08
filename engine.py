'''
custom autograd engine from scratch using 1st principles
'''


import numpy as np
import pandas as pd

# just like always the function is called tensor , stores -> the actual value of the node
# the children of that node , the operation to make that node 
class Tensor:
    def __init__(self, data: int, label : str, _children = (), _op : str = ""):
        self.data = data
        self.label = label
        self._children = set(_children)
        self._op = _op
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value({self.label}, data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other , Tensor) else Tensor(
                data = other , 
                label = str(other)
                )

        out = Tensor(
            data = self.data + other.data,
            label = "", 
            _children = (self, other),
            _op = "+"
        )

        def _backward():
            self.grad = out.grad * 1
            other.grad = out.grad * 1

        self._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other , Tensor) else Tensor(
                data = other , 
                label= str(other)
                )

        out = Tensor(
            data = self.data + other.data,
            label= "",
            _children = (self , other),
            _op = "*"
        )
        return out
    
    def __rmul__(self, other):
        return self + other
    