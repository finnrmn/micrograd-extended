import random
from enum import Enum
from .engine import Value

class Activation(Enum):
    ReLu = "ReLu"
    Tanh = "Tanh"
    Linear = "Linear"

class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
        
    def parameters(self):
        return []
    
class Neuron(Module):
    
    def __init__(self, nin, activation : Activation = Activation.ReLu):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.activation = activation
    
    def __call__(self, x):
        sum = Value(0.0)
        for i in range(len(x)):
            sum += self.w[i] * x[i]
        sum += self.b
        
        if self.activation == Activation.ReLu:
            return sum.relu()
        elif self.activation == Activation.Tanh:  
            return sum.tanh()
        elif self.activation == Activation.Linear:
            return sum
        
    def parameters(self):
        return self.w + [self.b] 
    
    def __repr__(self):
        return f"{self.activation.value}-Neuron({len(self.w)})"

class Layer(Module):
    
    def __init__(self, nin, nout, activation: Activation = Activation.ReLu, **kwargs):
        self.neurons = [Neuron(nin,activation, **kwargs) for _ in range(nout)]
    
    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        nin = len(self.neurons[0].w)
        return f"Layer({nin}, {len(self.neurons)}, {self.neurons[0].activation.value})"
    

class MLP(Module):
    
    def __init__(self, *layers: Layer):
        self.layers = list(layers)
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(l) for l in self.layers)}]"