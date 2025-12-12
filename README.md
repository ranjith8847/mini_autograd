# mini_autograd

This project implements scalar-based automatic differentiation,   computational graph construction, and backpropagation from scratch.   Includes a simple neural network module and training example.   Great for understanding how PyTorch’s autograd works internally.

# Introduction

A lightweight, educational automatic differentiation engine built from scratch.  
It mirrors the core ideas behind modern deep-learning frameworks: tensor operations, backpropagation, and computational graphs — all in a minimal and readable codebase.

---

## 🚀 Features

- Scalar & Tensor support (depending on your implementation)
- Dynamic computation graph
- Reverse-mode autodiff (backpropagation)
- Operator overloading for smooth mathematical expressions
- Gradient accumulation
- Support for common ops: +, -, *, /, **, tanh, exp
- Zero dependencies (pure Python)
- Easy to extend with new operators or activation functions

---

## 📁 Project Structure

```
📦 mini-autograd  
 ├── autograd.ipynb
 └── README.md  
```

---

## 🧠 Core Concepts

### 1. Value / Tensor Object
Each number stores:  
- The computed value  
- A gradient  
- A reference to the operation that produced it  
- Pointers to children in the computation graph  

### 2. Building the Graph
```python
c = a * b  
d = c + 3  
out = d.tanh()  
```

### 3. Backpropagation
```python
out.backward()  
```

---

## ✨ Example Usage

```python
from engine import Value

x = Value(2.0)  
w = Value(-3.0)  
b = Value(1.0)

y = w * x + b  
y.backward()

print(y.data)  
print(x.grad)  
print(w.grad)  
print(b.grad)
```

---

## 🧪 Example: mini MLP

```python
from engine import Value  
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            [Neuron(sz[i]) for _ in range(sz[i+1])]
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = [n(x) for n in layer]
        return x

model = MLP(3, [4, 4, 1])
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5]]
ys = [1.0, -1.0]

for _ in range(50):
    ypred = [model(x)[0] for x in xs]
    loss = sum((yp - yt)**2 for yp, yt in zip(ypred, ys))
    for p in model.layers:
        for n in p:
            for w in n.w:
                w.grad = 0
            n.b.grad = 0
    loss.backward()
    for p in model.layers:
        for n in p:
            for w in n.w:
                w.data -= 0.1 * w.grad
            n.b.data -= 0.1 * n.b.grad
```

---

## 📌 Goals

- Understand how autodiff systems work internally  
- Provide a minimal foundation for experimentation  
- Serve as a reference for learners wanting to build their own deep-learning engines  

---

## 🛠️ Extending

Add:  
- ReLU, sigmoid, softmax  
- 2D/ND tensors  
- Optimizers (SGD, Adam)  
- Graph visualization  
- GPU kernels  


