---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Basic pytorch stuff

```python
!jupytext --to markdown "Basic pytorch.ipynb"
```

```python
import numpy as np
import torch
import torch.nn as nn
```

## Make a 3d tensor 
pay attention that torch.int = int32 and simple int = int64

```python
y = torch.tensor([
     [[1, 2, 3],
      [4, 5, 6]],
     [[1, 2, 3],
      [4, 5, 6]],
     [[1, 2, 3],
      [4, 5, 6]]], 
    dtype = torch.int)
print(y, "\n\n", y.shape)
```

## Making summation over different dimentions 

```python
y.sum(dim=0)
```

```python
y.sum(dim=1)
```

```python
y.sum(dim=2)
```

## Float tensors

```python
x = torch.FloatTensor(2,3)
print(x, x.dtype)
```

```python
np_array = np.random.random((2,3)).astype(float)
np_array
```

```python
x1 = torch.FloatTensor(np_array)
x2 = torch.randn(2,3)
print(x1, x1.dtype, "\n\n",x2, x2.dtype)
```

## Integer tensors

```python
int_tensor = torch.arange(4, dtype=torch.int)
int_tensor, int_tensor.dtype
```

```python
int_tensor.view(2,2)
```

```python
torch.sum(y, dtype=torch.int)
```

```python
int_tensor.svd
```

```python
e = torch.exp(int_tensor)
e
```

```python
s*e
```

```python
torch.matmul(x1,x2.t())
```

```python
np.matmul(x1.numpy(), x2.t().numpy())
```

```python
x2.dtype
```

```python
x1.dtype
```

```python
x1 = torch.as_tensor(x1, dtype = torch.float64)
```

```python
torch.matmul(x1.float(), x1.float().t()).dtype
```

```python
x = torch.randn(3,2)
```

```python
cuda0 = torch.device('cuda:0')
```

```python
x.to(torch.float).dtype
```

```python
torch.FloatTensor().dtype
```

```python
x
```

```python
try:
    x.numpy()
except RuntimeError as e:
    print(e)
```

```python
torch.mm(torch.ones(2,3), x.cpu())
```

```python
x = torch.arange(0,4).float().requires_grad_(True)
x
```

```python
y = x**2
```

```python
y.sum()
```

```python
y.sum().backward()
```

```python
x.grad
```

```python
net = torch.nn.Linear(4,2)
```

```python
net
```

```python
f = torch.arange(0,4).float()
f
```

```python
y = net(f)
```

```python
y
```

```python
for param in net.parameters():
    print (param)
```

##  Моя первая нейронная сетка

```python
help(torch.nn.Module)
```

```python
class MyNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size,2)
        self.layer3 = torch.nn.Sigmoid()
        
    def forward(self, input_val):
        h = input_val
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        return h
    
    def print_params(self):
        for item in self.parameters():
            print(item)
```

```python
net = MyNet(4,16)
```

```python
net.print_params()
```

```python
net.forward(torch.rand(4))
```

```python
class MyNet2(torch.nn.Sequential):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(nn.Linear(input_size, hidden_size), 
                         nn.Linear(hidden_size, output_size), 
                         nn.Softmax())
    def print_params(self):
        for item in self.parameters():
            print(item)
```

```python
def make_net2(input_size, hidden_size, output_size):
    return nn.Sequential(nn.Linear(input_size, hidden_size), 
                             nn.Linear(hidden_size, output_size), 
                             nn.Sigmoid())
```

```python
ttt = MyNet2(4,16, 10)
```

```python
net2 = make_net2(4,16,10)
```

```python
net2.forward(torch.ones(4))
```

```python
net2
```

```python
ttt.forward(torch.ones(4))
```

```python
ttt.print_params()
```

```python
import matplotlib.pyplot as plt
```

```python
%matplotlib inline
```

```python
x = np.arange(-100,100,0.5)
above_zero = (x >=0).astype(int)
below_zero = (x<0).astype(int)
```

```python
y =  x*above_zero + 0.5*below_zero*x
```

```python
plt.figure(figsize=(10,5))
plt.axis('equal')
plt.plot(x,y)
plt.show()
```

```python
lin = nn.Linear(4)
```

```python
target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
```

```python
output = torch.full([10, 64], 0.999)  # A prediction (logit)
pos_weight = torch.ones([64])  # All weights are equal to 1
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion(output, target)  # -log(sigmoid(0.999))
```

```python
output
```

```python
target
```

```python

```
