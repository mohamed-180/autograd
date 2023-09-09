import torch
import numpy as np
from time import time



x = torch.linspace(-3, 3, 1000)
y = 3.45*x + 1.3 + torch.randn(1000)


t = time()
w = torch.randn(1, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
for i in range(1000):
	l = loss(w, b, x, y)
	l.backward()
	with torch.no_grad():
	    w -= .005*w.grad
	    b -= .005*b.grad 
	w.grad.zero_()
	b.grad.zero_()
print(time() - t)

