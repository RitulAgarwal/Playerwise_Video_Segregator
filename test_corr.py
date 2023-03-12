import numpy as np 
a = np.random.rand(6,144)
b = np.random.rand(9,144)

d = np.corrcoef(a, b)
print(d.shape)