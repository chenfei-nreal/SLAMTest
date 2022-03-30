import matplotlib.pyplot as plt
import numpy as np
from IPython import embed


us = []
vs = []
with open("./build/errors.txt") as f:
    lines = f.readlines()
    for line in lines:
        u, v = line.split()
        us.append(float(u))
        vs.append(float(v))
us = np.array(us)
vs = np.array(vs)


plt.scatter(us, vs)
plt.show()
