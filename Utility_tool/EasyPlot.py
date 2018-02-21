# for easy plotting

import matplotlib as mpl

import numpy as np
import matplotlib.pyplot as plt


for i in range(0,10):
    x=i
    y=np.random.randint(0,9)
    print(x)
    print(y)
    print("=====")
    plt.plot(x,y)
plt.show()