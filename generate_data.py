from typing import Mapping
import numpy as np 
import math,os
import matplotlib.pyplot as plt
from tqdm import tqdm
if __name__ == '__main__':
    x = np.arange(-np.pi,np.pi,0.001)

    a = 1
    b = 2
    c = 2
    d = 2
    
    y = a*np.cos(b*x)+c*np.sin(d*x)

    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Composite function')
    plt.axis()
    plt.show()
    f = open('a1b2c2d2.csv','w')
    for it,data in tqdm(enumerate(x)):
        context = '%.6f,%.6f\n'%(x[it],y[it])
        f.writelines(context)
    f.close()

