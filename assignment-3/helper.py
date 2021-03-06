import numpy as np 
import matplotlib.pyplot as plt

xx=np.arange(1,6)
nn=[0.000000,97.255005,695.342163,3591.2110,43094.5500]
nn=np.array(nn)/1000.0
plt.plot(xx,nn,linestyle='--', marker='o', color='b')

for x,n in zip(xx,nn):
    plt.text(x-0.15,n+0.15,s=f'value {n:.2f}')

plt.xlabel("N value in n-count-gram")
plt.ylabel("Run time")
plt.savefig("q1.png")
plt.show()