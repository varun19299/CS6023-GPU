import numpy as np 
import matplotlib.pyplot as plt

xx=np.arange(2,6)
nn=[64357.871094,32513.554688,16457.830078,12096.892578]/1000.0
nn=np.array(nn)
plt.plot(xx,nn,linestyle='--', marker='o', color='b')

for x,n in zip(xx,nn):
    plt.text(x-0.15,nn+0.15,s=f'value {n:.2f}')

plt.xlabel("Threads per Block in 2^n")
plt.ylabel("Run time")
plt.savefig("q2.png")
plt.show()