import numpy as np 
import matplotlib.pyplot as plt

xx=np.arange(2,6)
nn=[15989.210938,4473.976562,4380.023438,4042.729004]
nn=np.array(nn)/1000.0
plt.plot(xx,nn,linestyle='--', marker='o', color='b')

for x,n in zip(xx,nn):
    plt.text(x-0.15,n+0.15,s=f'value {n:.2f}')

plt.xlabel("TileSize in 2^n * 2^n")
plt.ylabel("Run time")
plt.savefig("q5.png")
plt.show()