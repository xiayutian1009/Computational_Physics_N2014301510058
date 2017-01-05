from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import random

steps=np.linspace(0,100,101)
x_c=np.zeros(101)
x_t=np.zeros(101)
x2 = np.zeros(101)


for i in range(100):
    rad=np.random.random()
    if rad <= 0.5:
        x_c[i] = x_c[i] - 1   
        x_t[i]=x_t[i-1]-1
        x2[i]=x_t[i]**2
    else:
        x_c[i] = x_c[i] + 1
        x_t[i]=x_t[i-1]+1
        x2[i]=x_t[i]**2
		
pl.scatter(steps,x2)
pl.xlim(0,100)
pl.ylabel("x")
pl.xlabel("t")
pl.show()



from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import random

steps=np.linspace(0,100,101)
x_c=np.zeros(1000)
x_av=np.zeros(101)


for i in range(100):
   for j in range(1000):
        rad=np.random.random()
        if rad <= 0.5:
            x_c[j] = x_c[j] - 1   
        else:
            x_c[j] = x_c[j] + 1
		
   total=sum(x_c)
   x_av[i]=total/1000
		
pl.scatter(steps,x_av)
pl.ylim(-10,10)
pl.xlim(0,100)
pl.ylabel("average x")
pl.xlabel("t")
pl.show()

from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import random

steps=np.linspace(0,100,101)
x_c=np.zeros(1000)
x_av=np.zeros(101)
x_t=np.zeros(1000)
x2=np.zeros(1000)

for i in range(100):
    for j in range(1000):
        rad=np.random.random()
        if rad <= 0.5:
            x_c[j] = x_c[j] - 1      
            x2[j]=x_t[j]**2	
        else:
            x_c[j] = x_c[j] + 1
        x2[j]=x_c[j]**2			
    total=sum(x2)
    x_av[i]=total/1000
		
pl.scatter(steps,x_av)
pl.plot(steps,steps,"r")
pl.ylim(0,120)
pl.xlim(0,100)
pl.ylabel("x^2")
pl.xlabel("t")
pl.title("Random walk in one dimension")
pl.show()


from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import random

steps=np.linspace(0,100,101)
x_c=np.zeros(101)


for i in range(100):
    rad=np.random.uniform(-1,1)
    x_c[i]=x_c[i-1]+rad 
  
		
pl.scatter(steps,x_c)
pl.xlim(0,100)
pl.ylabel("x")
pl.xlabel("t")
pl.title("Random walk in one dimension of random length")
pl.show()


from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import random

steps=np.linspace(0,100,101)
x_c=np.zeros(1000)
x_av=np.zeros(101)
x_t=np.zeros(101)
x2=np.zeros(1000)

for i in range(100):
    for j in range(1000):
        rad=np.random.uniform(-1,1)
        x_c[j]=x_c[j]+rad
        x2[j]=x_c[j]**2			
    total=sum(x2)
    x_av[i]=total/1000
    x_t[i]=steps[i]/3
		
pl.scatter(steps,x_av)
pl.plot(steps,x_t,"r")
pl.ylim(0,40)
pl.xlim(0,100)
pl.ylabel("x^2")
pl.xlabel("t")
pl.title("Random walk in one dimension of random length")
pl.show()


from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import random

steps=np.linspace(0,100,101)
x_c=np.zeros(101)
x_t=np.zeros(101)
x2 = np.zeros(101)


for i in range(100):
    rad=np.random.random()
    if rad <= 0.3:
        x_c[i] = x_c[i] - 1   
        x_t[i]=x_t[i-1]-1
    else:
        x_c[i] = x_c[i] + 1
        x_t[i]=x_t[i-1]+1
		
pl.scatter(steps,x_t)
pl.xlim(0,100)
pl.ylabel("x")
pl.xlabel("t")
pl.show()

from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import random

steps=np.linspace(0,100,101)
x_c=np.zeros(1000)
x_av=np.zeros(101)
x_t=np.zeros(1000)
x2=np.zeros(1000)


for i in range(100):
    for j in range(1000):
        rad=np.random.random()
        if rad <= 0.3:
            x_c[j] = x_c[j] - 1      
            x2[j]=x_t[j]**2	
        else:
            x_c[j] = x_c[j] + 1
        x2[j]=x_c[j]**2			
    total=sum(x2)
    x_av[i]=total/1000

x_c1=np.zeros(1000)
x_av1=np.zeros(101)
x_t1=np.zeros(1000)
x21=np.zeros(1000)	

for i in range(100):
    for j in range(1000):
        rad=np.random.random()
        if rad <= 0.4:
            x_c1[j] = x_c1[j] - 1      
            x21[j]=x_t1[j]**2	
        else:
            x_c1[j] = x_c1[j] + 1
        x21[j]=x_c1[j]**2			
    total=sum(x21)
    x_av1[i]=total/1000
	
pl.scatter(steps,x_av1,color="r",label="p=0.6")
pl.scatter(steps,x_av,label="p=0.7")
pl.ylim(-20,2000)
pl.xlim(0,100)
pl.ylabel("x^2")
pl.xlabel("t")
pl.title("Random walk in one dimension--unequal probability")
pl.legend(loc = 'upper left') 
pl.show()


import numpy as np
import matplotlib.pyplot as pl
from pylab import *
from math import *
import mpl_toolkits.mplot3d

D=1
dt=0.5
dx=1
length=100
time=50
k=100
y=[[0 for i in range(length)]for n in range(time)]#i represents x, n represents t

for i in range(length):
    y[0][i]=exp(-k*(i*dx-50)**2)


for n in range(time-2):
    for i in range(1,length-1):
        y[n+1][i]=0.5*(y[n][i+1]+y[n][i-1])  	
y=array(y)
add=array([1 for i in range(length)])
for n in range(0,time,20):
    yp=y[n]+add*n/20
pl.plot(range(length),y)
pl.show()


from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import random

steps=np.linspace(0,1000,1001)
x_c=np.zeros(1001)
y_c=np.zeros(1001)


for i in range(1000):
    rad=np.random.random()
    if rad <= 0.25:
        x_c[i] = x_c[i-1] - 1  
        y_c[i] = y_c[i-1]   		
    elif 0.25<rad<=0.5:
        x_c[i] = x_c[i-1] + 1
        y_c[i] = y_c[i-1]
    elif 0.5<rad<=0.75:
        x_c[i] = x_c[i-1] 
        y_c[i] = y_c[i-1]+1
    else :
        x_c[i] = x_c[i-1] 
        y_c[i] = y_c[i-1]-1	
pl.scatter(x_c,y_c)
#pl.xlim(0,100)
pl.ylabel("x")
pl.xlabel("y")
pl.show()

from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import random

steps=np.linspace(0,1000,1001)
x_c=np.zeros(10001)
y_c=np.zeros(10001)
x_v=np.zeros(1001)
y_v=np.zeros(1001)

for i in range(1000):
    for j in range(10000):
        rad=np.random.random()
        if rad <= 0.25:
            x_c[j] = x_c[j-1] - 1  
            y_c[j] = y_c[j-1]   		
        elif 0.25<rad<=0.5:
            x_c[j] = x_c[j-1] + 1
            y_c[j] = y_c[j-1]
        elif 0.5<rad<=0.75:
            x_c[j] = x_c[j-1] 
            y_c[j] = y_c[j-1]+1
        else :
            x_c[j] = x_c[j-1] 
            y_c[j] = y_c[j-1]-1	
    tmpx=x_c[9999]
    tmpy=y_c[9999]
    x_v[i]=tmpx
    y_v[i]=tmpy	
pl.scatter(x_v,y_v,color="y")
pl.xlim(-100,100)
pl.ylim(-100,100)
pl.ylabel("y")
pl.xlabel("x")
pl.title("0ne thousand cream in coffee t=10000")
pl.show()



from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import random
import math

steps=np.linspace(0,1000,1001)
x_c=np.zeros(10001)
y_c=np.zeros(10001)
x_v=np.zeros(1001)
y_v=np.zeros(1001)
pi=np.zeros((10,10))
lnp=np.zeros((10,10))
entropy=0

for i in range(1000):
    for j in range(10000):
        rad=np.random.random()
        if rad <= 0.25:
            x_c[j] = x_c[j-1] - 1  
            y_c[j] = y_c[j-1]   		
        elif 0.25<rad<=0.5:
            x_c[j] = x_c[j-1] + 1
            y_c[j] = y_c[j-1]
        elif 0.5<rad<=0.75:
            x_c[j] = x_c[j-1] 
            y_c[j] = y_c[j-1]+1
        else :
            x_c[j] = x_c[j-1] 
            y_c[j] = y_c[j-1]-1	
    tmpx=x_c[49]
    tmpy=y_c[49]
    x_v[i]=tmpx
    y_v[i]=tmpy
    for k in range (10):
        for f in range (10):
            if 100-20*k>=x_v[i]>100-20*(k+1) and 100-20*f>=y_v[i]>100-20*(f+1):
                pi[k][f]=pi[k][f]+1/1000
for g in range (10):
        for m in range (10):
            if pi[g][m]>0:
                entropy=entropy-pi[g][m]*math.log(pi[g][m])
print entropy
import matplotlib.pyplot as pl
x=[0,10,50,100,200,400,600,800,1000,2000,3000,4000,5000,6000]
y=[1.036771,1.355669,1.377068,1.439547,1.789050,2.340163,2.662033,2.863047,3.079114,3.693579,3.920797,3.981498,3.858740,3.974414]
pl.plot(x,y,"r")
pl.ylim(0,5)
pl.xlim(0,6000)
pl.ylabel("entropy")
pl.xlabel("time")
pl.title("Entropy versus time")
pl.show()


from __future__ import division
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
dx=1
dt=0.5
D=1
time=10
length=100
den=np.zeros(100)
den1=np.zeros((100,100))
x=range(100)
y=range(100)
density=[[0 for i in range(100)]for n in range(length)]
for j in range(100):
    for k in range(length):
        density[49][0]=1
    else:
        density[j][0]=0
for n in range(length-1):		
    for i in range(1,99):
        density[i][n+1]=0.5*(density[i+1][n]+density[i-1][n])
for g in range(100):
    for m in range(length):
        den[g]=density[g][49]
for l in range(1,99):
    if den[l]==0:
        den[l]=0.5*(den[l-1]+den[l+1])


pl.plot(x,den)
pl.plot(y,den)
pl.xlabel("x")
pl.ylabel("density")
pl.title("Diffusion in one dimension,t=100dt")
#pl.ylim(0,0.3)
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

x = np.linspace(0,99,100)
y = np.linspace(0,99,100)
X, Y = np.meshgrid(x, y)
Z = density


fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_xlabel('time')
ax.set_ylabel('x')
ax.set_zlabel('density')
ax.set_title('Diffusion versus time')
pl.show()

		

