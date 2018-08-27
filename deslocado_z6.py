import numpy as np
import matplotlib.pyplot as plt
import math as math
import networkx as nx
import cmath as cm

class node:
    def __init__(self, rho, theta, ang_freq, neighbors):
        self.rho = rho
        self.theta = theta
        self.rn = 0
        self.tn = 0
#        self.b = b
        self.w = ang_freq
        self.n = neighbors

    def f(self):
        return (a*abs(self.z)**4+b*abs(self.z)**2+c+self.w*1j)*self.z
        
    def eq_rho(self, eps):
        f = (a *  self.rho ** 4 + b * self.rho ** 2 + c) * (self.rho - delta)
        f += d * (eps.real * np.cos(self.theta) + eps.imag * np.sin(self.theta) )
        return f   
    
    def eq_theta(self, eps):
        f = self.w * self.rho + d * (eps.imag * np.cos(self.theta) - eps.real * np.sin(self.theta) )
        return f
    
    def fut(self):
        eps = np.random.normal(0,1.)+np.random.normal(0,1.)*1j
        soma_rho=0.0
        soma_theta=0.0
        for i,w in enumerate(self.n):
            soma_rho += nod[w].rho*np.cos(self.theta-nod[w].theta)
            soma_theta += nod[w].rho*np.sin(nod[w].theta-self.theta)
        self.rn = (self.eq_rho(eps) + beta * soma_rho) * dt
        self.tn = (self.eq_theta(eps) + beta * soma_theta) * dt
    
    def mov(self):
        self.rho += self.rn
        self.theta += self.tn


#Constructing connection network
global a,b,beta,c,d,dt,noise,N,delta
a=-1
b=2.
c=-0.9
d = 0#.1
delta=0.1
icto=math.sqrt((-b+math.sqrt(b**2-4*a*c))/(2*a))
beta=0
dt=0.01
noise=0.04
N=20
T=100
Nij=N*(N-1)//2

count=0
for p in [0.4]:#, 0.5, 0.6, 0.7, 0.8, 0.9]:
#    bo=5
#    bth=2*math.sqrt(a*c)
#    tau=1000
#    zo=-0.5*bth/a

    g=nx.Graph()
    g.add_node(0)
    g.add_node(1)
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(N,p)
    
    node_neigh_list=[]
    for i in nx.nodes(g):
        list_neigh=[]
        for j in nx.all_neighbors(g, i):
            list_neigh.append(j)
        node_neigh_list.append(list_neigh)


    #initialize N nodes
    nod=list(node(np.random.normal(0,1.), np.random.normal(0,2*np.pi), np.random.uniform(-0.2,0.2), node_neigh_list[i]) for i in range(N))

    #Node evolution
    t=0
    y=[]
    for n in nod:
        y += [[n.rho * np.sin(n.theta)]]
    #d=[]
    #b=[]
    while(t<T):
        t+=dt
#        if(t%10<dt):
#            print "t=",int(t)
        list(map(lambda i:i.fut(), nod))
        list(map(lambda i:i.mov(), nod))
        
        for i,n in enumerate(nod):
            y[i] += [n.rho * np.sin(n.theta)]

            
plt.plot(y[0])
plt.plot(y[1])
plt.plot(y[2])
plt.plot(y[3])
plt.show()

