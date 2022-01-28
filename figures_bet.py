# -*- coding: utf-8 -*-
"""
Created on Wed May  5 18:25:34 2021

@author: goell
"""

import scipy.special as sp
from math import log, floor
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import binom


#Initial situation taken from "Greed"
u = 1.5 # if the coin shows heads
d = 0.6 # if the coin shows tails
a = 100 # initial capital/points
n = 100 # number of coin tosses
p = 0.5 # probability for "heads"

# Definition of k_0 (threshold between winning and losing states )
def k0(n,u,d):
    return floor(-n*(log(d)/(log(u)-log(d))))

# Expected net profit (bounded)
def g(a,n,u,d,p):
    v1 = 0
    c = k0(n,u,d)
    for k in range(0,c+1):
        v1 += binom(n,k) * (d**n * (u/d) ** k - 1) * (p ** k) * ((1-p) ** (n-k))
    v2 = 0
    for k in range(c+1,n+1):
        v2 += binom(n,k) * (p ** k) * ((1-p) ** (n-k))
    return a * (v1+v2)


# Computation of the expected net profit for n between 1 and 200
n_vec = range(1,201) # reminder: range(1,m) = {1,...,m-1} 
g_vec_n_var = []
for m in n_vec:
    g_vec_n_var.append(g(100,m,1.5,0.6,0.5))

# Scatter-plot of the expected net profit 
plt.figure()
plt.scatter(n_vec, g_vec_n_var, s=1)
plt.hlines(0,0,200,linestyles = "dashed", linewidth = 0.75)
plt.xlabel(r"$n\in \{1,\ldots,200\}$")
plt.ylabel(r"expected net profit")
plt.savefig("expected_net_profit_n_var.eps", format = "eps")

# Illustration of the convergence of the expected net profit towards -a
n_vec2 = range(1,301)
g_vec_konv = []
for m in n_vec2:
    g_vec_konv.append(g(100,m,1.6,0.5,0.5))

plt.figure()
plt.plot(n_vec2,g_vec_konv)
plt.hlines(-100,0,300,linestyles = "dashed", linewidth = 0.75)
plt.xlabel(r"$n\in \{1,\ldots,300\}$")
plt.ylabel(r"expected net profit")
plt.savefig("expected_net_profit_convergence.eps", format = "eps")

# Expected net profit for a,n fixed, varying u and 3 diff. values of d
u_vec = np.linspace(1,2.7)
g_vec_u_055 = [] # d=0.55
g_vec_u_06 = [] # d=0.6
g_vec_u_065 = [] # d=0.65
a=100
for U in u_vec:
    g_vec_u_055.append(g(a,n,U,0.55,p))
    g_vec_u_06.append(g(a,n,U,0.6,p))
    g_vec_u_065.append(g(a,n,U,0.65,p))
 
# Plot: expected net profit for varying u
plt.figure()
plt.hlines(0,1,2.7, linestyles = "dashed", linewidth = 0.75)
plt.plot(u_vec, g_vec_u_055,label=r"$d=0.55$")
plt.plot(u_vec, g_vec_u_06,label=r"$d=0.6$")
plt.plot(u_vec, g_vec_u_065,label=r"$d=0.65$")
plt.plot(1.5, g(a,n,1.5,0.6,p),'o', markersize = 3, label=r"$d=0.6,\, u=1.5$ (Elsberg)")
plt.xlabel(r"$u\in [1,2.7]$")
plt.ylabel(r"expected net profit")
plt.legend()
plt.savefig("expected_net_profit_u_var.eps", format = "eps")


# Expected net profit for a,n fixed, varying d and 3 diff. values of u
d_vec = np.linspace(0.4,0.8)
g_vec_d_145 = [] # u=1.45
g_vec_d_15 = [] # u=1.5
g_vec_d_155 = [] # u=1.55
a=100
for D in d_vec:
    g_vec_d_145.append(g(a,n,1.45,D,p))
    g_vec_d_15.append(g(a,n,1.5,D,p))
    g_vec_d_155.append(g(a,n,1.55,D,p))
    
# Plot: expected net profit for varying d
plt.figure()
plt.hlines(0,0.4,0.8,linestyles = "dashed", linewidth = 0.75)
plt.plot(d_vec, g_vec_d_145,label=r"$u=1.45$")
plt.plot(d_vec, g_vec_d_15,label=r"$u=1.5$")
plt.plot(d_vec, g_vec_d_155,label=r"$u=1.55$")
plt.plot(0.6, g(a,n,1.5,0.6,p),'o', markersize = 3,  label=r"$d=0.6,\, u=1.5$ (Elsberg)")
plt.xlabel(r"$d\in [0.4,0.8]$")
plt.ylabel(r"expected net profit")
plt.legend()
plt.savefig("expected_net_profit_d_var", format = "eps")


# Redefine expected net profit in terms of only u,d
def g1(u,d):
    return g(a,n,u,d,p)

# Function to find u, d so that the game is fair
def null_u(d):
    return fsolve(lambda u: g1(u,d), 1.6)


# Computation of fair u,d
d_vec2 = np.linspace(0.5,0.69)
null_vec = []
for D in d_vec2:
    null_vec.append(null_u(D))

# Plot: fair pairs d, u
plt.figure()
plt.plot(d_vec2, null_vec, label = r"up factor $u$")
plt.plot(d_vec2, 1/d_vec2, label = r"$u=\frac{1}{d}$ (Theorem 2)")
plt.plot(0.6, 1.5, 'o', label = r"Elsberg")
plt.ylabel(r"up factor $u$ (fair game)")
plt.xlabel(r"down factor $d$")
plt.legend()
plt.savefig("up-factor_fair.eps", format = 'eps')

# Simulation of the score over time

# Score after each round
rep = range(1,n+1) 
sim = []
y = a
for k in rep:
    s = np.random.binomial(1,p)
    if s == 1: # heads
        y *= u
    else: # tails
        y *= d
    sim.append(y)

trenn = " "
liste = ["final score:" , str(round(sim[n-1],2))]
b = trenn.join(liste)

# Plot: score over time
plt.figure()
plt.scatter(rep,sim, s=2, label = r"score after $\ell$ rounds")
plt.hlines(100, 0, 100, linestyles = "dashed", linewidth = 0.75, label = r"initial value $100$")
plt.plot(100, sim[n-1], "r.", label = b)
plt.xlabel(r"$\ell\in \{1,\ldots, 100\}$")
plt.legend(loc = "best")
plt.savefig("score.eps", format = "eps")
plt.savefig("score")
print(sim[n-1])

# Simulation: score after n rounds

# Computation: final score after 100 rounds (100 times)
rep2 = range(1, 101) # 100 simulations
payoff = []
g_end = 0
for j in rep2:
    y = 100
    for k in rep:
        s = np.random.binomial(1,p)
        if s == 1: # heads
            y *= u
        else: # tails
            y *= d
    if y >= a: # win at the end of the game
        payoff.append(a)
        g_end += 1
    else: # loss at the end of the game
        payoff.append(y-a)
print(g_end)

# Plot: simulation of 100 final scores 
plt.figure()
plt.scatter(rep2, payoff, s=2)
plt.hlines(0, 0, 100, linestyles = "dashed", linewidth = 0.75)
plt.ylabel(r"net profit after $100$ rounds")
plt.xlabel(r"number of simulations")
plt.savefig("simulation_end_punktestand.eps", format="eps")
plt.savefig("simulation_end_punktestand")

# =============================================================================
# Variation of the probability p for heads/tails
 
# Find the fair p
def g3(p):
   return g(a,n,u,d,p)

p_fair = fsolve(g3, 0.58)
print(p_fair)
 
# Expected net profit in terms of p
 
p_vec = np.linspace(0.0001,0.999)
g_vec_p = []
a=100
for P in p_vec:
    g_vec_p.append(g(a,n,u,d,P))

# Plot: expected net profit in terms of p
   
plt.figure()
plt.plot(p_vec, g_vec_p)
plt.hlines(0,0,1, linestyles = "dashed", linewidth = 0.75)
plt.plot(0.5, g(a,n,u,d,0.5), 'o', markersize = 3, label = r"$p = 0.5$ (Elsberg)")
plt.plot(p_fair, g(a,n,u,d,p_fair), 'o', markersize = 3, label = r"$p\approx 0.551$ (fair)")
plt.xlabel(r"$p\in [0,1]$")
plt.ylabel(r"expected net profit")
plt.legend()
plt.savefig("expected_net_profit_p_var.eps", format = "eps")
# =============================================================================

##############################################################################
                               #  Variance # 
##############################################################################                    

# =============================================================================
 
def p0(p,n,u,d): #P(X_n \ge k_0)
    q = 1-p
    k_0 = k0(n,u,d)
    q0 = 0
    for k in range(k_0+1,n+1):
        q0 += sp.binom(n,k)*p**k * q ** (n-k)
    return q0
 
# Summands of the variance
 
def e1(p,n,u,d):
    k_0 = k0(n,u,d)
    q = 1-p
    e1 = 0
    for k in range(0,k_0+1):
        e1 += sp.binom(n,k)*(p*u**2)**k * (q*d**2)**(n-k)
    return e1

def e2(p,n,u,d):
    return 4*p0(p,n,u,d)*(1-p0(p,n,u,d))

def e3(p,n,u,d):
    k_0 = k0(n,u,d)
    q = 1-p
    e3 = 0
    for k in range(0,k_0+1):
        e3 += sp.binom(n,k)**2 * (p*u)**(2*k) * (q*d)**(2*n-2*k)
    return e3
 
def e4(p,n,u,d):
    k_0 = k0(n,u,d)
    q = 1-p
    q0 = p0(p,n,u,d)
    e4 = 0
    for k in range(0,k_0+1):
        e4 += sp.binom(n,k)*(p*u)**k * (q*d)**(n-k)
    return 4 * e4 * q0

def e5(p,n,u,d):
    k_0 = k0(n,u,d)
    q = 1-p
    e5 = 0
    for k in range(0, k_0+1):
        for j in range(0, k_0+1):
            if j!= k:
                e5 += sp.binom(n,k)*sp.binom(n,j)*(p*u)**(k+j) * (q*d)**(2*n-k-j)
            else: 
                pass
    return e5

# Variance (total)
 
def var_fun(p,n,u,d):
    return e1(p,n,u,d) +  e2(p,n,u,d) - e3(p,n,u,d) - e4(p,n,u,d) - e5(p,n,u,d)
 
 
    
n_vec = np.linspace(1,500,120)

var = []
e1_vec = []
e2_vec = []
e3_vec = []
e4_vec = []
e5_vec = []
 
for m1 in n_vec:
    m = floor(m1)
    var.append(var_fun(p,m,u,d))
    e1_vec.append(e1(p,m,u,d))
    e2_vec.append(e2(p,m,u,d))
    e3_vec.append(e3(p,m,u,d))
    e4_vec.append(e4(p,m,u,d))
    e5_vec.append(e5(p,m,u,d))

# Plot: convergence of the total variance as n-> infinity
 
plt.figure()
plt.plot(n_vec, var)
plt.hlines(0,0,500, linestyles = "dashed", linewidth = 0.75)
plt.xlabel(r"$n\in \{1,\ldots,500\}$")
plt.savefig("variance_convergence_elsberg.eps", format = "eps")
 
# Plot: convergence of the summands as n-> infinity 
 
plt.figure()
plt.plot(n_vec, e1_vec, label=r"$v_1(n,u,d,p)$")
plt.plot(n_vec, e2_vec, label=r"$v_2(n,u,d,p)$")
plt.plot(n_vec, e3_vec, label=r"$v_3(n,u,d,p)$")
plt.plot(n_vec, e4_vec, label=r"$v_4(n,u,d,p)$")
plt.plot(n_vec, e5_vec, "-.", label=r"$v_5(n,u,d,p)$")
plt.hlines(0,0,500, linestyles = "dashed", linewidth = 0.75)
plt.xlabel(r"$n\in \{1,\ldots,500\}$")
plt.legend()
plt.savefig("variance_summands_convergence_elsberg.eps", format = "eps")
 

# Variance for varying p
 
n=200
p_vec = np.linspace(0,1,100)
 
varp = []
e1_vecp = []
e2_vecp = []
e3_vecp = []
e4_vecp = []
e5_vecp = []

for p in p_vec:
    varp.append(var_fun(p,n,u,d))
    e1_vecp.append(e1(p,n,u,d))
    e2_vecp.append(e2(p,n,u,d))
    e3_vecp.append(e3(p,n,u,d))
    e4_vecp.append(e4(p,n,u,d))
    e5_vecp.append(e5(p,n,u,d))
 
u=1.5
d=0.6
plt.figure()
plt.plot(p_vec, varp, label = r"$\mathbb{V}(T_n)$")
plt.vlines(np.log(1/d)/np.log(u/d),0,1, linestyles = "dashed", linewidth = 0.75, label = r"$p = \frac{\ln(d^{-1})}{\ln(u d^{-1})}$")
plt.xlabel(r"$p\in [0,1]$")
plt.legend()
plt.savefig("variance_total_p_var.eps", format="eps")
 
plt.figure()
plt.plot(p_vec, e1_vecp, label=r"$v_1(n,u,d,p)$")
plt.plot(p_vec, e2_vecp, label=r"$v_2(n,u,d,p)$")
plt.plot(p_vec, e3_vecp, label=r"$v_3(n,u,d,p)$")
plt.plot(p_vec, e4_vecp, label=r"$v_4(n,u,d,p)$")
plt.plot(p_vec, e5_vecp, "-.", label=r"$v_5(n,u,d,p)$")
plt.vlines(np.log(1/d)/np.log(u/d),0,1, linestyles = "dashed", linewidth = 0.75, label = r"$p = \frac{\ln(d^{-1})}{\ln(u d^{-1})}$")
plt.xlabel(r"$p\in [0,1]$")
plt.legend()
plt.savefig("variance_summands_p_var.eps", format="eps")
 

# Variance for varying u
     
n=200
u_vec = np.linspace(1,3,100)
d=0.6
p=0.5
 
varu = []
e1_vecu = []
e2_vecu = []
e3_vecu = []
e4_vecu = []
e5_vecu = []
 
for u in u_vec:
    varu.append(var_fun(p,n,u,d))
    e1_vecu.append(e1(p,n,u,d))
    e2_vecu.append(e2(p,n,u,d))
    e3_vecu.append(e3(p,n,u,d))
    e4_vecu.append(e4(p,n,u,d))
    e5_vecu.append(e5(p,n,u,d))

plt.figure()
plt.plot(u_vec, varu, label = r"$\mathbb{V}(T_n)$")
plt.vlines(d**(1-1/p),0,1,linestyles = "dashed", linewidth = 0.75, label = r"$u = d^{\frac{p-1}{p}}$")
plt.legend()
plt.xlabel(r"$u\in [1,3]$")
plt.savefig("variance_total_u_var.eps", format = "eps")
 
 
plt.figure()
plt.plot(u_vec, e1_vecu, label=r"$v_1(n,u,d,p)$")
plt.plot(u_vec, e2_vecu, label=r"$v_2(n,u,d,p)$")
plt.plot(u_vec, e3_vecu, label=r"$v_3(n,u,d,p)$")
plt.plot(u_vec, e4_vecu, label=r"$v_4(n,u,d,p)$")
plt.plot(u_vec, e5_vecu, "-.", label=r"$v_5(n,u,d,p)$")
plt.vlines(d**(1-1/p),0,1,linestyles = "dashed", linewidth = 0.75, label = r"$u = d^{\frac{p-1}{p}}$")
plt.legend()
plt.xlabel(r"$u\in [1,3]$")
plt.savefig("variance_summands_u_var.eps", format = "eps")
 

# Variance for varying d
 
n=200
d_vec = np.linspace(0.01,1,100)
u=1.5
p=0.5

vard = []
e1_vecd = []
e2_vecd = []
e3_vecd = []
e4_vecd = []
e5_vecd = []
 
for d in d_vec:
    vard.append(var_fun(p,n,u,d))
    e1_vecd.append(e1(p,n,u,d))
    e2_vecd.append(e2(p,n,u,d))
    e3_vecd.append(e3(p,n,u,d))
    e4_vecd.append(e4(p,n,u,d))
    e5_vecd.append(e5(p,n,u,d))
 
plt.figure()
plt.plot(d_vec, vard, label = r"$\mathbb{V}(T_n)$")
plt.vlines(u**(p/(p-1)),0,1,linestyles = "dashed", linewidth = 0.75, label = r"$d = u^{\frac{p}{p-1}}$")
plt.xlabel(r"$d\in (0,1]$")
plt.legend()
plt.savefig("variance_total_d_var.eps", format = "eps")
 
plt.figure()
plt.plot(d_vec, e1_vecd, label=r"$v_1(n,u,d,p)$")
plt.plot(d_vec, e2_vecd, label=r"$v_2(n,u,d,p)$")
plt.plot(d_vec, e3_vecd, label=r"$v_3(n,u,d,p)$")
plt.plot(d_vec, e4_vecd, label=r"$v_4(n,u,d,p)$")
plt.plot(d_vec, e5_vecd, "-.", label=r"$v_5(n,u,d,p)$")
plt.vlines(u**(p/(p-1)),0,1,linestyles = "dashed", linewidth = 0.75, label = r"$d = u^{\frac{p}{p-1}}$")
plt.xlabel(r"$d\in (0,1]$")
plt.legend()
plt.savefig("variance_summands_d_var.eps", format = "eps")
 
#=============================================================================



