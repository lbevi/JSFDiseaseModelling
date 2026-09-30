# Modified Ross-McDonald JSF Modelling
from scipy.integrate import solve_ivp

# Packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import jsf
import sympy as sp
import sys
import os

# Recursion Limit
sys.setrecursionlimit(10000)

# Human Parameters
s_h0 = 1000 #initial susceptibles
i_h0 = 0 #initial infected
r_h0 = 0 #initial recovered

u_h = 4.5*10**-5 #death rate per day 
b_h = 1.2*10**-4 * (s_h0 + i_h0 + r_h0) # birth rate per day 
g_h = 1/14 # recovery rate per day 

# Mosquito Parameters
s_m0 = 2000 #initial susceptibles
i_m0 = 50 #initial infected

b = 0.6 #bite rate per day 
u_m = 0.0477 #death rate per day
B_M = 0.0477 *(s_m0+i_m0)

def b_m(t):
       t = np.array(t)
       b_av  = B_M #average birth rate 
       b_diff = 0.2*b_av #difference between min and max
       return b_av+b_diff*np.cos(2*np.pi*t/365)

# Disease Parameters
t_hm = 0.10 #probability of transmission from human to mosquito
t_mh = 0.15 #probability of transmission from mosquito to human

# Simulation Parameters
t_max = 1000 #Max simulation time
IC = [s_h0, i_h0, r_h0, s_m0, i_m0]
random.seed(67)

rates = lambda x, t: [b_h if x[0] + x[1] + x[2] > 0 else 0,  #human birth 
            u_h*x[0],  #human susceptible death
            u_h*x[1],  #human infected death (not due to disease)
            u_h*x[2],  #human recovered death
            b_m(t) if x[3] + x[4] > 0 else 0,  #mosquito birth
            u_m*x[3],  #mosquito susceptible death
            u_m*x[4],  #mosquito exposed death
            b*t_mh*x[4]*x[0]/(x[0]+x[1]+x[2]) if (x[0]+x[1]+x[2])>0 else 0,  #human becomes infected due to infected mosquito
            g_h*x[1],  #human recoveres
            b*t_hm*x[3]*x[1]/(x[0]+x[1]+x[2]) if (x[0]+x[1]+x[2])>0 else 0,  #mosquito becomes exposed due to infected human
]

#State Space
#x[0] = s_h
#x[1] = i_h
#x[2] = r_h
#x[3] = s_m
#x[4] = i_m

# Reactant & Product Matrix
r_matrix = [[0, 0, 0, 0, 0],  #human birth 
            [1, 0, 0, 0, 0],  #human susceptible death
            [0, 1, 0, 0, 0],  #human infected death (not due to disease)
            [0, 0, 1, 0, 0],  #human recovered death
            [0, 0, 0, 0, 0],  #mosquito birth 
            [0, 0, 0, 1, 0],  #mosquito susceptible death
            [0, 0, 0, 0, 1],  #mosquito infected death
            [1, 0, 0, 0, 1],  #human becomes infected due to infected mosquito
            [0, 1, 0, 0, 0],  #human recoveres
            [0, 1, 0, 1, 0]]  #mosquito becomes infected due to infected human

p_matrix = [[1, 0, 0, 0, 0],  #human birth
            [0, 0, 0, 0, 0],  #human susceptible death
            [0, 0, 0, 0, 0],  #human infected death (not due to disease)
            [0, 0, 0, 0, 0],  #human recovered death
            [0, 0, 0, 1, 0],  #mosquito birth
            [0, 0, 0, 0, 0],  #mosquito susceptible death
            [0, 0, 0, 0, 0],  #mosquito infected death
            [0, 1, 0, 0, 1],  #human becomes infected due to infected mosquito
            [0, 0, 1, 0, 0],  #human recovers
            [0, 1, 0, 0, 1]]  #mosquito becomes infected due to infected human

# JSF Options
stoich = {
         "nu": [ [a - b for a, b in zip(r1, r2)]
                for r1, r2 in zip(p_matrix, r_matrix) ],
         "DoDisc": [1, 1, 1, 1, 1, 1, 1, 1],
         "nuReactant": r_matrix,
         "nuProduct": p_matrix,
         }

my_opts = {
            "EnforceDo": [0, 0, 0, 0 ,0 ,0],
            "dt": 1,
            "SwitchingThreshold": [0]*5
            #"SwitchingThreshold": [0, 0, 0, 0, 0, 0, 0, 0]
           }

sim = jsf.jsf(IC, rates, stoich, t_max, config=my_opts, method="exact")

#Solve ODE
def solveODE(IC, t_max, b_h, u_h, g_h, b, u_m, t_mh, t_hm):

    x = solve_ivp(systemODE, (0, t_max), np.array(IC), t_eval = np.linspace(0, t_max, 100*t_max), args=(b_h, u_h, g_h, b, u_m, t_mh, t_hm))

    return x

# Defines ODE equation
def systemODE(t, x, b_h, u_h, g_h, b, u_m, t_mh, t_hm):
    dxdt = np.zeros_like(x)
    #Define states
    S_h, I_h, R_h = x[0:3]
    S_m, I_m = x[3:5]

    N_h = S_h + I_h  + R_h
    N_m = S_m + I_m

    λ_h = b * t_mh * I_m / (N_h+1e-12)
    λ_m = b * t_hm * I_h  / (N_h+1e-12)

    #Human equations
    #S_h
    dxdt[0] = b_h - λ_h*S_h - u_h*S_h
    #I_h
    dxdt[1] = λ_h*S_h - g_h*I_h - u_h*I_h
    #R_h
    dxdt[2] = g_h*I_h - u_h*R_h
    #Mosquito equations
    #S_m 
    dxdt[3] =  b_m(t) - λ_m*S_m - u_m*S_m
    #I_m
    dxdt[4] = λ_m*S_m - u_m*I_m

    return dxdt

ODE = solveODE(IC, t_max, b_h, u_h, g_h, b, u_m, t_mh, t_hm)

bR = np.sqrt((b**2*t_hm*t_mh*b_m(sim[1])*u_h)/((u_h+g_h)*u_m**2*b_h))

plt.plot(sim[1], bR, label=r"$R_0$", color='b')
plt.xlabel("Time (days)", fontsize=14)
plt.ylabel(r"$R_0$", fontsize=14)
plt.axhline(y=1, linestyle='--', color='black')
plt.savefig("RMDODER0.png", dpi=150, bbox_inches="tight")
plt.show()

TotH = np.array(sim[0][0]) + np.array(sim[0][1]) + np.array(sim[0][2])
bRt = np.sqrt((b**2*t_hm*t_mh*np.array(sim[0][3])*np.array(sim[0][0]))/((u_h+g_h)*u_m*TotH**2))

plt.plot(sim[1], bRt, label=r"$R_t$", color='gray')
plt.xlabel("Time (days)", fontsize=14)
plt.ylabel(r"$R_t$", fontsize=14)
plt.axhline(y=1, linestyle='--', color='black')
plt.savefig("RMDODERt.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(sim[1], sim[0][0], label="S", color='purple')
plt.plot(sim[1], sim[0][1], label="I", color='green')
plt.plot(sim[1], sim[0][2], label="R", color='red')
#plt.plot(ODE.t, ODE.y[0], label = "S_ODE")
#plt.plot(ODE.t, ODE.y[1], label = "I_ODE")
#plt.plot(ODE.t, ODE.y[2], label = "R_ODE")
plt.xlabel("Time (days)", fontsize=14)
plt.ylabel("Human Population", fontsize=14)
plt.savefig("RMDODEHuman.png", dpi=150, bbox_inches="tight")
plt.show()



plt.figure()
plt.plot(sim[1], sim[0][3], label="S", color='skyblue')
plt.plot(sim[1], sim[0][4], label="I", color='orange')
#plt.plot(ODE.t, ODE.y[3], label = "S_ODE")
#plt.plot(ODE.t, ODE.y[4], label = "I_ODE")
plt.xlabel("Time (days)", fontsize=14)
plt.ylabel("Mosquito Population", fontsize=14)
plt.savefig("RMDODEMosquito.png", dpi=150, bbox_inches="tight")
plt.show()


# plt.figure()
# plt.plot(sim[1], np.array(sim[0][0]) + np.array(sim[0][1]) + np.array(sim[0][2]), label="Humans")
# plt.xlabel("Time (days)")
# plt.ylabel("Total Human Population")
# plt.title("Total Human Population vs Time")
# plt.legend()
# plt.savefig("TotalHuman.png", dpi=1000, bbox_inches="tight")
# plt.show()


# plt.figure()
# plt.plot(sim[1], np.array(sim[0][3]) + np.array(sim[0][4]), label="Mosquitoes")
# plt.xlabel("Time (days)")
# plt.ylabel("Total Mosquito Population")
# plt.title("Total Mosquito Population vs Time")
# plt.legend()
# plt.savefig("TotalMosquito.png", dpi=1000, bbox_inches="tight")
# plt.show()

# reps = 1000
# prob_data = np.zeros((2, 21))
# for i in range(0,21):
#        sum = 0
#        print("curr i", i)
#        for j in range(0,reps + 1):       
#               stoich = {
#               "nu": [ [a - b for a, b in zip(r1, r2)]
#                      for r1, r2 in zip(p_matrix, r_matrix) ],
#               "DoDisc": [1, 1, 1],
#               "nuReactant": r_matrix,
#               "nuProduct": p_matrix,
#               }

#               my_opts = {
#               "EnforceDo": [0, 0, 0],
#               "dt": 0.01,
#               "SwitchingThreshold": [50*i, 50*i, 50*i]
#               }

#               #sim = jsf.jsf(IC, rates, stoich, t_max, config=my_opts, method="exact")
#               sim = jsf.jsf(IC, rates, stoich, t_max, config=my_opts, method="operator-splitting")
#               final = len(sim[0][0])

#               if round(sim[0][0][final-1]) == 0 or round(sim[0][1][final-1]) == 0 or round(sim[0][2][final-1]) == 0:
#                      sum = sum + 1
#        prob_data[0,i] = 50*i
#        prob_data[1,i] = sum/reps

# plt.plot(prob_data[0], prob_data[1], label="probability")       
# plt.xlabel("Switching threshold")
# plt.ylabel("Probability")
# plt.legend()
# plt.show()

# print(prob_data)






