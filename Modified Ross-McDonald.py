# Modified Ross-McDonald JSF Modelling

# Packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import jsf
import sympy as sp
import sys

# Recursion Limit
sys.setrecursionlimit(10000)

# Human Parameters
u_h = 1/(70*365) #death rate per day (lifespan 70y)
b_h = 1/(65*365) #1/(35*365) # birth rate per day (a little bit more than death rate)
g_h = 1/14 # recovery rate per day (14 days infectious)

s_h0 = 10000 #initial susceptibles
e_h0 = 0 #initial exposed
i_h0 = 0 #initial infected
r_h0 = 0 #initial recovered
d_h0 = 0 #initial dead due to malaria

# Mosquito Parameters
b = 0.5 #bite rate per day 
u_m =  1/9 #death rate per day (lifespan 10d)
b_m = 1/9 #birth rate per day (a little bit more than birth rate)

s_m0 = 100000 #initial susceptibles
e_m0 = 0 #initial exposed
i_m0 = 10 #initial infected

# Disease Parameters
p_death = 0.001 #probability of death due to malaria
d_h = -np.log(1 - p_death) * g_h #human death rate due to malaria per day
w_h = 1/(31) #waning immunity rate per day (1 months of being immune)
o_h = 1/12 #human latent period per day (12 day latent period)
o_m = 1/10 #mosquito latent period per day (10 day latent period)
t_hm = 0.10 #probability of transmission from human to mosquito
t_mh = 0.15 #probability of transmission from mosquito to human

# Equilibria points
# Variables
S_H, E_H, I_H, D_H, R_H, S_M, E_M, I_M = sp.symbols('S_H E_H I_H D_H R_H S_M E_M I_M')

# Equations
dS_Hdt = sp.Eq((S_H+E_H+I_H+R_H)*b_h -S_H*u_h-b*t_mh*I_M*S_H/(S_H+E_H+I_H+R_H)+w_h*R_H, 0)
dE_Hdt = sp.Eq(b*t_mh*I_M*S_H/(S_H+E_H+I_H+R_H)-(o_h+u_h)*E_H, 0)
dI_Hdt = sp.Eq(o_h*E_H-(d_h+u_h+g_h)*I_H, 0)
dD_Hdt = sp.Eq(d_h*I_H, 0)
dR_Hdt = sp.Eq(g_h*I_H-(u_h+w_h)*R_H, 0)
dS_Mdt = sp.Eq((S_M+E_M+I_M)*b_m-u_m*S_M-b*t_hm*S_M*I_H/(S_H+E_H+I_H+R_H), 0)
dE_Mdt = sp.Eq(b*t_hm*S_M*I_H/(S_H+E_H+I_H+R_H)-(o_m+u_m)*E_M, 0)
dI_Mdt = sp.Eq(o_m*E_M-u_m*I_M, 0)

#guess = {S_H: 10000, E_H: 0, I_H: 0, R_H: 0, S_M: 100000, E_M: 0, I_M: 0}
solution = sp.solve((dS_Hdt, dE_Hdt, dI_Hdt, dR_Hdt, dS_Mdt, dE_Mdt, dI_Mdt), (S_H, E_H, I_H, R_H, S_M, E_M, I_M))

print(solution)

# Simulation Parameters
t_max = 1000 #Max simulation time
IC = [s_h0, e_h0, i_h0, r_h0, s_m0, e_m0, i_m0 ,d_h0]

rates = lambda x, _: [b_h*(x[0]+x[1]+x[2]+x[3]),  #human birth
            u_h*x[0],  #human susceptible death
            u_h*x[1],  #human exposed death
            u_h*x[2],  #human infected death (not due to disease)
            u_h*x[3],  #human recovered death
            b_m*(x[4]+x[5]+x[6]),  #mosquito birth
            u_m*x[4],  #mosquito susceptible death
            u_m*x[5],  #mosquito exposed death
            u_m*x[6],  #mosquito infected death
            b*t_mh*x[6]*x[0]/(x[0]+x[1]+x[2]+x[3]) if (x[0]+x[1]+x[2]+x[3])>0 else 0,  #human becomes exposed due to infected mosquito
            o_h*x[1],  #human becomes infected from being exposed
            d_h*x[2],  #human dies due to malaria
            g_h*x[2],  #human recoveres
            w_h*x[3],  #human becomes susceptible
            b*t_hm*x[4]*x[2]/(x[0]+x[1]+x[2]+x[3]) if (x[0]+x[1]+x[2]+x[3])>0 else 0,  #mosquito becomes exposed due to infected human
            o_m*x[5]]  #mosquito becomes infected from being exposed

#State Space
#x[0] = s_h
#x[1] = e_h
#x[2] = i_h
#x[3] = r_h
#x[4] = s_m
#x[5] = e_m
#x[6] = i_m
#x[7] = d_h

# Reactant & Product Matrix
r_matrix = [[0, 0, 0, 0, 0, 0, 0, 0],  #human birth
            [1, 0, 0, 0, 0, 0, 0, 0],  #human susceptible death
            [0, 1, 0, 0, 0, 0, 0, 0],  #human exposed death
            [0, 0, 1, 0, 0, 0, 0, 0],  #human infected death (not due to disease)
            [0, 0, 0, 1, 0, 0, 0 ,0],  #human recovered death
            [0, 0, 0, 0, 0, 0, 0, 0],  #mosquito birth
            [0, 0, 0, 0, 1, 0, 0, 0],  #mosquito susceptible death
            [0, 0, 0, 0, 0, 1, 0, 0],  #mosquito exposed death
            [0, 0, 0, 0, 0, 0, 1, 0],  #mosquito infected death
            [1, 0, 0, 0, 0, 0, 1, 0],  #human becomes exposed due to infected mosquito
            [0, 1, 0, 0, 0, 0, 0, 0],  #human becomes infected from being exposed
            [0, 0, 1, 0, 0, 0, 0, 0],  #human dies due to malaria
            [0, 0, 1, 0, 0, 0, 0, 0],  #human recoveres
            [0, 0, 0, 1, 0, 0, 0, 0],  #human becomes susceptible
            [0, 0, 1, 0, 1, 0, 0, 0],  #mosquito becomes exposed due to infected human
            [0, 0, 0, 0, 0, 1, 0, 0]]  #mosquito becomes infected from being exposed

p_matrix = [[1, 0, 0, 0, 0, 0, 0, 0],  #human birth
            [0, 0, 0, 0, 0, 0, 0, 0],  #human susceptible death
            [0, 0, 0, 0, 0, 0, 0 ,0],  #human exposed death
            [0, 0, 0, 0, 0, 0, 0, 0],  #human infected death (not due to disease)
            [0, 0, 0, 0, 0, 0, 0, 0],  #human recovered death
            [0, 0, 0, 0, 1, 0, 0, 0],  #mosquito birth
            [0, 0, 0, 0, 0, 0, 0, 0],  #mosquito susceptible death
            [0, 0, 0, 0, 0, 0, 0, 0],  #mosquito exposed death
            [0, 0, 0, 0, 0, 0, 0, 0],  #mosquito infected death
            [0, 1, 0, 0, 0, 0, 1, 0],  #human becomes exposed due to infected mosquito
            [0, 0, 1, 0, 0, 0, 0, 0],  #human becomes infected from being exposed
            [0, 0, 0, 0, 0, 0, 0, 1],  #human dies due to malaria
            [0, 0, 0, 1, 0, 0, 0, 0],  #human recovers
            [1, 0, 0, 0, 0, 0, 0, 0],  #human becomes susceptible
            [0, 0, 1, 0, 0, 1, 0, 0],  #mosquito becomes exposed due to infected human
            [0, 0, 0, 0, 0, 0, 1, 0]]  #mosquito becomes infected from being exposed

# JSF Options
stoich = {
         "nu": [ [a - b for a, b in zip(r1, r2)]
                for r1, r2 in zip(p_matrix, r_matrix) ],
         "DoDisc": [1, 1, 1, 1, 1, 1, 1, 1],
         "nuReactant": r_matrix,
         "nuProduct": p_matrix,
         }

my_opts = {
            "EnforceDo": [0, 0, 0, 0 ,0 ,0 ,0 ,0],
            "dt": 1,
            "SwitchingThreshold": [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
            #"SwitchingThreshold": [0, 0, 0, 0, 0, 0, 0, 0]
           }

sim = jsf.jsf(IC, rates, stoich, t_max, config=my_opts, method="operator-splitting")

# Printing Final T value
fin = len(sim[0][0])

print("S_H", sim[0][0][fin-1])
print("E_H", sim[0][1][fin-1])
print("I_H", sim[0][2][fin-1])
print("R_H", sim[0][3][fin-1])
print("S_M", sim[0][4][fin-1])
print("E_M", sim[0][5][fin-1])
print("I_M", sim[0][6][fin-1])
print("D_H", sim[0][7][fin-1])

plt.figure()
plt.plot(sim[1], sim[0][0], label="S_h")
plt.plot(sim[1], sim[0][1], label="I_h")
plt.plot(sim[1], sim[0][2], label="E_h")
plt.plot(sim[1], sim[0][3], label="R_h")
plt.plot(sim[1], sim[0][7], label="D_h")
plt.axhline(y=my_opts["SwitchingThreshold"][1], color="k", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Human Population")
plt.title("Human States vs Time")
plt.legend()
plt.show()

plt.figure()
plt.plot(sim[1], sim[0][4], label="S_m")
plt.plot(sim[1], sim[0][5], label="E_m")
plt.plot(sim[1], sim[0][6], label="I_m")
plt.axhline(y=my_opts["SwitchingThreshold"][1], color="k", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Mosquito Population")
plt.title("Mosquito States vs Time")
plt.legend()
plt.show()

plt.figure()
plt.plot(sim[1], np.array(sim[0][0]) + np.array(sim[0][1]) + np.array(sim[0][2]) + np.array(sim[0][3]), label="Humans")
plt.xlabel("Time")
plt.ylabel("Total Human Population")
plt.title("Total Human Population vs Time")
plt.legend()
plt.show()

plt.figure()
plt.plot(sim[1], np.array(sim[0][4]) + np.array(sim[0][5]) + np.array(sim[0][6]), label="Mosquitoes")
plt.xlabel("Time")
plt.ylabel("Total Mosquito Population")
plt.title("Total Mosquito Population vs Time")
plt.legend()
plt.show()

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






