# 1-Patch Model

# Packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sympy as sp
import sys
import mjsf
import time

# Recursion Limit
sys.setrecursionlimit(10000)

# Human Parameters
u_h = 1/(70*365) #death rate per day (lifespan 70y)
b_h = 1/(65*365) #1/(35*365) # birth rate per day (a little bit more than death rate)
g_h = 1/14 # recovery rate per day (14 days infectious)

s_h0 = 10000 #initial susceptibles
e_h0 = 0 #initial exposed
i_h0 = 0 #initial sympto
a_h0 = 0 #initial asympto
r_h0 = 0 #initial recovered
d_h0 = 0 #initial dead due to malaria

# Mosquito Parameters
b_S = 0.5 #bite rate per day on symptomatic human
b_A = 1 #bite rate per day on asymptomatic human/susceptible human 
u_m = 1/9 #death rate per day (lifespan 9d)
#b_m = 1/9 #birth rate per day (a little bit more than birth rate)

# Mosquito birth function
def b(t):
    b_diff = 0.005 #difference between min and max
    b_av  = 1/9 #average birth rate
    phi = np.pi/2 #phase shift

    b_min = b_av - b_diff #min birth rate
    b_max = b_av + b_diff #max birth rate

    return ((b_min+b_max)/2)+((b_max-b_min)/2)*np.sin(2*np.pi*t/365+phi)

s_m0 = 30000 #initial susceptibles
e_m0 = 0 #initial exposed
i_m0 = 10 #initial infected

# Disease Parameters
p_death = 597000/263000000 #probability of death due to malaria (WHO) (homogeneity)
d_h = p_death * g_h #human death rate due to malaria per day
w_h = 1/31 #waning immunity rate per day (1 months of being immune)
o_h = 1/12 #human latent period per day (12 day latent period)
o_m = 1/10 #mosquito latent period per day (10 day latent period)
t_hm = 0.20 #probability of transmission from human to mosquito
t_mh = 0.15 #probability of transmission from mosquito to human
p_non_asy = 0.7 #probability of asymptomatic case

# Simulation Parameters
t_max = 1000 #Max simulation time
IC = [s_h0, e_h0, i_h0, a_h0, d_h0, r_h0, s_m0, e_m0, i_m0]
timeShow = 10 #Display current time every timeShow

rates = lambda x, t: [b_h*(x[0]+x[1]+x[2]+x[3]+x[5]),  #human birth
            u_h*x[0],  #human susceptible death
            u_h*x[1],  #human exposed death
            u_h*x[2],  #human sympto death (not due to disease)
            u_h*x[3],  #human asympto death (not due to disease)
            u_h*x[5],  #human recovered death
            b(t)*(x[6]+x[7]+x[8]),  #mosquito birth
            u_m*x[6],  #mosquito susceptible death
            u_m*x[7],  #mosquito exposed death
            u_m*x[8],  #mosquito infected death
            b_A*t_mh*x[0]*x[8]/(x[0]+x[1]+x[2]+x[3]+x[5]) if x[0]+x[1]+x[2]+x[3]+x[5] > 0 else 0,  #human becomes exposed due to infected mosquito
            o_h*p_non_asy*x[1],  #human becomes sympto from being exposed
            o_h*(1-p_non_asy)*x[1],  #human becomes asympto from being exposed
            d_h*x[2],  #human dies due to malaria
            g_h*x[2],  #human sympto recoveres
            g_h*x[3],  #human asympto recoveres
            w_h*x[5],  #human becomes susceptible
            b_S*t_hm*x[6]*x[2]/(x[0]+x[1]+x[2]+x[3]+x[5]) if x[0]+x[1]+x[2]+x[3]+x[5] > 0 else 0,  #mosquito becomes exposed due to sympto human
            b_A*t_hm*x[6]*x[3]/(x[0]+x[1]+x[2]+x[3]+x[5]) if x[0]+x[1]+x[2]+x[3]+x[5] > 0 else 0,  #mosquito becomes exposed due to asympto human
            o_m*x[7]]  #mosquito becomes infected from being exposed

#State Space
#x[0] = s_h
#x[1] = e_h
#x[2] = i_h
#x[3] = a_h
#x[4] = d_h
#x[5] = r_h
#x[6] = s_m
#x[7] = e_m
#x[8] = i_m

# Reactant & Product Matrix
r_matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0],  #human birth
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  #human susceptible death
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  #human exposed death
            [0, 0, 1, 0, 0, 0, 0, 0, 0],  #human sympto death (not due to disease)
            [0, 0, 0, 1, 0, 0, 0 ,0, 0],  #human asympto death (not due to disease)
            [0, 0, 0, 0, 0, 1, 0 ,0, 0],  #human recovered death
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  #mosquito birth
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  #mosquito susceptible death
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  #mosquito exposed death
            [0, 0, 0, 0, 0, 0, 0, 0, 1],  #mosquito infected death
            [1, 0, 0, 0, 0, 0, 0, 0, 1],  #human becomes exposed due to infected mosquito
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  #human becomes sympto from being exposed
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  #human becomes asympto from being exposed
            [0, 0, 1, 0, 0, 0, 0, 0, 0],  #human dies due to malaria
            [0, 0, 1, 0, 0, 0, 0, 0, 0],  #human sympto recoveres
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  #human asympto recoveres
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  #human becomes susceptible
            [0, 0, 1, 0, 0, 0, 1, 0, 0],  #mosquito becomes exposed due to sympto human
            [0, 0, 0, 1, 0, 0, 1, 0, 0],  #mosquito becomes exposed due to asympto human
            [0, 0, 0, 0, 0, 0, 0, 1, 0]]  #mosquito becomes infected from being exposed

p_matrix = [[1, 0, 0, 0, 0, 0, 0, 0, 0],  #human birth
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  #human susceptible death
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  #human exposed death
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  #human sympto death (not due to disease)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  #human asympto death (not due to disease)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  #human recovered death
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  #mosquito birth
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  #mosquito susceptible death
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  #mosquito exposed death
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  #mosquito infected death
            [0, 1, 0, 0, 0, 0, 0, 0, 1],  #human becomes exposed due to infected mosquito
            [0, 0, 1, 0, 0, 0, 0, 0, 0],  #human becomes sympto from being exposed
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  #human becomes asympto from being exposed
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  #human dies due to malaria
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  #human sympto recovers
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  #human asympto recovers
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  #human becomes susceptible
            [0, 0, 1, 0, 0, 0, 0, 1, 0],  #mosquito becomes exposed due to sympto human
            [0, 0, 0, 1, 0, 0, 0, 1, 0],  #mosquito becomes exposed due to asympto human
            [0, 0, 0, 0, 0, 0, 0, 0, 1]]  #mosquito becomes infected from being exposed

# JSF Options
stoich = {
         "nu": [ [a - b for a, b in zip(r1, r2)]
                for r1, r2 in zip(p_matrix, r_matrix) ],
         "DoDisc": [1, 1, 1, 1, 1, 1, 1, 1, 1],
         "nuReactant": r_matrix,
         "nuProduct": p_matrix,
         }

my_opts = {
            "EnforceDo": [0, 0, 0, 0 ,0 ,0 ,0 ,0, 0],
            "dt": 1,
            "SwitchingThreshold": [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
            #"SwitchingThreshold": [0, 0, 0, 0, 0, 0, 0, 0]
           }

# Simulation and Recording Time
start = time.time()
sim = mjsf.jsf(IC, rates, stoich, t_max, config=my_opts, timeShow=timeShow, method="exact")
end = time.time() 
elapsed = round(end - start,3)
print(f"Elapsed time: {elapsed} seconds")

# # Printing Final T value
# fin = len(sim[0][0])

# print("S_H", sim[0][0][fin-1])
# print("E_H", sim[0][1][fin-1])
# print("I_H", sim[0][2][fin-1])
# print("R_H", sim[0][3][fin-1])
# print("S_M", sim[0][4][fin-1])
# print("E_M", sim[0][5][fin-1])
# print("I_M", sim[0][6][fin-1])
# print("D_H", sim[0][7][fin-1])

plt.figure()
plt.plot(sim[1], sim[0][0], label="S_h")
plt.plot(sim[1], sim[0][1], label="E_h")
plt.plot(sim[1], sim[0][2], label="I_h")
plt.plot(sim[1], sim[0][3], label="A_h")
plt.plot(sim[1], sim[0][4], label="D_h")
plt.plot(sim[1], sim[0][5], label="R_h")
plt.axhline(y=my_opts["SwitchingThreshold"][1], color="k", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Human Population")
plt.title("Human States vs Time")
plt.legend()
plt.show()

plt.figure()
plt.plot(sim[1], sim[0][6], label="S_m")
plt.plot(sim[1], sim[0][7], label="E_m")
plt.plot(sim[1], sim[0][8], label="I_m")
plt.axhline(y=my_opts["SwitchingThreshold"][1], color="k", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Mosquito Population")
plt.title("Mosquito States vs Time")
plt.legend()
plt.show()

plt.figure()
plt.plot(sim[1], np.array(sim[0][0]) + np.array(sim[0][1]) + np.array(sim[0][2]) + np.array(sim[0][3]) + np.array(sim[0][5]), label="Humans")
plt.xlabel("Time")
plt.ylabel("Total Human Population")
plt.title("Total Human Population vs Time")
plt.legend()
plt.show()

plt.figure()
plt.plot(sim[1], np.array(sim[0][6]) + np.array(sim[0][7]) + np.array(sim[0][8]), label="Mosquitoes")
plt.xlabel("Time")
plt.ylabel("Total Mosquito Population")
plt.title("Total Mosquito Population vs Time")
plt.legend()
plt.show()






