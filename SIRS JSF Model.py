# SIRS JSF Modelling

# Packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import jsf

# SIRS Model
# s_dot = -beta*s*i + w*r
# i_dot = beta*s*i  - g*i
# r_dot = g*i - w*r

# Parameters
b = 2.0 # infection rate
g = 0.4 # recovery rate
w = 0.1 # rate for antibodies to wear off
N = 1000 # population size
IC = [999, 1, 0] # Initial condition (S,I,R)
R_0 = b/g # R_0 = Avg number of people infected by one infected person
t_max = 20 #Max simulation time

rates = lambda x, _: [b/N * x[0] * x[1], #b*s*i
                      g * x[1], #g*i
                      w * x[2]] #w*r

# Reactant & Product Matrix
r_matrix = [[1, 1, 0],  #disease spread
            [0, 1, 0],  #someone recovering
            [0, 0, 1]] #someone becoming susceptible

p_matrix = [[0, 2, 0], #disease spread
            [0, 0, 1],  #someone recovering
            [1, 0, 0]] #someone becoming susceptible

# JSF Options
stoich = {
         "nu": [ [a - b for a, b in zip(r1, r2)]
                for r1, r2 in zip(p_matrix, r_matrix) ],
         "DoDisc": [1, 1, 1],
         "nuReactant": r_matrix,
         "nuProduct": p_matrix,
         }

my_opts = {
            "EnforceDo": [0, 0, 0],
            "dt": 0.01,
            "SwitchingThreshold": [200, 200, 200]
           }

sim = jsf.jsf(IC, rates, stoich, t_max, config=my_opts, method="exact")

final = len(sim[0][0])
print(sim[0][1][final-1])

print(sim[0][0][final-1]+sim[0][1][final-1]+sim[0][2][final-1])
plt.figure()
plt.plot(sim[1], sim[0][0], label="S")
plt.plot(sim[1], sim[0][1], label="I")
plt.plot(sim[1], sim[0][2], label="R")
plt.axhline(y=my_opts["SwitchingThreshold"][1], color="k", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Population Proportion")
plt.legend()
plt.show()

plt.figure()

reps = 1000
prob_data = np.zeros((2, 21))
for i in range(0,21):
       sum = 0
       print("curr i", i)
       for j in range(0,reps + 1):       
              stoich = {
              "nu": [ [a - b for a, b in zip(r1, r2)]
                     for r1, r2 in zip(p_matrix, r_matrix) ],
              "DoDisc": [1, 1, 1],
              "nuReactant": r_matrix,
              "nuProduct": p_matrix,
              }

              my_opts = {
              "EnforceDo": [0, 0, 0],
              "dt": 0.01,
              "SwitchingThreshold": [50*i, 50*i, 50*i]
              }

              #sim = jsf.jsf(IC, rates, stoich, t_max, config=my_opts, method="exact")
              sim = jsf.jsf(IC, rates, stoich, t_max, config=my_opts, method="operator-splitting")
              final = len(sim[0][0])

              if round(sim[0][0][final-1]) == 0 or round(sim[0][1][final-1]) == 0 or round(sim[0][2][final-1]) == 0:
                     sum = sum + 1
       prob_data[0,i] = 50*i
       prob_data[1,i] = sum/reps

plt.plot(prob_data[0], prob_data[1], label="probability")       
plt.xlabel("Switching threshold")
plt.ylabel("Probability")
plt.legend()
plt.show()

print(prob_data)






