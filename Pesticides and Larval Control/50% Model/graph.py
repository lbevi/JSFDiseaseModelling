# Packages
import numpy as np
import random
import matplotlib.pyplot as plt
import sympy as sp
import sys
import mjsf
import time
import model as m
import Rt as rCalc
import helper as h
import h5py
import os
import multiprocessing as mp

# Recursion Limit
sys.setrecursionlimit(100000)

#Number of Simulations
numSim = 100

# Print settings
np.set_printoptions(threshold=np.inf)   

# Compartments and Patch Number
comp = ["S_h", "E_h", "I_h", "A_h", "D_h", "R_h", "S_m", "E_m", "I_m", "V_h"]
hComp = ["S", "E", "I", "A", "D", "R", "V"]
mComp = ["S", "E", "I"]
k = len(comp) # number of compartments per patch

#Human Parameters
# S, E, I, A, D, R, V (IC)
# u_h, b_h, v, phi, beta
hP1 = m.createHumanPatch(25000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0.02, 1/365, 0.25)
hP2 = m.createHumanPatch(1000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 1/365, 0.25)
hP3 = m.createHumanPatch(50, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 1/365, 0.25)
hP4 = m.createHumanPatch(5000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 1/365, 0.25)
hP5 = m.createHumanPatch(15, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 1/365, 0.25)

hPatch = [hP1, hP2, hP3, hP4, hP5]

n = len(hPatch); #patch size
tot_states = n*k #total number of states

#Mosquito Parameters
# S, E, I (IC)
# b_sy, b_asy, b_su, u_m
red=0.75
inc=1.25
mP1 = m.createMosquitoPatch(50000, 0, 50,
                               0.4, 0.6, 0.6, 0.0477)
mP2 = m.createMosquitoPatch(2000, 0, 0,
                               0.4, 0.6, 0.6, 0.0477)
mP3 = m.createMosquitoPatch(100, 0, 0,
                               0.4, 0.6, 0.6, 0.0477)
mP4 = m.createMosquitoPatch(10000, 0, 0,
                               0.4, 0.6, 0.6, 0.0477)
mP5 = m.createMosquitoPatch(30, 0, 0,
                               0.4, 0.6, 0.6, 0.0477)

mPatch = [mP1, mP2, mP3, mP4, mP5]

#Disease Parameters
# d_h, gI_h, gA_h w_h, o_h, o_m, t_hm, t_mh, p_sy, chi
dP1 = m.createDiseasePatch(1.8e-3, 0.08, 0.01, 0.01, 1/15, 1/9, 0.022, 0.24, 0.288, 0.75)
dP2 = dP1
dP3 = dP1
dP4 = dP1
dP5 = dP1

dPatch = [dP1, dP2, dP3, dP4, dP5]

# Migration Parameters
# Note: [i,j] means from patch i-1 to patch j-1
tH = np.zeros((n,n)) #human transfer
tM = np.zeros((n,n)) #mosquito transfer

tH[0,1] = 10/hPatch[0]["ic"]["S"]; tH[1,0] = 10/hPatch[1]["ic"]["S"] #edge 1<->2
tH[0,2] = 0.1/hPatch[0]["ic"]["S"]; tH[2,0] = 0.1/hPatch[2]["ic"]["S"] #edge 1<->3
tH[0,3] = 100/hPatch[0]["ic"]["S"]; tH[3,0] = 100/hPatch[3]["ic"]["S"] #edge 1<->4
tH[1,2] = 0.1/hPatch[1]["ic"]["S"]; tH[2,1] = 0.1/hPatch[2]["ic"]["S"] #edge 2<->3
tH[3,4] = 0.05/hPatch[3]["ic"]["S"]; tH[4,3] = 0.05/hPatch[4]["ic"]["S"] #edge 4<->5

mParam = {}
for c in comp:
    if c.endswith("_h"):
        mParam[c] = tH.copy()
    elif c.endswith("_m"):
        mParam[c] = tM.copy()

mParam["D_h"] = np.zeros((n,n))

# IC Vector
IC = m.createIC(hPatch, mPatch, hComp, mComp, n)

print(IC)

# Simulation Parameters
t_max = 5000 #Max simulation time
timeShow = 0 #Display current time every timeShow
random.seed(68) # Set seed
seed = None

# State notation: (i = 0, 1, ..., 8)
#x[0] = S_h, x[1] = E_h, x[2] = I_h, #x[3] = A_h, #x[4] = D_h, #x[5] = R_h, x[9] = V_h
#x[6] = S_m, x[7] = E_m, x[8] = I_m, 
#x[i+10(n-1)] for states in the nth patch, i is the comp type

# JSF Options
my_opts = {
            "EnforceDo": [0]*tot_states,
            "dt": 1,
            "SwitchingThreshold": [2000]*tot_states
           }

tStep = my_opts["dt"]

# Rainfall/temperature variaton
rngMatrix = m.generateTempRainRNG(n, t_max/tStep, seed)

# Create rates vector for state x at time t
rates = lambda x, t: m.createRatesVector(x, t, hPatch, mPatch, dPatch, mParam, n, k, comp, rngMatrix, tStep)
numEvents = 31 # Total number of non-migration events (count in createRatesVector function)

# Calculate number of total events across all patches
allEvents = numEvents*n + n*(n-1)*(k-1)  #numEvents * n + migration events (k-1 due to dead patch)

# Create reactant and product matrix [reactant_matrix, product_matrix] (matrices need to be lists of lists, not np.array)
rMatrix = m.createReactantMatrix(n, k, numEvents, allEvents)
pMatrix = m.createProductMatrix(n, k, numEvents, allEvents)

stoich = {
         "nu": [ [a - b for a, b in zip(r1, r2)]
                for r1, r2 in zip(pMatrix, rMatrix) ],
         "DoDisc": [1]*tot_states,
         "nuReactant": rMatrix,
         "nuProduct": pMatrix,
         }
# Load back simulations
sim = []

with h5py.File("VaccineModel1patch50.h5", "r") as f:
    keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))
    sim = [(f[k]["states"][:], f[k]["time"][:]) for k in keys] #i = sim num, j = states/time, k = specific state
    r0 =  [(f[k]["r0"][:]) for k in keys] 
    rt =  [(f[k]["rt"][:]) for k in keys]
    rng = [(f[k]["rng"][:]) for k in keys]         


#Print 1 Simulation
#h.printR0(sim[0], r0[0])
#h.printRt(sim[0], rt[0])
#combineInfectedStates = 1 # Combine infected states into one line
#h.printAllPatch(n, k, sim[0], combineInfectedStates, hComp, mComp, my_opts)
#h.printOverlapODE(n, k, sim[0], hComp, mComp, my_opts, hPatch, mPatch, dPatch, mParam, IC, t_max, rngMatrix, tStep)

# Disease elim
#h.PlotDiseaseElim(sim, n, k, numSim)

#Average plots
#Average plots
#Median ODE
totSim = len(sim)
odeSol = []
for i in range(totSim):   
    xODE=m.solveODE(hPatch, mPatch, dPatch, mParam, n, k, IC, t_max, rng[i], tStep)
    odeSol.append(xODE.y)
    
odeSol = np.array(odeSol) 
xODE.y = np.median(odeSol, axis=0)    

t = np.transpose(np.array(sim[0][1]))
sim = np.transpose(np.stack([s[0] for s in sim]), (1,0,2))
ciVec = [95, 80, 50]   

h.averageNewPlots(t, sim, r0, rt, ciVec, n, k, comp, my_opts, xODE)

#u_h, b_h, v, phi, beta

# t_max = 5000

# hP1 = m.createHumanPatch(25000, 0, 0, 0, 0, 0, 0,
#                              4.5e-5, 1.2e-4, 0.03, 1/365, 0.25)
# hP2 = m.createHumanPatch(1000, 0, 0, 0, 0, 0, 0,
#                              4.5e-5, 1.2e-4, 0.03, 1/365, 0.25)
# hP3 = m.createHumanPatch(50, 0, 0, 0, 0, 0, 0,
#                              4.5e-5, 1.2e-4, 0.03, 1/365, 0.25)
# hP4 = m.createHumanPatch(5000, 0, 0, 0, 0, 0, 0,
#                              4.5e-5, 1.2e-4, 0.03, 1/365, 0.25)
# hP5 = m.createHumanPatch(15, 0, 0, 0, 0, 0, 0,
#                              4.5e-5, 1.2e-4, 0.03, 1/365, 0.25)
# hPatch = [hP1, hP2, hP3, hP4, hP5]

# rngNew = m.generateTempRainRNG(n, t_max/tStep, seed)
# xODE=m.solveODE(hPatch, mPatch, dPatch, mParam, n, k, IC, t_max, np.tile(rngNew, (1, 10)), tStep)
# plt.figure()
# plt.plot(xODE.t, xODE.y[1] + xODE.y[2] + xODE.y[3], color='b')
# plt.xlabel("Time")
# plt.ylabel("S_h")
# plt.title(f"test")
# plt.margins(x=0)
# plt.show()