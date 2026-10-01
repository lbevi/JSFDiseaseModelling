# n-Patch Model

#To-do:

# Packages
import numpy as np
import random
import sys
import mjsf
import model as m
import Rt as rCalc
import h5py
import os
import multiprocessing as mp
import time

# Recursion Limit
sys.setrecursionlimit(100000)

#Number of Simulations
numSim = 100

# Print settings
np.set_printoptions(threshold=np.inf)   

# Compartments and Patch Number
comp = ["S_h", "E_h", "I_h", "A_h", "D_h", "R_h", "S_m", "E_m", "I_m"]
hComp = ["S", "E", "I", "A", "D", "R"]
mComp = ["S", "E", "I"]
k = len(comp) # number of compartments per patch

#Human Parameters
# S, E, I, A, D, R, (IC)
# u_h, b_h 
hP1 = m.createHumanPatch(25000, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4)
hP2 = m.createHumanPatch(1000, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4)
hP3 = m.createHumanPatch(50, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4)
hP4 = m.createHumanPatch(5000, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4)
hP5 = m.createHumanPatch(15, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4)

hPatch = [hP1, hP2, hP3, hP4, hP5]

n = len(hPatch); #patch size
tot_states = n*k #total number of states

#Mosquito Parameters
# S, E, I (IC)
# b_sy, b_asy, b_su, u_m

# BITE RATES INC BY 0.2

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
t_max = 2000 #Max simulation time
timeShow = 0 #Display current time every timeShow
random.seed(67) # Set seed
seed = 67

# State notation: (i = 0, 1, ..., 8)
#x[0] = S_h, x[1] = E_h, x[2] = I_h, #x[3] = A_h, #x[4] = D_h, #x[5] = R_h
#x[6] = S_m, x[7] = E_m, x[8] = I_m
#x[i+9(n-1)] for states in the nth patch, i is the comp type

# JSF Options
my_opts = {
            "EnforceDo": [0]*tot_states,
            "dt": 1,
            "SwitchingThreshold": [500]*tot_states
           }

tStep = my_opts["dt"]

numEvents = 26 # Total number of non-migration events (count in createRatesVector function)

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

def run_sim(i):
    # Run simulation
    # Rainfall/temperature variaton
    rngMatrix = m.generateTempRainRNG(n, t_max/tStep, seed + i)

    # Create rates vector for state x at time t
    rates = lambda x, t: m.createRatesVector(x, t, hPatch, mPatch, dPatch, mParam, n, k, comp, rngMatrix, tStep)
    sim = mjsf.jsf(IC, rates, stoich, t_max, config=my_opts, timeShow = timeShow, method="exact")
    r0 = rCalc.calculateR0(n, hPatch, mPatch, dPatch, mParam, timeShow, rngMatrix, tStep, sim[1])
    rt = rCalc.calculateRt(n, k, hPatch, mPatch, dPatch, mParam, timeShow, rngMatrix, tStep, sim)
    t = sim[1]; x = sim[0]
    return i, t, x, r0, rt, rngMatrix

# Run all simulations in parallel
if __name__ == "__main__":
    with mp.Pool(processes=numSim) as pool: 
        results = pool.map(run_sim, range(numSim))

# Saving simulations
with h5py.File("simBaseModelo=500.h5", "w") as f: 
        for i, t, x, r0, rt, rngMatrix in results: 
            grp = f.create_group(f"sim_{i}") 
            grp.create_dataset("time", data=t) 
            grp.create_dataset("states", data=x)
            grp.create_dataset("r0", data=r0)
            grp.create_dataset("rt", data=rt)
            grp.create_dataset("rng", data=rngMatrix)
            
# Load back simulations
sim = []

with h5py.File("simBaseModel.h5", "r") as f:
       keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))
       sim = [(f[k]["states"][:], f[k]["time"][:]) for k in keys]
       r0 =  [(f[k]["r0"][:]) for k in keys] 
       rt =  [(f[k]["rt"][:]) for k in keys]
       rng = [(f[k]["rng"][:]) for k in keys]

    # # Load back simulations
    # sim = []

    # with h5py.File("simBaseModel.h5", "r") as f:
    #     keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))
    #     sim = [(f[k]["time"][:], f[k]["states"][:]) for k in keys]

    # for i in range(numSim):
    #     plt.figure()
    #     plt.plot(sim[i][1], sim[i][0][0], label="Rt")
    #     plt.xlabel("Time")
    #     plt.ylabel("Rt")
    #     plt.title("Rt versus time")
    #     plt.legend()
    #     plt.show()            