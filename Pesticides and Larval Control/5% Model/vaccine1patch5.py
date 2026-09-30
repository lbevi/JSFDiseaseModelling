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
                             4.5e-5, 1.2e-4, 0., 1/365, 0.25)
hP5 = m.createHumanPatch(15, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 1/365, 0.25)

hPatch = [hP1, hP2, hP3, hP4, hP5]

n = len(hPatch); #patch size
tot_states = n*k #total number of states

#Mosquito Parameters
# S, E, I (IC)
# b_sy, b_asy, b_su, u_m
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
seed = 67

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
    with h5py.File("VaccineModel1patch5.h5", "w") as f: 
        for i, t, x, r0, rt, rngMatrix in results: 
            grp = f.create_group(f"sim_{i}") 
            grp.create_dataset("time", data=t) 
            grp.create_dataset("states", data=x)
            grp.create_dataset("r0", data=r0)
            grp.create_dataset("rt", data=rt)
            grp.create_dataset("rng", data=rngMatrix)


# Simulation and Recording Time
# start = time.time()
# sim = mjsf.jsf(IC, rates, stoich, t_max, config=my_opts, timeShow=timeShow, method="exact")
# end = time.time() 
# simTime = round(end - start,3)
# print(f"Simulation time: {simTime} seconds")

# #Rt
# r0 = rCalc.calculateR0(n, hPatch, mPatch, dPatch, mParam, timeShow, rngMatrix, tStep, sim[1])
# rt = rCalc.calculateRt(n, k, hPatch, mPatch, dPatch, mParam, timeShow, rngMatrix, tStep, sim)
# h.printR0(sim, r0)
# h.printRt(sim, rt)

# #Printing 
# combineInfectedStates = 1 # Combine infected states into one line
# h.printAllPatch(n, k, sim, combineInfectedStates, hComp, mComp, my_opts)
# h.printOverlapODE(n, k, sim, hComp, mComp, my_opts, hPatch, mPatch, dPatch, mParam, IC, t_max, rngMatrix, tStep)

# Average simulations
# totSim = 25
# ciVec = [95, 80, 50] #Confidence regions
# x = [[None for _ in range(totSim)] for _ in range(n*k)] # x[i][j] i = state num, j = sim num
# RtVec = []
# for i in range(totSim):
#     print(f"Simulation {i+1}")
#     rngMatrix = m.generateTempRainRNG(n, t_max/tStep, seed)
#     rates = lambda x, t: m.createRatesVector(x, t, hPatch, mPatch, dPatch, mParam, n, k, comp, rngMatrix, tStep)
#     sim = mjsf.jsf(IC, rates, stoich, t_max, config=my_opts, timeShow=timeShow, method="exact")
#     S_h_t = [sim[0][k*i] for i in range(n)]
#     S_m_t = [sim[0][len(hComp)+k*i] for i in range(n)]
#     RtVec.append(rCalc.calculateRt(n, hPatch, mPatch, dPatch, mParam, sim[1], S_h_t, S_m_t, timeShow))
#     for j in range(n*k):
#        x[j][i] = sim[0][j]      
#        t = sim[1] 
# x = np.array(x)
# h.averagePlots(t, x, RtVec, ciVec, n, k, comp, my_opts)

# Test ODE model
# print(rngMatrix)
# x=m.solveODE(hPatch, mPatch, dPatch, mParam, n, k, IC, t_max, rngMatrix, tStep)
# fig, axs = plt.subplots(n, k)
# for i in range(n): #no states
#     for j in range(k):
#        axs[i, j].plot(x.t, x.y[j+i*k], color='b')
#        axs[i, j].set_title(f"{comp[j]} (Patch {i+1})")
# plt.show()       