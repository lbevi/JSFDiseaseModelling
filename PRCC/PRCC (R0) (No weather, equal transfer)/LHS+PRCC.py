import numpy as np
import pandas as pd
from scipy.stats import qmc, rankdata
import pingouin as pg
import sys
import model as m
import Rt as rCalc
import h5py
import os
import multiprocessing as mp
import matplotlib.pyplot as plt

# LHS Sampling
paramBounds = {
    '$v_1$': [0.001, 0.1], #vaccination rate per day (patch 1)
    '$v_2$': [0.001, 0.1], #vaccination rate per day (patch 2)
    '$v_3$': [0.001, 0.1], #vaccination rate per day (patch 3)
    '$v_4$': [0.001, 0.1], #vaccination rate per day (patch 4)
    '$v_5$': [0.001, 0.1], #vaccination rate per day (patch 5)
    'beta': [0.01, 0.80], #vaccine efficacy
    '$b_{sy1}$': [0.01, 0.4], #bite rate per day on symptomatic human (patch 1)
    '$b_{sy2}$': [0.01, 0.4], #bite rate per day on symptomatic human (patch 2)
    '$b_{sy3}$': [0.01, 0.4], #bite rate per day on symptomatic human (patch 3)
    '$b_{sy4}$': [0.01, 0.4], #bite rate per day on symptomatic human (patch 4)
    '$b_{sy5}$': [0.01, 0.4], #bite rate per day on symptomatic human (patch 5)
    '$b_{asy1}$': [0.01, 0.6], #bite rate per day on asympto/susceptible human (patch 1)
    '$b_{asy2}$': [0.01, 0.6], #bite rate per day on asympto/susceptible human (patch 2)
    '$b_{asy3}$': [0.01, 0.6], #bite rate per day on asympto/susceptible human (patch 3)
    '$b_{asy4}$': [0.01, 0.6], #bite rate per day on asympto/susceptible human (patch 4)
    '$b_{asy5}$': [0.01, 0.6], #bite rate per day on asympto/susceptible human (patch 5)
    '$b_{av1}$': [0.0001, 0.05], #Average birth rate of mosquitoes (patch 1)
    '$b_{av2}$': [0.0001, 0.05], #Average birth rate of mosquitoes (patch 2)
    '$b_{av3}$': [0.0001, 0.05], #Average birth rate of mosquitoes (patch 3)
    '$b_{av4}$': [0.0001, 0.05], #Average birth rate of mosquitoes (patch 4)
    '$b_{av5}$': [0.0001, 0.05], #Average birth rate of mosquitoes (patch 5)
    '$u_{m1}$': [0.0477, 0.1], #Average death rate of mosquitoes (patch 1)
    '$u_{m2}$': [0.0477, 0.1], #Average death rate of mosquitoes (patch 2)
    '$u_{m3}$': [0.0477, 0.1], #Average death rate of mosquitoes (patch 3)
    '$u_{m4}$': [0.0477, 0.1], #Average death rate of mosquitoes (patch 4)
    '$u_{m5}$': [0.0477, 0.1], #Average death rate of mosquitoes (patch 5)
}
nSamples = 1000000
nParams = len(paramBounds)

#Initialise LHS sampler and sample
sampler = qmc.LatinHypercube(d=nParams)
sampleRaw = sampler.random(n=nSamples)
lowerBounds = [bounds[0] for bounds in paramBounds.values()]
upperBounds = [bounds[1] for bounds in paramBounds.values()]
samples = qmc.scale(sampleRaw, lowerBounds, upperBounds)

#Calculate averaged R0 for samples 
R0_vec = np.zeros(nSamples)

def compute_R0(samples):
    # Extract each parameter from samples
    v1 = samples[0]; v2 = samples[1]; v3 = samples[2]; v4 = samples[3]; v5 = samples[4]
    beta = samples[5]
    b_sy1 = samples[6]; b_sy2 = samples[7]; b_sy3 = samples[8]; b_sy4 = samples[9]; b_sy5 = samples[10]
    b_asy1 = samples[11]; b_asy2 = samples[12]; b_asy3 = samples[13]; b_asy4 = samples[14]; b_asy5 = samples[15]
    b_av1 = samples[16]; b_av2 = samples[17]; b_av3 = samples[18]; b_av4 = samples[19]; b_av5 = samples[20]
    u_m1 = samples[21]; u_m2 = samples[22]; u_m3 = samples[23]; u_m4 = samples[24]; u_m5 = samples[25]

    # Prepare patch matrices
    # Compartments and Patch Number
    comp = ["S_h", "E_h", "I_h", "A_h", "D_h", "R_h", "S_m", "E_m", "I_m", "V_h"]
    hComp = ["S", "E", "I", "A", "D", "R", "V"]
    mComp = ["S", "E", "I"]
    k = len(comp) # number of compartments per patch

    # Human Parameters
    # S, E, I, A, D, R, V (IC)
    # u_h, b_h, v, phi, beta
    hP1 = m.createHumanPatch(25000, 0, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4, v1, 1/365, beta)
    hP2 = m.createHumanPatch(1000, 0, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4, v2, 1/365, beta)
    hP3 = m.createHumanPatch(50, 0, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4, v3, 1/365, beta)
    hP4 = m.createHumanPatch(5000, 0, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4, v4, 1/365, beta)
    hP5 = m.createHumanPatch(15, 0, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4, v5, 1/365, beta)

    hPatch = [hP1, hP2, hP3, hP4, hP5]

    n = len(hPatch); #patch size
    tot_states = n*k #total number of states

    # Mosquito Parameters
    # S, E, I (IC)
    # b_sy, b_asy, b_su, u_m

    mP1 = m.createMosquitoPatch(50000, 0, 50,
                                b_sy1, b_asy1, b_asy1, u_m1)
    mP2 = m.createMosquitoPatch(2000, 0, 0,
                                b_sy2, b_asy2, b_asy2, u_m2)
    mP3 = m.createMosquitoPatch(100, 0, 0,
                                b_sy3, b_asy3, b_asy3, u_m3)
    mP4 = m.createMosquitoPatch(10000, 0, 0,
                                b_sy4, b_asy4, b_asy4, u_m4)
    mP5 = m.createMosquitoPatch(30, 0, 0,
                                b_sy5, b_asy5, b_asy5, u_m5)

    mPatch = [mP1, mP2, mP3, mP4, mP5]

    # Disease Parameters
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

    tH[0,1] = 0.001; tH[1,0] = 0.001 #edge 1<->2
    tH[0,2] = 0.001; tH[2,0] = 0.001 #edge 1<->3
    tH[0,3] = 0.001; tH[3,0] = 0.001 #edge 1<->4
    tH[1,2] = 0.001; tH[2,1] = 0.001 #edge 2<->3
    tH[3,4] = 0.001; tH[4,3] = 0.001 #edge 4<->5

    mParam = {}
    for c in comp:
        if c.endswith("_h"):
            mParam[c] = tH.copy()
        elif c.endswith("_m"):
            mParam[c] = tM.copy()

    mParam["D_h"] = np.zeros((n,n))

    # IC Vector
    IC = m.createIC(hPatch, mPatch, hComp, mComp, n)

    # Birth vector 
    birthVec = [b_av1, b_av2, b_av3, b_av4, b_av5]

    # Run R0 calculation
    return rCalc.calculateR0(n, hPatch, mPatch, dPatch, mParam, birthVec)

# Multiprocessing
if __name__ == "__main__":

    n_cpus = 100

    with mp.Pool(processes=n_cpus) as pool:
        R0_vec = np.array(
            pool.map(compute_R0, samples)
        )
        
# Convert to pandas and initialise
df = pd.DataFrame(samples, columns=paramBounds.keys())
df["R0"] = R0_vec
PRCCResults = []

# Compute PRCC and p-value for each param
for param in paramBounds.keys():
    covars = [c for c in paramBounds.keys() if c != param]
    res = pg.partial_corr(data=df,x=param,y="R0",covar=covars,method="spearman")
    PRCCResults.append([param,res["r"].iloc[0], res["p-val"].iloc[0]])

PRCCResults = pd.DataFrame(PRCCResults,columns=["Parameter", "PRCC", "p-value"])
PRCCResults.to_csv('PRCCvals', index=False)

# Print table
print(PRCCResults)

# Print bar graph
plt.figure(figsize=(12,6))
plt.bar(PRCCResults["Parameter"], PRCCResults["PRCC"])
plt.axhline(0, linewidth=1)

plt.xlabel("Parameter")
plt.ylabel("PRCC")
plt.title("Partial Rank Correlation Coefficients for $R_0$")

plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("Partial Rank Correlation Coefficients for $R_0$", dpi=1000, bbox_inches="tight")