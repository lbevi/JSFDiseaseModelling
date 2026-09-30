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
                             4.5e-5, 1.2e-4, 0.02, 1/365, 0.25)
hP2 = m.createHumanPatch(1000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0.02, 1/365, 0.25)
hP3 = m.createHumanPatch(50, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0.02, 1/365, 0.25)
hP4 = m.createHumanPatch(5000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0.02, 1/365, 0.25)
hP5 = m.createHumanPatch(15, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0.02, 1/365, 0.25)

hPatch = [hP1, hP2, hP3, hP4, hP5]

n = len(hPatch); #patch size
tot_states = n*k #total number of states

# Mosquito Parameters
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

# Birth vector 
birthVec = [0.03, 0.03, 0.03, 0.03, 0.03]

mum = np.linspace(0.01, 0.99, 20)

R0Vec = []
for x in mum:

    hP1 = m.createHumanPatch(25000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0.02, 1/365, x)
    hP2 = m.createHumanPatch(1000, 0, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4, 0.02, 1/365, x)
    hP3 = m.createHumanPatch(50, 0, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4, 0.02, 1/365, x)
    hP4 = m.createHumanPatch(5000, 0, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4, 0.02, 1/365, x)
    hP5 = m.createHumanPatch(15, 0, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4, 0.02, 1/365, x)

    hPatch = [hP1, hP2, hP3, hP4, hP5]
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
    birthVec = [0.03, 0.03, 0.03, 0.03, 0.03]
    R0Vec.append(rCalc.calculateR0(n, hPatch, mPatch, dPatch, mParam, birthVec))

plt.figure(figsize=(12,6))
plt.plot(mum, R0Vec)
plt.show()