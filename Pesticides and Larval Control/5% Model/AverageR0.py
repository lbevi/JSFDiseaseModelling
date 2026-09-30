# n-Patch Model

#To-do:

# Packages
import numpy as np
import random
import matplotlib.pyplot as plt
import sympy as sp
import sys
import mjsf
import time
import h5py
import os
import multiprocessing as mp
import model as m
import helper as h


# Recursion Limit
sys.setrecursionlimit(100000)

#Number of Simulations
numSim = 100

#Working directory
os.chdir("D:/Storage/Work/Homework/Thesis/Code/Final Code/Pesticides and Larval Control")

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

n = 5; #patch size
k = 10 #number of compartments
# Load back simulations
sim = []

def load_simulation(filepath):
    with h5py.File(filepath, "r") as f:
        keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))

        sim = [(f[k]["states"][:], f[k]["time"][:]) for k in keys]
        r0 = [f[k]["r0"][:] for k in keys]
        rt = [f[k]["rt"][:] for k in keys]
        rng = [f[k]["rng"][:] for k in keys]

    return sim, r0, rt, rng


# Load the three simulations
sim1, nointR0, rt1, rng1 = load_simulation(
    "D:/Storage/Work/Homework/Thesis/Code/Final Code/Base Model Simulations/o = 2000/simBaseModelo=2000.h5"
)

sim1, onepatchr0, rt1, rng1 = load_simulation(
    "D:/Storage/Work/Homework/Thesis/Code/Final Code/Vaccine Model/1-patch/VaccineModel1patch.h5"
)

sim1, allpatchr0, rt1, rng1 = load_simulation(
    "D:/Storage/Work/Homework/Thesis/Code/Final Code/Vaccine Model/allpatch/VaccineModel.h5"
)
sim1, r05, rt1, rng1 = load_simulation(
    "D:/Storage/Work/Homework/Thesis/Code/Final Code/Pesticides and Larval Control/VaccineModel1Patch5/VaccineModel1patch5.h5"
)

sim2, r025, rt2, rng2 = load_simulation(
    "D:/Storage/Work/Homework/Thesis/Code/Final Code/Pesticides and Larval Control/VaccineModel1Patch25/VaccineModel1patch25.h5"
)

sim3, r050, rt3, rng3 = load_simulation(
    "D:/Storage/Work/Homework/Thesis/Code/Final Code/Pesticides and Larval Control/VaccineModel1Patch50/VaccineModel1patch50.h5"
)       

# Data list
R0 = np.zeros((6, 3))

# Convert to np array
onepatchr0 = np.array(onepatchr0)
allpatchr0 = np.array(allpatchr0)
r05 = np.array(r05)
r025 = np.array(r025)
r050 = np.array(r050)

# Calculate averages
R0[0,0] = np.mean(nointR0)
R0[1,0] = np.mean(allpatchr0[:,1000:5000])
R0[2,0] = np.mean(onepatchr0[:,1000:5000])
R0[3,0] = np.mean(r05[:,1000:5000])
R0[4,0] = np.mean(r025[:,1000:5000])
R0[5,0] = np.mean(r050[:,1000:5000])

#Human Parameters
# S, E, I, A, D, R, (IC)
# u_h, b_h 
hP1 = m.createHumanPatch(25000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 0, 0)
hP2 = m.createHumanPatch(1000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 0, 0)
hP3 = m.createHumanPatch(50, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 0, 0)
hP4 = m.createHumanPatch(5000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 0, 0)
hP5 = m.createHumanPatch(15, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 0, 0)

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

R0[0,1] = h.calculateLocalR0(hPatch, mPatch, dPatch, 0, 1)
R0[0,2] = h.calculateLocalR0(hPatch, mPatch, dPatch, 1, 1)

# All-patch
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

R0[1,1] = h.calculateLocalR0(hPatch, mPatch, dPatch, 0, 1)
R0[1,2] = h.calculateLocalR0(hPatch, mPatch, dPatch, 1, 1)

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

R0[2,1] = h.calculateLocalR0(hPatch, mPatch, dPatch, 0, 1)
R0[2,2] = h.calculateLocalR0(hPatch, mPatch, dPatch, 1, 1)

#5% 
mP1 = m.createMosquitoPatch(50000, 0, 50,
                               0.4, 0.6, 0.6, 0.0477*1.05)
mPatch = [mP1, mP2, mP3, mP4, mP5]

R0[3,1] = h.calculateLocalR0(hPatch, mPatch, dPatch, 0, 0.95)
R0[3,2] = h.calculateLocalR0(hPatch, mPatch, dPatch, 1, 1)

#25% 
mP1 = m.createMosquitoPatch(50000, 0, 50,
                               0.4, 0.6, 0.6, 0.0477*1.25)
mPatch = [mP1, mP2, mP3, mP4, mP5]

R0[4,1] = h.calculateLocalR0(hPatch, mPatch, dPatch, 0, 0.75)
R0[4,2] = h.calculateLocalR0(hPatch, mPatch, dPatch, 1, 1)

#50% 
mP1 = m.createMosquitoPatch(50000, 0, 50,
                               0.4, 0.6, 0.6, 0.0477*1.50)
mPatch = [mP1, mP2, mP3, mP4, mP5]

R0[5,1] = h.calculateLocalR0(hPatch, mPatch, dPatch, 0, 0.5)
R0[5,2] = h.calculateLocalR0(hPatch, mPatch, dPatch, 1, 1)

print(R0)