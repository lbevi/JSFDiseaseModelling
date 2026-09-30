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

# Recursion Limit
sys.setrecursionlimit(100000)

#Number of Simulations
numSim = 100

#Working directory
os.chdir("D:/Storage/Work/Homework/Thesis/Code/Final Code/Vaccine Model")


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
sim1, r01, rt1, rng1 = load_simulation(
    "D:/Storage/Work/Homework/Thesis/Code/Final Code/Vaccine Model/1-patch/VaccineModel1patch.h5"
)

sim2, r02, rt2, rng2 = load_simulation(
    "D:/Storage/Work/Homework/Thesis/Code/Final Code/Vaccine Model/allpatch/VaccineModel.h5"
)

sim3, r03, rt3, rng3 = load_simulation(
    "D:/Storage/Work/Homework/Thesis/Code/Final Code/Vaccine Model/extreme/VaccineModelExtreme.h5"
)       

# Human and mosquito partial elim
fig, axs = plt.subplots(n, 3, squeeze=False, constrained_layout=True, figsize=(15, 15))
N = numSim
for m in range(3):
    tElimBoth = [[] for i in range(n)] 
    countElimBoth = [0 for i in range(n)]
    for i in range(n):   
        countElim = 0      
        for j in range(N):
            if m == 0:    
                totalI = sim1[j][0][1+i*k] + sim1[j][0][2+i*k] + sim1[j][0][3+i*k] + sim1[j][0][7+i*k] + sim1[j][0][8+i*k]
            if m == 1:    
                totalI = sim2[j][0][1+i*k] + sim2[j][0][2+i*k] + sim2[j][0][3+i*k] + sim2[j][0][7+i*k] + sim2[j][0][8+i*k]
            if m == 2:
                totalI = sim3[j][0][1+i*k] + sim3[j][0][2+i*k] + sim3[j][0][3+i*k] + sim3[j][0][7+i*k] + sim3[j][0][8+i*k]    
            idx = np.where((totalI == 0))[0]
            idx = idx[idx>1000] #must be at least after t=100
            if idx.size:
                if m == 0:
                    t = sim1[j][1][idx[0]]
                if m == 1:
                    t = sim2[j][1][idx[0]]
                if m == 2: 
                    t = sim3[j][1][idx[0]]         
                tElimBoth[i].append(t)
                countElim += 1    
        if m == 0:
            s = "1"
        if m == 1:
            s = "A"
        if m == 2: 
            s = "E"
        if countElim == 0:
            axs[i, m].set_xticks([])
            axs[i, m].set_yticks([])                        
        countElimBoth[i] = countElim
        axs[i, 0].annotate( f"Patch {i+1}",xy=(-0.15, 0.5), xycoords="axes fraction",rotation=90,ha="center",va="center",fontsize=14)
        axs[i, m].text(0.98, 0.97, rf"$p_{{{s}({i+1})}} = {countElim/N}$", transform=axs[i, m].transAxes,ha="right",va="top", fontsize=14)
        if countElim:
            axs[i, m].text(0.98, 0.75, rf"$SE_{{{s}({i+1})}}  = {np.std(tElimBoth[i], ddof=1)/np.sqrt(countElim):.0f}$", transform=axs[i, m].transAxes,ha="right",va="top", fontsize=14)     
            axs[i, m].text(0.98, 0.86, rf"$t_{{{s}({i+1})}} = {np.mean(tElimBoth[i]):.0f}$", transform=axs[i, m].transAxes,ha="right",va="top", fontsize=14)
            axs[i, m].axvline(x=np.mean(tElimBoth[i]), color="r", linestyle="--")
        axs[i, m].hist(tElimBoth[i], bins=30)
        axs[i, m].tick_params(axis='both', labelsize=14)
    axs[0, 0].set_title("1-patch", fontsize=14)
    axs[0, 1].set_title("All-patch", fontsize=14)
    axs[0, 2].set_title("Extreme", fontsize=14)
    axs[n-1, m].set_xlabel("Time (days)", fontsize = 14)    
plt.savefig("VacModelAllModelsPartialElim", dpi=150, bbox_inches="tight")
plt.show()

