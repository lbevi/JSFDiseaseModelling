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
import model as m
import Rt as rCalc
import helper as h
import h5py
import os
import multiprocessing as mp

# Recursion Limit
sys.setrecursionlimit(100000)

#Working directory
os.chdir("D:/Storage/Work/Homework/Thesis/Code/Final Code/Base Model Simulations/o = 500")

# Print settings
np.set_printoptions(threshold=np.inf)   


with h5py.File("simBaseModelo=500.h5", "r") as f:
    keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))
    sim = [(f[k]["states"][:], f[k]["time"][:]) for k in keys] #i = sim num, j = states/time, k = specific state
    r0 =  [(f[k]["r0"][:]) for k in keys] 
    rt =  [(f[k]["rt"][:]) for k in keys]
    rng = [(f[k]["rng"][:]) for k in keys]         

iComp= [sim[1][0][2+9*k] for k in range(0,5)]
iComp.extend([sim[1][0][3+9*k] for k in range(0,5)])
np.savetxt("vector.csv", iComp, delimiter=",")

print(len(iComp[0]))

fig, axs = plt.subplots(2, 3, squeeze=False, constrained_layout=True, figsize=(15, 1.5*2*3))
axs[0, 0].plot(sim[1][1], iComp[0], label = "I")
axs[0, 0].plot(sim[1][1], iComp[5], label = "A")
axs[0, 0].axhline(y=500, color="k", linestyle="--")
axs[0, 0].set_xticks([])
axs[0, 0].tick_params(axis='both', labelsize=14)
axs[0, 0].set_ylabel("Human Population", fontsize=14)
axs[0, 0].set_title(f"Patch 1", fontsize=14)

axs[0, 1].plot(sim[1][1], iComp[1], label = "I")
axs[0, 1].plot(sim[1][1], iComp[6], label = "A")
axs[0, 1].set_xticks([])
axs[0, 1].tick_params(axis='both', labelsize=14)
axs[0, 1].set_title(f"Patch 2", fontsize=14)

axs[0, 2].plot(sim[1][1], iComp[2], label = "I")
axs[0, 2].plot(sim[1][1], iComp[7], label = "A")
axs[0, 2].tick_params(axis='both', labelsize=14)
axs[0, 2].set_title(f"Patch 3", fontsize=14)
axs[0, 2].set_xlabel("Time (days)", fontsize=14)

axs[1, 0].plot(sim[1][1], iComp[3], label = "I")
axs[1, 0].plot(sim[1][1], iComp[8], label = "A")
axs[1, 0].axhline(y=500, color="k", linestyle="--")
axs[1, 0].set_xlabel("Time (days)", fontsize=14)
axs[1, 0].tick_params(axis='both', labelsize=14)
axs[1, 0].set_ylabel("Human Population", fontsize=14)
axs[1, 0].set_title(f"Patch 4", fontsize=14)

axs[1, 1].plot(sim[1][1], iComp[4], label = "I")
axs[1, 1].plot(sim[1][1], iComp[9], label = "A")
axs[1, 1].set_xlabel("Time (days)", fontsize=14)
axs[1, 1].tick_params(axis='both', labelsize=14)
axs[1, 1].set_title(f"Patch 5", fontsize=14)

axs[1,2].axis("off")
plt.savefig("NPE Data", dpi=500, bbox_inches="tight")
plt.show()

# plt.figure()
# plt.plot(sim[1][1], iComp[0])
# plt.plot(sim[1][1], iComp[1])
# plt.plot(sim[1][1], iComp[2])
# plt.plot(sim[1][1], iComp[3])
# plt.plot(sim[1][1], iComp[4])
# plt.show()

# plt.figure()
# plt.plot(sim[1][1], iComp[5])
# plt.plot(sim[1][1], iComp[6])
# plt.plot(sim[1][1], iComp[7])
# plt.plot(sim[1][1], iComp[8])
# plt.plot(sim[1][1], iComp[9])
# plt.show()