import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sympy as sp
import sys
import mjsf
import time
import model as m
import Rt as c
from model import V_ON

# Bonus helper functions

# Print Rt graph
def printR0(sim, R0):
    plt.figure()
    plt.plot(sim[1], R0, label=r"$R_0$")
    plt.xlabel("Time (days)")
    plt.ylabel(r"$R_0$")
    plt.title(r"$R_0$ vs Time")
    plt.legend()
    plt.savefig("VacModelR01patch100", dpi=1000, bbox_inches="tight")
    plt.show()

def printRt(sim, Rt):
    plt.figure()
    plt.plot(sim[1], Rt, label=r"$R_t$")
    plt.xlabel("Time (days)")
    plt.ylabel(r"$R_t$")
    plt.title(r"$R_t$ vs Time")
    plt.legend()
    plt.savefig("VacModelRt1patch100", dpi=1000, bbox_inches="tight")
    plt.show()    

# Print All Patches
def printAllPatch(n, k, sim, combineInfectedStates, hComp, mComp, my_opts):

    Thresh = my_opts["SwitchingThreshold"][1]

    fig, axs = plt.subplots(n, 4, squeeze=False, constrained_layout=True, figsize=(30, 4*n))
    for i in range(n):
        # Human states
        if combineInfectedStates == 0:
             for h in hComp:            
                axs[i, 0].plot(sim[1], sim[0][hComp.index(h)+i*k], label = h)
        else:
            axs[i, 0].plot(sim[1], sim[0][0+i*k], label = "S")           
            axs[i, 0].plot(sim[1], np.array(sim[0][1+i*k])  + np.array(sim[0][2+i*k]) + np.array(sim[0][3+i*k]), label = "E+I+A")
            axs[i, 0].plot(sim[1], sim[0][4+i*k], label = "D") 
            axs[i, 0].plot(sim[1], sim[0][5+i*k], label = "R")
            axs[i, 0].plot(sim[1], sim[0][9+i*k], label = "V") 

        # Include threshold line if any population is greater than the threshold

        for j in range(len(hComp)):
            if np.any(np.array(sim[0][0+i*k]) > Thresh) or np.any(np.array(sim[0][4+i*k]) > Thresh) or np.any(np.array(sim[0][5+i*k]) > Thresh) or np.any(np.array(sim[0][9+i*k]) > Thresh):
                axs[i, 0].axhline(y=Thresh, color="k", linestyle="--")
                break        

        if combineInfectedStates == 1:
            if np.any(np.array(sim[0][2+i*k]) > Thresh) or np.any(np.array(sim[0][3+i*k]) > Thresh):       
                axs[i, 0].axhline(y=Thresh, color="k", linestyle="--")

        axs[i, 0].set_xlabel("Time (days)")
        axs[i, 0].set_ylabel("Human Population")
        axs[i, 0].set_title(f"Human States vs Time (Patch {i+1})")
        axs[i, 0].legend()

        # Mosquito states
        toggleLine = 0
        axs[i, 1].plot(sim[1], sim[0][6+i*k], label = "S")
        axs[i, 1].plot(sim[1], sim[0][7+i*k], label = "E")
        axs[i, 1].plot(sim[1], sim[0][8+i*k], label = "I")
        if toggleLine == 0:
            if np.any(np.array(sim[0][6+i*k]) > Thresh) or np.any(np.array(sim[0][7+i*k]) > Thresh) or np.any(np.array(sim[0][8+i*k]) > Thresh):
                axs[i, 1].axhline(y=Thresh, color="k", linestyle="--")
                toggleLine = 1
        
        axs[i, 1].set_xlabel("Time (days)")
        axs[i, 1].set_ylabel("Mosquito Population")
        axs[i, 1].set_title(f"Mosquito States vs Time (Patch {i+1})")
        axs[i, 1].legend()

        # Total human states
        hPop = np.array(sim[0][0+i*k]) + np.array(sim[0][1+i*k]) + np.array(sim[0][2+i*k]) + np.array(sim[0][3+i*k]) + np.array(sim[0][5+i*k]) + np.array(sim[0][9+i*k])
        axs[i, 2].plot(sim[1], hPop, label = "Total Population")
        axs[i, 2].set_xlabel("Time (days)")
        axs[i, 2].set_ylabel("Total Human Population")
        axs[i, 2].set_title(f"Total Human Population vs Time (Patch {i+1})")
        axs[i, 2].legend()

        # Total mosquito states
        mPop = np.array(sim[0][6+i*k]) + np.array(sim[0][7+i*k]) + np.array(sim[0][8+i*k])
        axs[i, 3].plot(sim[1], mPop, label = "Total Population")
        axs[i, 3].set_xlabel("Time (days)")
        axs[i, 3].set_ylabel("Total Mosquito Population")
        axs[i, 3].set_title(f"Total Mosquito Population vs Time (Patch {i+1})")
        axs[i, 3].legend()
    plt.savefig("VacModelAllPatches1patch50", dpi=500, bbox_inches="tight")    
    plt.show()

# Print All Patches with ODE solution

def printOverlapODE(n, k, sim, hComp, mComp, my_opts, hPatch, mPatch, dPatch, mParam, IC, t_max, rngMatrix, tStep):
    Thresh = my_opts["SwitchingThreshold"][1]

    ODE_sol = m.solveODE(hPatch, mPatch, dPatch, mParam, n, k, IC, t_max, rngMatrix, tStep)

    fig, axs = plt.subplots(n, 6, squeeze=False, constrained_layout=True, figsize=(30, 4*n))
    
    for i in range(n):
        #S, V R
        axs[i, 0].plot(sim[1], sim[0][0+i*k], label = r"$S_{JSF}$")
        axs[i, 0].plot(sim[1], sim[0][5+i*k], label = r"$R_{JSF}$")
        axs[i, 0].plot(sim[1], sim[0][9+i*k], label = r"$V_{JSF}$")            
        axs[i, 0].plot(ODE_sol.t, ODE_sol.y[i*k], label = r"$S_{ODE}$")
        axs[i, 0].plot(ODE_sol.t, ODE_sol.y[5+i*k], label = r"$R_{ODE}$")
        axs[i, 0].plot(ODE_sol.t, ODE_sol.y[9+i*k], label = r"$V_{ODE}$")
        axs[i, 0].set_xlabel("Time (days)")
        axs[i, 0].set_ylabel("Human Population")
        axs[i, 0].set_title(f"Human States ODE, JSF (Patch {i+1})")
        axs[i, 0].legend()    
        if np.any(np.array(sim[0][0+i*k]) > Thresh) or np.any(np.array(sim[0][5+i*k]) > Thresh):
            axs[i, 0].axhline(y=Thresh, color="k", linestyle="--")

        # Inf states    
        axs[i, 1].plot(sim[1], np.array(sim[0][1+i*k])  + np.array(sim[0][2+i*k]) + np.array(sim[0][3+i*k]), label = r"$E+I+A_{JSF}$")
        axs[i, 1].plot(ODE_sol.t, ODE_sol.y[1+i*k]  + ODE_sol.y[2+i*k] + ODE_sol.y[3+i*k], label = r"$E+I+A_{ODE}$")
        axs[i, 1].set_xlabel("Time (days)")
        axs[i, 1].set_ylabel("Human Population")
        axs[i, 1].set_title(f"Infected Human ODE, JSF (Patch {i+1})")
        axs[i, 1].legend()
        if np.any(np.array(sim[0][1+i*k]) > Thresh) or np.any(np.array(sim[0][2+i*k])> Thresh) or np.any(np.array(sim[0][3+i*k]) > Thresh):
            axs[i, 1].axhline(y=Thresh, color="k", linestyle="--")   

        #D
        axs[i, 2].plot(sim[1], sim[0][4+i*k], label = r"$D_{JSF}$")           
        axs[i, 2].plot(ODE_sol.t, ODE_sol.y[4+i*k], label = r"$D_{ODE}$")
        axs[i, 2].set_xlabel("Time (days)")
        axs[i, 2].set_ylabel("Human Population")
        axs[i, 2].set_title(f"Dead Humans ODE, JSF (Patch {i+1})")
        axs[i, 2].legend()
        if np.any(np.array(sim[0][4+i*k]) > Thresh):
            axs[i, 2].axhline(y=Thresh, color="k", linestyle="--")

        #Mosquito states
        axs[i, 3].plot(sim[1], sim[0][6+i*k], label = r"$S_{JSF}$")
        axs[i, 3].plot(sim[1], sim[0][7+i*k], label = r"$E_{JSF}$")
        axs[i, 3].plot(sim[1], sim[0][8+i*k], label = r"$I_{JSF}$")    
        axs[i, 3].plot(ODE_sol.t, ODE_sol.y[6+i*k], label = r"$S_{ODE}$")       
        axs[i, 3].plot(ODE_sol.t, ODE_sol.y[7+i*k], label = r"$E_{ODE}$")
        axs[i, 3].plot(ODE_sol.t, ODE_sol.y[8+i*k], label = r"$I_{ODE}$")
        axs[i, 3].set_xlabel("Time (days)")
        axs[i, 3].set_ylabel("Mosquito Population")
        axs[i, 3].set_title(f"Mosquito ODE, JSF (Patch {i+1})")
        axs[i, 3].legend()
        if np.any(np.array(sim[0][6+i*k]) > Thresh) or np.any(np.array(sim[0][7+i*k]) > Thresh) or np.any(np.array(sim[0][8+i*k]) > Thresh):
            axs[i, 3].axhline(y=Thresh, color="k", linestyle="--")

        # Total human states
        hPop_JSF = np.array(sim[0][0+i*k]) + np.array(sim[0][1+i*k]) + np.array(sim[0][2+i*k]) + np.array(sim[0][3+i*k]) + np.array(sim[0][5+i*k]) + np.array(sim[0][9+i*k])
        hPop_ODE = ODE_sol.y[0+i*k] + ODE_sol.y[1+i*k] + ODE_sol.y[2+i*k] + ODE_sol.y[3+i*k] + ODE_sol.y[5+i*k] + ODE_sol.y[9+i*k]
        axs[i, 4].plot(sim[1], hPop_JSF, label = "Total Population JSF")
        axs[i, 4].plot(ODE_sol.t, hPop_ODE, label = "Total Population ODE")
        axs[i, 4].set_xlabel("Time (days)")
        axs[i, 4].set_ylabel("Total Human Population")
        axs[i, 4].set_title(f"Total Human ODE, JSF (Patch {i+1})")
        axs[i, 4].legend()

        # Total mosquito states
        mPop_JSF = np.array(sim[0][6+i*k]) + np.array(sim[0][7+i*k]) + np.array(sim[0][8+i*k])
        mPop_ODE = ODE_sol.y[6+i*k] + ODE_sol.y[7+i*k] + ODE_sol.y[8+i*k]
        axs[i, 5].plot(sim[1], mPop_JSF, label = "Total Population JSF")
        axs[i, 5].plot(ODE_sol.t, mPop_ODE, label = "Total Population ODE")
        axs[i, 5].set_xlabel("Time (days)")
        axs[i, 5].set_ylabel("Total Mosquito Population")
        axs[i, 5].set_title(f"Total Mosquito ODE, JSF (Patch {i+1})")
        axs[i, 5].legend()
    plt.savefig("VacModelAllPatches+ODE1patch50", dpi=500, bbox_inches="tight")    
    plt.show()

def averagePlots(t, x, R0Vec, RtVec, ciVec, n, k, comp, my_opts, xODE):
    Thresh = my_opts["SwitchingThreshold"][1]

    # Plot Rt
    plt.figure()
    for ci in ciVec:
        low = np.percentile(R0Vec, 50 - ci / 2, axis=0)
        high = np.percentile(R0Vec, 50 + ci / 2, axis=0)
        plt.fill_between(t, low, high, color='r', alpha=0.2)
    median = np.median(R0Vec, axis=0) # Plot Median
    plt.plot(t, median, color='b', label = "Median")
    plt.xlabel("Time (days)")
    plt.ylabel(r"$R_0$")
    plt.title(r"CI of $R_0$ vs Time")
    plt.legend()
    plt.margins(x=0)
    plt.savefig("VacModelCIR01patch50", dpi=1000, bbox_inches="tight")
    plt.show()

    # Plot Rt
    plt.figure()
    for ci in ciVec:
        low = np.percentile(RtVec, 50 - ci / 2, axis=0)
        high = np.percentile(RtVec, 50 + ci / 2, axis=0)
        plt.fill_between(t, low, high, color='r', alpha=0.2)
    median = np.median(RtVec, axis=0) # Plot Median
    plt.plot(t, median, color='b', label = "Median")
    plt.xlabel("Time (days)")
    plt.ylabel(r"$R_t$")
    plt.title(r"CI of $R_t$ vs Time")
    plt.legend()
    plt.margins(x=0)
    plt.savefig("VacModelCIRt1patch50", dpi=1000, bbox_inches="tight")
    plt.show()

    # Plot states
    fig, axs = plt.subplots(n, 9, squeeze=False, constrained_layout=True, figsize=(40, 4*n))
    for i in range(n): #no states
        # Plot individual states
        # Plot S_h
        for ci in ciVec:
            low = np.percentile(x[0+i*k], 50 - ci / 2, axis=0)
            high = np.percentile(x[0+i*k], 50 + ci / 2, axis=0)
            axs[i, 0].fill_between(t, low, high, color='r', alpha=0.2)
        median = np.median(x[0+i*k], axis=0) # Plot Median
        axs[i, 0].plot(t, median, color='b', label = "Median")
        axs[i, 0].set_xlabel("Time (days)")
        axs[i, 0].set_ylabel(r"$S_H$")
        axs[i, 0].set_title(rf"$S_H$ (Patch {i+1})")
        if np.any(np.array(x[0+i*k]) > Thresh):
            axs[i, 0].axhline(y=Thresh, color="k", linestyle="--")
        axs[i, 0].plot(xODE.t, xODE.y[0+i*k], color='g', label = 'ODE')
        axs[i, 0].legend()
        
        # Plot E_h + I_h + A_h
        for ci in ciVec:
            low = np.percentile(x[1+i*k] + x[2+i*k]+ x[3+i*k], 50 - ci / 2, axis=0)
            high = np.percentile(x[1+i*k] + x[2+i*k]+ x[3+i*k], 50 + ci / 2, axis=0)
            axs[i, 1].fill_between(t, low, high, color='r', alpha=0.2)
        median = np.median(x[1+i*k] + x[2+i*k]+ x[3+i*k], axis=0) # Plot Median
        axs[i, 1].plot(t, median, color='b', label = "Median")
        axs[i, 1].set_xlabel("Time (days)")
        axs[i, 1].set_ylabel(r"$E_H$+$I_H$+$A_H$")
        axs[i, 1].set_title(rf"$E_H$+$I_H$+$A_H$ (Patch {i+1})")
        axs[i, 1].plot(xODE.t, xODE.y[1+i*k] + xODE.y[2+i*k]+ xODE.y[3+i*k], color='g', label = 'ODE')
        axs[i, 1].legend()

        # Plot D_h
        for ci in ciVec:
            low = np.percentile(x[4+i*k], 50 - ci / 2, axis=0)
            high = np.percentile(x[4+i*k], 50 + ci / 2, axis=0)
            axs[i, 2].fill_between(t, low, high, color='r', alpha=0.2)
        median = np.median(x[4+i*k], axis=0) # Plot Median
        axs[i, 2].plot(t, median, color='b', label = "Median")
        axs[i, 2].set_xlabel("Time (days)")
        axs[i, 2].set_ylabel(r"$D_H$")
        axs[i, 2].set_title(rf"$D_H$ (Patch {i+1})")
        if np.any(np.array(x[4+i*k]) > Thresh):
            axs[i, 2].axhline(y=Thresh, color="k", linestyle="--")
        axs[i, 2].plot(xODE.t, xODE.y[4+i*k], color='g', label = 'ODE')
        axs[i, 2].legend()

        # Plot R_h
        for ci in ciVec:
            low = np.percentile(x[5+i*k], 50 - ci / 2, axis=0)
            high = np.percentile(x[5+i*k], 50 + ci / 2, axis=0)
            axs[i, 3].fill_between(t, low, high, color='r', alpha=0.2)
        median = np.median(x[5+i*k], axis=0) # Plot Median
        axs[i, 3].plot(t, median, color='b', label = "Median")
        axs[i, 3].set_xlabel("Time (days)")
        axs[i, 3].set_ylabel(r"$R_H$")
        axs[i, 3].set_title(rf"$R_H$ (Patch {i+1})")
        if np.any(np.array(x[5+i*k]) > Thresh):
            axs[i, 3].axhline(y=Thresh, color="k", linestyle="--")
        axs[i, 3].plot(xODE.t, xODE.y[5+i*k], color='g', label = 'ODE')
        axs[i, 3].legend()

        # Plot V_h
        for ci in ciVec:
            low = np.percentile(x[9+i*k], 50 - ci / 2, axis=0)
            high = np.percentile(x[9+i*k], 50 + ci / 2, axis=0)
            axs[i, 4].fill_between(t, low, high, color='r', alpha=0.2)
        median = np.median(x[9+i*k], axis=0) # Plot Median
        axs[i, 4].plot(t, median, color='b', label = "Median")
        axs[i, 4].set_xlabel("Time (days)")
        axs[i, 4].set_ylabel(r"$V_H$")
        axs[i, 4].set_title(rf"$V_H$ (Patch {i+1})")
        if np.any(np.array(x[9+i*k]) > Thresh):
            axs[i, 4].axhline(y=Thresh, color="k", linestyle="--")
        axs[i, 4].plot(xODE.t, xODE.y[9+i*k], color='g', label = 'ODE')
        axs[i, 4].legend()


        # Plot S_m
        for ci in ciVec:
            low = np.percentile(x[6+i*k], 50 - ci / 2, axis=0)
            high = np.percentile(x[6+i*k], 50 + ci / 2, axis=0)
            axs[i, 5].fill_between(t, low, high, color='r', alpha=0.2)
        median = np.median(x[6+i*k], axis=0) # Plot Median
        axs[i, 5].plot(t, median, color='b', label = "Median")
        axs[i, 5].set_xlabel("Time (days)")
        axs[i, 5].set_ylabel(r"$S_M$")
        axs[i, 5].set_title(rf"$S_M$ (Patch {i+1})")
        if np.any(np.array(x[6+i*k]) > Thresh):
            axs[i, 5].axhline(y=Thresh, color="k", linestyle="--")
        axs[i, 5].plot(xODE.t, xODE.y[6+i*k], color='g', label = 'ODE')
        axs[i, 5].legend()


        # Plot E_m + I_m
        for ci in ciVec:
            low = np.percentile(x[7+i*k] + x[8+i*k], 50 - ci / 2, axis=0)
            high = np.percentile(x[7+i*k] + x[8+i*k], 50 + ci / 2, axis=0)
            axs[i, 6].fill_between(t, low, high, color='r', alpha=0.2)
        median = np.median(x[7+i*k] + x[8+i*k], axis=0) # Plot Median
        axs[i, 6].plot(t, median, color='b', label = "Median")
        axs[i, 6].set_xlabel("Time (days)")
        axs[i, 6].set_ylabel(r"$E_M$ + $I_M$")
        axs[i, 6].set_title(rf"$E_M$ + $I_M$ (Patch {i+1})")
        axs[i, 6].plot(xODE.t, xODE.y[7+i*k] + xODE.y[8+i*k], color='g', label = 'ODE')
        axs[i, 6].legend()

        #Total human
        N_h = x[0+i*k] + x[1+i*k] + x[2+i*k] + x[3+i*k] + x[5+i*k] + x[9+i*k]
        N_hODE = xODE.y[0+i*k] + xODE.y[1+i*k] + xODE.y[2+i*k] + xODE.y[3+i*k] + xODE.y[5+i*k] + xODE.y[9+i*k]
        for ci in ciVec:
            low = np.percentile(N_h, 50 - ci / 2, axis=0)
            high = np.percentile(N_h, 50 + ci / 2, axis=0)
            axs[i, 7].fill_between(t, low, high, color='r', alpha=0.2)
        median = np.median(N_h, axis=0) # Plot Median
        axs[i, 7].plot(t, median, color='b', label = 'Median')
        axs[i, 7].set_xlabel("Time (days)")
        axs[i, 7].set_ylabel(r"$N_H$")
        axs[i, 7].set_title(rf"$N_H$ (Patch {i+1})")
        axs[i, 7].plot(xODE.t, N_hODE, color='g', label = 'ODE')
        axs[i, 7].legend()
        #Total mosquito
        N_m = x[6+i*k] + x[7+i*k] + x[8+i*k]
        N_mODE = xODE.y[6+i*k] + xODE.y[7+i*k] + xODE.y[8+i*k]
        for ci in ciVec:
            low = np.percentile(N_m, 50 - ci / 2, axis=0)
            high = np.percentile(N_m, 50 + ci / 2, axis=0)
            axs[i, 8].fill_between(t, low, high, color='r', alpha=0.2)
        median = np.median(N_m, axis=0) # Plot Median
        axs[i, 8].plot(t, median, color='b')
        axs[i, 8].set_xlabel("Time (days)")
        axs[i, 8].set_ylabel(r"$N_M$")
        axs[i, 8].set_title(rf"$N_M$ (Patch {i+1})")
        axs[i, 8].plot(xODE.t, N_mODE, color='g', label = 'ODE')
        axs[i, 8].legend()
    plt.margins(x=0)
    plt.savefig("VacModelCIAllPatches1patch50", dpi=500, bbox_inches="tight")
    plt.show()

def PlotDiseaseElim(sim, n, k, N):
    # Define malaria elimination time as the last time population drops to 0, given it doesn't rise to above 5 infected individuals at onces
    fig, axs = plt.subplots(n, 2, squeeze=False, constrained_layout=True, figsize=(10, 15))

    tElimHu = [[] for i in range(n)] 
    countElimHu = [0 for i in range(n)] 
    # Human Elimination in patch i
    for i in range(n):   
        countElim = 0      
        for j in range(N):
            totalI = sim[j][0][1+i*k] + sim[j][0][2+i*k] + sim[j][0][3+i*k]
            idx = np.where((totalI == 0))[0]
            idx = idx[idx>1000] #must be at least after t=
            if idx.size:
                t = sim[j][1][idx[0]] 
                tElimHu[i].append(t)
                countElim += 1
                         
        countElimHu[i] = countElim

        axs[i, 0].hist(tElimHu[i], bins=30)
        axs[i, 0].set_xlabel("Time (days)")
        axs[i, 0].set_ylabel("Frequency")
        axs[i, 0].set_title(rf"Human Malaria Elimination Time (Patch {i+1})")
        
    # Mosquito Elimination in patch i
    tElimMo = [[] for i in range(n)] 
    countElimMo = [0 for i in range(n)] 
    for i in range(n):   
        countElim = 0      
        for j in range(N):
            totalI = sim[j][0][1+i*k] + sim[j][0][2+i*k] + sim[j][0][3+i*k]
            idx = np.where((totalI == 0))[0]
            idx = idx[idx>1000] #must be at least after t=
            if idx.size:
                t = sim[j][1][idx[0]] 
                tElimMo[i].append(t)
                countElim += 1
                         
        countElimMo[i] = countElim

        axs[i, 1].hist(tElimMo[i], bins=30)
        axs[i, 1].set_xlabel("Time (days)")
        axs[i, 1].set_ylabel("Frequency")
        axs[i, 1].set_title(rf"Mosquito Malaria Elimination Time (Patch {i+1})")
    plt.savefig("VacModelPartialElim1patch50", dpi=500, bbox_inches="tight")
    plt.show() 

    # Complete malaria elimination
    countElim = 0 
    tElimT = [] 
    for j in range(N):
        totalI = 0    
        for i in range(n):      
            totalI += sim[j][0][1+i*k] + sim[j][0][2+i*k] + sim[j][0][3+i*k] + sim[j][0][7+i*k] + sim[j][0][8+i*k]
        idx = np.where(totalI == 0)[0]
        if idx.size:
            if sim[j][1][idx[0]] > 1000:
                tElimT.append(sim[j][1][idx[0]])
                countElim += 1
                    
    plt.hist(tElimT, bins=30)
    plt.xlabel("Time (days)")
    plt.ylabel("Frequency")
    plt.title(rf"Compete Malaria Elimination Time (Patch {i+1})")    
    plt.savefig("VacModelCompleteElim1patch50", dpi=500, bbox_inches="tight")
    plt.show()

def averageNewPlots(t, x, R0Vec, RtVec, ciVec, n, k, comp, my_opts, xODE):
    Thresh = my_opts["SwitchingThreshold"][1]

    # Plot R0
    plt.figure()
    for ci in ciVec:
        low = np.percentile(R0Vec, 50 - ci / 2, axis=0)
        high = np.percentile(R0Vec, 50 + ci / 2, axis=0)
        plt.fill_between(t, low, high, color='r', alpha=0.2)
    median = np.median(R0Vec, axis=0) # Plot Median
    plt.plot(t, median, color='b', label = "Median")
    plt.xlabel("Time (days)", fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.ylabel(r"$R_0$", fontsize=14)
    plt.axhline(y=1, linestyle='--', color="k")
    plt.margins(x=0)
    plt.savefig("VacModelCIR01patch", dpi=150, bbox_inches="tight")
    plt.show()

    # Plot Rt
    plt.figure()
    for ci in ciVec:
        low = np.percentile(RtVec, 50 - ci / 2, axis=0)
        high = np.percentile(RtVec, 50 + ci / 2, axis=0)
        plt.fill_between(t, low, high, color='r', alpha=0.2)
    median = np.median(RtVec, axis=0) # Plot Median
    plt.plot(t, median, color='b', label = "Median")
    plt.xlabel("Time (days)", fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.ylabel(r"$R_t$", fontsize=14)
    plt.axhline(y=1, linestyle='--', color="k")
    plt.margins(x=0)
    plt.axvline(x=1000, linestyle='--', color="k", linewidth=1)  
    plt.savefig("VacModelCIRt1patch", dpi=150, bbox_inches="tight")
    plt.show()

    # Plot states
    fig, axs = plt.subplots(n, 4, squeeze=False, constrained_layout=True, figsize=(12, 2*n))
    for i in range(n): #no states
        # Annotation
        axs[i, 0].annotate( f"Patch {i+1}",xy=(-0.40, 0.5), xycoords="axes fraction",rotation=90,ha="center",va="center",fontsize=14)

        # Plot individual states
        # Plot S_h
        for ci in ciVec:
            low = np.percentile(x[0+i*k], 50 - ci / 2, axis=0)
            high = np.percentile(x[0+i*k], 50 + ci / 2, axis=0)
            axs[i, 0].fill_between(t, low, high, color='r', alpha=0.2)
        median = np.median(x[0+i*k], axis=0) # Plot Median
        axs[i, 0].plot(t, median, color='b', label = "Median")
        if np.any(np.array(x[0+i*k]) > Thresh):
            axs[i, 0].axhline(y=Thresh, color="k", linestyle="--")
        axs[i, 0].plot(xODE.t, xODE.y[0+i*k], color='g', label = 'ODE')
        axs[i, 0].tick_params(axis='both', labelsize=14)
        axs[i, 0].axvline(x=1000, linestyle='--', color="k")  
        if i!=n-1:
            axs[i, 0].set_xticks([])

        # Plot V_h
        for ci in ciVec:
            low = np.percentile(x[9+i*k], 50 - ci / 2, axis=0)
            high = np.percentile(x[9+i*k], 50 + ci / 2, axis=0)
            axs[i, 1].fill_between(t, low, high, color='r', alpha=0.2)
        median = np.median(x[9+i*k], axis=0) # Plot Median
        axs[i, 1].plot(t, median, color='b', label = "Median")
        if np.any(np.array(x[9+i*k]) > Thresh):
            axs[i, 1].axhline(y=Thresh, color="k", linestyle="--")
        axs[i, 1].plot(xODE.t, xODE.y[9+i*k], color='g', label = 'ODE')
        axs[i, 1].tick_params(axis='both', labelsize=14)
        if i!=n-1:
            axs[i, 1].set_xticks([])
        axs[i, 1].axvline(x=1000, linestyle='--', color="k")                

        # Plot E_h + I_h + A_h
        for ci in ciVec:
            low = np.percentile(x[1+i*k] + x[2+i*k]+ x[3+i*k], 50 - ci / 2, axis=0)
            high = np.percentile(x[1+i*k] + x[2+i*k]+ x[3+i*k], 50 + ci / 2, axis=0)
            axs[i, 2].fill_between(t, low, high, color='r', alpha=0.2)
        median = np.median(x[1+i*k] + x[2+i*k]+ x[3+i*k], axis=0) # Plot Median
        axs[i, 2].plot(t, median, color='b', label = "Median")
        axs[i, 2].plot(xODE.t, xODE.y[1+i*k] + xODE.y[2+i*k]+ xODE.y[3+i*k], color='g', label = 'ODE')
        axs[i, 2].tick_params(axis='both', labelsize=14)
        if i!=n-1:
            axs[i, 2].set_xticks([])
        axs[i, 2].axvline(x=1000, linestyle='--', color="k")

        # Plot N_M
        N_M = x[6+i*k] + x[7+i*k] + x[8+i*k]
        for ci in ciVec:
            low = np.percentile(N_M, 50 - ci / 2, axis=0)
            high = np.percentile(N_M, 50 + ci / 2, axis=0)
            axs[i, 3].fill_between(t, low, high, color='r', alpha=0.2)
        median = np.median(N_M, axis=0) # Plot Median
        axs[i, 3].plot(t, median, color='b', label = "Median")
        if np.any(np.array(N_M) > Thresh):
            axs[i, 3].axhline(y=Thresh, color="k", linestyle="--")
        axs[i, 3].plot(xODE.t, xODE.y[6+i*k] + xODE.y[7+i*k] + xODE.y[8+i*k], color='g', label = 'ODE')
        axs[i, 3].tick_params(axis='both', labelsize=14)
        if i!=n-1:
            axs[i, 3].set_xticks([])        
        axs[i, 3].axvline(x=1000, linestyle='--', color="k")

    axs[0, 0].set_title(rf"$S_H$", fontsize=14)
    axs[0, 1].set_title(rf"$V_H$", fontsize=14)
    axs[0, 2].set_title(rf"$E_H$+$I_H$+$A_H$", fontsize=14)
    axs[0, 3].set_title(rf"$N_M$", fontsize=14)
    axs[n-1, 0].set_xlabel("Time (days)", fontsize=14)
    axs[n-1, 1].set_xlabel("Time (days)", fontsize=14)
    axs[n-1, 2].set_xlabel("Time (days)", fontsize=14)
    axs[n-1, 3].set_xlabel("Time (days)", fontsize=14)
    plt.margins(x=0)
    plt.savefig("VacModelCI1patch", dpi=150, bbox_inches="tight")
    plt.show()
    
def plotStates(n, k, sim, my_opts, hPatch, mPatch, dPatch, mParam, IC, t_max, rngMatrix, tStep):

    ODE_sol = m.solveODE(hPatch, mPatch, dPatch, mParam, n, k, IC, t_max, rngMatrix, tStep)
    Thresh = my_opts["SwitchingThreshold"][1]
    fig, axs = plt.subplots(n, 3, squeeze=False, constrained_layout=True, figsize=(10, 2.5*n))
    
    for i in range(n):
        # Annotation
        axs[i, 0].annotate( f"Patch {i+1}",xy=(-0.40, 0.5), xycoords="axes fraction",rotation=90,ha="center",va="center",fontsize=14)

        #S and V
        axs[i, 0].plot(sim[1], sim[0][0+i*k], label = r"$S_{JSF}$", color="tab:blue")
        axs[i, 0].plot(ODE_sol.t, ODE_sol.y[i*k], label = r"$S_{ODE}$", linestyle=(0, (1, 1)), color="tab:blue", linewidth=2)
        axs[i, 0].plot(sim[1], sim[0][9+i*k], label = r"$R_{JSF}$", color="tab:green")              
        axs[i, 0].plot(ODE_sol.t, ODE_sol.y[9+i*k], label = r"$R_{ODE}$", linestyle=(0, (1, 1)), color="tab:green", linewidth=2)
        axs[i, 0].axvline(x=1000, linestyle='--', color='black')
        axs[i, 0].tick_params(axis='both', labelsize=14)
        if i!=n-1:
            axs[i, 0].set_xticks([])    
        if np.any(np.array(sim[0][0+i*k]) > Thresh) or np.any(np.array(sim[0][5+i*k]) > Thresh):
            axs[i, 0].axhline(y=Thresh, color="k", linestyle="--")

        # E+I+A  
        axs[i, 1].plot(sim[1], np.array(sim[0][1+i*k])+np.array(sim[0][2+i*k])+np.array(sim[0][3+i*k]), label = r"$I_{JSF}$", color="tab:red")
        axs[i, 1].plot(ODE_sol.t, ODE_sol.y[1+i*k]+ODE_sol.y[2+i*k]+ODE_sol.y[3+i*k], label = r"$A_{ODE}$", linestyle=(0, (1, 1)), color="tab:red", linewidth=2)
        axs[i, 1].tick_params(axis='both', labelsize=14)
        if i!=n-1:
            axs[i, 1].set_xticks([])  
        axs[i, 1].axvline(x=1000, linestyle='--', color="k")    

        #Total mosquitoes
        axs[i, 2].plot(sim[1], np.array(sim[0][6+i*k]) + np.array(sim[0][7+i*k]) + np.array(sim[0][8+i*k]), label = r"$D_{JSF}$", color="orange")
        axs[i, 2].plot(ODE_sol.t, ODE_sol.y[6+i*k] + ODE_sol.y[7+i*k] + ODE_sol.y[8+i*k], label = r"$D_{ODE}$", linestyle=(0, (1, 1)), color="orange", linewidth=2)
        axs[i, 2].tick_params(axis='both', labelsize=14)
        if i!=n-1:
            axs[i, 2].set_xticks([])    
        if np.any(np.array(sim[0][4+i*k]) > Thresh):
            axs[i, 2].axhline(y=Thresh, color="k", linestyle="--")
        axs[i, 2].axvline(x=1000, linestyle='--', color="k")

    axs[0, 0].set_title(r"$S_{H}$" + " and " + r"$V_{H}$", fontsize=14)
    axs[0, 1].set_title(r"$E_{H}+I_{H}+A_{H}$", fontsize=14)
    axs[0, 2].set_title(r"$N_{M}$", fontsize=14)
    axs[n-1, 0].set_xlabel("Time (days)", fontsize=14)
    axs[n-1, 1].set_xlabel("Time (days)", fontsize=14)
    axs[n-1, 2].set_xlabel("Time (days)", fontsize=14)         
    plt.savefig("VacModelHstates1patch", dpi=150, bbox_inches="tight")    
    plt.show()
    




