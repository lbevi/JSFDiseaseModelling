import sympy as sp
import numpy as np
import model as m
from model import V_ON

# Print settings
np.set_printoptions(threshold=np.inf)   

# Compartments and Patch Number
allNames = ["S_h", "E_h", "I_h", "A_h", "R_h", "S_m", "E_m", "I_m"]
allNamesInt =  ["S_h", "V_h", "E_h", "I_h", "A_h", "R_h", "S_m", "E_m", "I_m"]
iNames = ["E_h", "I_h", "A_h", "E_m", "I_m"]
sNamesInt = ["S_h", "R_h", "S_m", "V_h"]
sNames = ["S_h", "S_m", "R_h"]

#NO INTERVENTION MODEL
#Human Parameters
# S, E, I, A, D, R, (IC)
# u_h, b_h 
hP1 = m.createHumanPatchNoInt(25000, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4)
hP2 = m.createHumanPatchNoInt(1000, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4)
hP3 = m.createHumanPatchNoInt(50, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4)
hP4 = m.createHumanPatchNoInt(5000, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4)
hP5 = m.createHumanPatchNoInt(15, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4)

hPatch = [hP1, hP2, hP3, hP4, hP5]

n = len(hPatch); #patch size

#Mosquito Parameters
# S, E, I (IC)
# b_sy, b_asy, b_su, u_m

# BITE RATES INC BY 0.2

mP1 = m.createMosquitoPatchNoInt(50000, 0, 50,
                               0.4, 0.6, 0.6, 0.0477)
mP2 = m.createMosquitoPatchNoInt(2000, 0, 0,
                               0.4, 0.6, 0.6, 0.0477)
mP3 = m.createMosquitoPatchNoInt(100, 0, 0,
                               0.4, 0.6, 0.6, 0.0477)
mP4 = m.createMosquitoPatchNoInt(10000, 0, 0,
                               0.4, 0.6, 0.6, 0.0477)
mP5 = m.createMosquitoPatchNoInt(30, 0, 0,
                               0.4, 0.6, 0.6, 0.0477)

mPatch = [mP1, mP2, mP3, mP4, mP5]

#Disease Parameters
# d_h, gI_h, gA_h w_h, o_h, o_m, t_hm, t_mh, p_sy, chi
dP1 = m.createDiseasePatchNoInt(1.8e-3, 0.08, 0.01, 0.01, 1/15, 1/9, 0.022, 0.24, 0.288, 0.75)
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
for c in allNames:
    if c.endswith("_h"):
        mParam[c] = tH.copy()
    elif c.endswith("_m"):
        mParam[c] = tM.copy()

mParam["D_h"] = np.zeros((n,n))

#Compute DFE, EE and Stability
[StateVar, dStateVar] = m.interventionDisabled(hPatch, mPatch, dPatch, mParam, n, allNames)

dfe, sol = m.findDFE(StateVar, dStateVar, iNames, sNames, allNames, n)
#m.printPoint(dfe)
stability = m.checkStability(StateVar, dStateVar, dfe)
print("No-intervention DFE:", stability)

guess = dict(dfe)
idx = m.createStateMatrix(n, allNames)
for name in iNames:
    for j in idx[name]:
        guess[StateVar[j]] = 10000.0

ee, converged, msg = m.findEE(StateVar, dStateVar, guess)
#m.printPoint(ee)
if not converged:
    print("fsolve did not converge:", msg)
stability = m.checkStability(StateVar, dStateVar, ee)
print("No-intervention EE:", stability)    
print()


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
for c in allNamesInt:
    if c.endswith("_h"):
        mParam[c] = tH.copy()
    elif c.endswith("_m"):
        mParam[c] = tM.copy()

mParam["D_h"] = np.zeros((n,n))

#1-patch model
[IntStateVar, IntdStateVar] = m.interventionEnabled(hPatch, mPatch, dPatch, mParam, n, allNamesInt, 1, 1)

dfe, sol = m.findDFE(IntStateVar, IntdStateVar, iNames, sNamesInt, allNamesInt, n)
#m.printPoint(dfe)
stability = m.checkStability(IntStateVar, IntdStateVar, dfe)
print("1-patch DFE", stability)

guess = dict(dfe)
idx = m.createStateMatrix(n, allNamesInt)
for name in iNames:
    for j in idx[name]:
        guess[IntStateVar[j]] = 10000.0

ee, converged, msg = m.findEE(IntStateVar, IntdStateVar, guess)
#m.printPoint(ee)
if not converged:
    print("fsolve did not converge:", msg)
stability = m.checkStability(IntStateVar, IntdStateVar, ee)
print("1-patch EE",stability)
m.sameEquilibrium(dfe, ee, IntStateVar, 1e-6)
print()

#1-patch model with 20% efficacy
hP1 = m.createHumanPatch(25000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0.02, 1/365, 0.8)
hP2 = m.createHumanPatch(1000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 1/365, 0.8)
hP3 = m.createHumanPatch(50, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 1/365, 0.8)
hP4 = m.createHumanPatch(5000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 1/365, 0.8)
hP5 = m.createHumanPatch(15, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0, 1/365, 0.8)

hPatch = [hP1, hP2, hP3, hP4, hP5]

[IntStateVar, IntdStateVar] = m.interventionEnabled(hPatch, mPatch, dPatch, mParam, n, allNamesInt, 1, 1)

dfe, sol = m.findDFE(IntStateVar, IntdStateVar, iNames, sNamesInt, allNamesInt, n)
#m.printPoint(dfe)
stability = m.checkStability(IntStateVar, IntdStateVar, dfe)
print("1-patch DFE 20 efficacy", stability)

guess = dict(dfe)
idx = m.createStateMatrix(n, allNamesInt)
for name in iNames:
    for j in idx[name]:
        guess[IntStateVar[j]] = 10000.0

ee, converged, msg = m.findEE(IntStateVar, IntdStateVar, guess)
#m.printPoint(ee)
if not converged:
    print("fsolve did not converge:", msg)
stability = m.checkStability(IntStateVar, IntdStateVar, ee)
print("1-patch EE 20 efficacy",stability)
m.sameEquilibrium(dfe, ee, IntStateVar, 1e-6)
print()

#all-patch model
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

[IntStateVar, IntdStateVar] = m.interventionEnabled(hPatch, mPatch, dPatch, mParam, n, allNamesInt, 1, 1)

dfe, sol = m.findDFE(IntStateVar, IntdStateVar, iNames, sNamesInt, allNamesInt, n)
#m.printPoint(dfe)
stability = m.checkStability(IntStateVar, IntdStateVar, dfe)
print("all-patch DFE", stability)

guess = dict(dfe)
idx = m.createStateMatrix(n, allNamesInt)
for name in iNames:
    for j in idx[name]:
        guess[IntStateVar[j]] = 10000.0

ee, converged, msg = m.findEE(IntStateVar, IntdStateVar, guess)
#m.printPoint(ee)
if not converged:
    print("fsolve did not converge:", msg)
stability = m.checkStability(IntStateVar, IntdStateVar, ee)
print("all-patch EE",stability)
m.sameEquilibrium(dfe, ee, IntStateVar, 1e-6)
print()

#extreme model
hP1 = m.createHumanPatch(25000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0.06, 1/365, 0.1)
hP2 = m.createHumanPatch(1000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0.06, 1/365, 0.1)
hP3 = m.createHumanPatch(50, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0.06, 1/365, 0.1)
hP4 = m.createHumanPatch(5000, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0.06, 1/365, 0.1)
hP5 = m.createHumanPatch(15, 0, 0, 0, 0, 0, 0,
                             4.5e-5, 1.2e-4, 0.06, 1/365, 0.1)
hPatch = [hP1, hP2, hP3, hP4, hP5]

[IntStateVar, IntdStateVar] = m.interventionEnabled(hPatch, mPatch, dPatch, mParam, n, allNamesInt, 1, 1)

dfe, sol = m.findDFE(IntStateVar, IntdStateVar, iNames, sNamesInt, allNamesInt, n)
#m.printPoint(dfe)
stability = m.checkStability(IntStateVar, IntdStateVar, dfe)
print("extreme DFE", stability)

guess = dict(dfe)
idx = m.createStateMatrix(n, allNamesInt)
for name in iNames:
    for j in idx[name]:
        guess[IntStateVar[j]] = 10000.0

ee, converged, msg = m.findEE(IntStateVar, IntdStateVar, guess)
#m.printPoint(ee)
if not converged:
    print("fsolve did not converge:", msg)
stability = m.checkStability(IntStateVar, IntdStateVar, ee)
print("extreme EE",stability)
m.sameEquilibrium(dfe, ee, IntStateVar, 1e-6)
print()

#5% model
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

[IntStateVar, IntdStateVar] = m.interventionEnabled(hPatch, mPatch, dPatch, mParam, n, allNamesInt, 1.05, 0.95)

dfe, sol = m.findDFE(IntStateVar, IntdStateVar, iNames, sNamesInt, allNamesInt, n)
#m.printPoint(dfe)
stability = m.checkStability(IntStateVar, IntdStateVar, dfe)
print("5-model DFE", stability)

guess = dict(dfe)
idx = m.createStateMatrix(n, allNamesInt)
for name in iNames:
    for j in idx[name]:
        guess[IntStateVar[j]] = 10000.0

ee, converged, msg = m.findEE(IntStateVar, IntdStateVar, guess)
#m.printPoint(ee)
if not converged:
    print("fsolve did not converge:", msg)
stability = m.checkStability(IntStateVar, IntdStateVar, ee)
print("5-model EE", stability)
m.sameEquilibrium(dfe, ee, IntStateVar, 1e-6)
print()

#25% model
[IntStateVar, IntdStateVar] = m.interventionEnabled(hPatch, mPatch, dPatch, mParam, n, allNamesInt, 1.25, 0.75)

dfe, sol = m.findDFE(IntStateVar, IntdStateVar, iNames, sNamesInt, allNamesInt, n)
#m.printPoint(dfe)
stability = m.checkStability(IntStateVar, IntdStateVar, dfe)
print("25-model DFE", stability)

guess = dict(dfe)
idx = m.createStateMatrix(n, allNamesInt)
for name in iNames:
    for j in idx[name]:
        guess[IntStateVar[j]] = 10000.0

ee, converged, msg = m.findEE(IntStateVar, IntdStateVar, guess)
#m.printPoint(ee)
if not converged:
    print("fsolve did not converge:", msg)
stability = m.checkStability(IntStateVar, IntdStateVar, ee)
print("25-model EE", stability)
m.sameEquilibrium(dfe, ee, IntStateVar, 1e-6)
print()

#50% model
[IntStateVar, IntdStateVar] = m.interventionEnabled(hPatch, mPatch, dPatch, mParam, n, allNamesInt, 1.50, 0.50)

dfe, sol = m.findDFE(IntStateVar, IntdStateVar, iNames, sNamesInt, allNamesInt, n)
#m.printPoint(dfe)
stability = m.checkStability(IntStateVar, IntdStateVar, dfe)
print("50-model DFE", stability)

guess = dict(dfe)
idx = m.createStateMatrix(n, allNamesInt)
for name in iNames:
    for j in idx[name]:
        guess[IntStateVar[j]] = 10000.0

ee, converged, msg = m.findEE(IntStateVar, IntdStateVar, guess)
#m.printPoint(ee)
if not converged:
    print("fsolve did not converge:", msg)
stability = m.checkStability(IntStateVar, IntdStateVar, ee)
print("50-model EE", stability)
m.sameEquilibrium(dfe, ee, IntStateVar, 1e-6)
print()