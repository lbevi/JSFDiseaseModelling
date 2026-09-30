import numpy as np
import model as m
import sympy as sp
from scipy.optimize import root

# model functions for n-Patch model
def createHumanPatch(S, E, I, A, R, D, V, u_h, L_h, v, phi, beta):

    return { 
            "u_h": u_h,    #death rate per day
            "L_h": L_h,    #birth rate per day
            "v": v,        #vaccination rate per day
            "phi": phi,    #waning vaccination rate rate per day
            "beta": beta,  #reduction in infectiousness from vaccine
            "ic": {
                "S": S,    #initial susceptibles
                "E": E,    #initial exposed
                "I": I,    #initial sympto
                "A": A,    #initial asympto
                "R": R,    #initial recovered
                "D": D,    #initial dead due to malaria
                "V": V,    #initial vaccinated
            }
    }

def createMosquitoPatch(S, E, I, b_sy, b_asy, b_su, u_m):

    return { 
            "b_sy": b_sy,     #bite rate per day on symptomatic human
            "b_asy": b_asy,   #bite rate per day on asymptomatic human 
            "b_su": b_su,     #bite rate per day on susceptible human 
            "u_m": u_m,       #death rate per day (lifespan 9d)
            "ic": {
                "S": S,       #initial susceptibles
                "E": E,       #initial exposed
                "I": I        #initial sympto
        
            }    
    }

def createDiseasePatch(d_h, gI_h, gA_h, w_h, o_h, o_m, t_hm, t_mh, p_sy, chi):

    return { 
            "d_h": d_h,          #human death rate due to malaria per day
            "gI_h": gI_h,        #recovery rate per day (14 days infectious)
            "gA_h": gA_h,        #recovery rate per day (14 days asymptomatic)
            "w_h": w_h,          #waning immunity rate per day (1 month)
            "o_h": o_h,          #human latent period per day (12 day latent period)
            "o_m": o_m,          #mosquito latent period per day (10 day latent period)
            "t_hm": t_hm,        #probability of transmission from human to mosquito
            "t_mh": t_mh,        #probability of transmission from mosquito to human
            "p_sy": p_sy,        #probability of symptomatic case 
            "chi": chi,          #Relative infectiousness between asymptomatic and symptomatic      
    }

def createHumanPatchNoInt(S, E, I, A, R, D, u_h, L_h):

    return { 
            "u_h": u_h,    #death rate per day
            "L_h": L_h,    #birth rate per day
            "ic": {
                "S": S,    #initial susceptibles
                "E": E,    #initial exposed
                "I": I,    #initial sympto
                "A": A,    #initial asympto
                "R": R,    #initial recovered
                "D": D     #initial dead due to malaria
            }
    }

def createMosquitoPatchNoInt(S, E, I, b_sy, b_asy, b_su, u_m):

    return { 
            "b_sy": b_sy,     #bite rate per day on symptomatic human
            "b_asy": b_asy,   #bite rate per day on asymptomatic human 
            "b_su": b_su,     #bite rate per day on susceptible human 
            "u_m": u_m,       #death rate per day (lifespan 9d)
            "ic": {
                "S": S,       #initial susceptibles
                "E": E,       #initial exposed
                "I": I        #initial sympto
        
            }    
    }

def createDiseasePatchNoInt(d_h, gI_h, gA_h, w_h, o_h, o_m, t_hm, t_mh, p_sy, chi):

    return { 
            "d_h": d_h,          #human death rate due to malaria per day
            "gI_h": gI_h,        #recovery rate per day (14 days infectious)
            "gA_h": gA_h,        #recovery rate per day (14 days asymptomatic)
            "w_h": w_h,          #waning immunity rate per day (1 month)
            "o_h": o_h,          #human latent period per day (12 day latent period)
            "o_m": o_m,          #mosquito latent period per day (10 day latent period)
            "t_hm": t_hm,        #probability of transmission from human to mosquito
            "t_mh": t_mh,        #probability of transmission from mosquito to human
            "p_sy": p_sy,        #probability of symptomatic case 
            "chi": chi,          #Relative infectiousness between asymptomatic and symptomatic      
    }

# Vaccination switch on
V_ON = 1000
def V(t, v):
    if t >= V_ON:
        return v
    else: 
        return 0  

# Nets switch on (births)
def bRed(t, i, red):
    if t >= V_ON and i == 0:
        return red
    else:
        return 1

# Nets switch on (deaths)
def uRed(t, i, inc):
    if t >= V_ON and i == 0:
        return inc
    else:
        return 1
    
def interventionEnabled(hPatch, mPatch, dPatch, mParam, n, names, inc, red):
    #Define states          
    states = {name: list(sp.symbols(f"{name}0:{n}")) for name in names}
    dstates = {name: list(sp.symbols(f"{name}0:{n}")) for name in names}

    for i in range(n):     

        N_h =states["S_h"][i] + states["E_h"][i] + states["A_h"][i] + states["I_h"][i] + states["R_h"][i] + states["V_h"][i]

        λ_h = mPatch[i]["b_su"] * dPatch[i]["t_mh"] * states["I_m"][i] / N_h
        λ_m = (mPatch[i]["b_sy"] * states["I_h"][i] + mPatch[i]["b_asy"] * dPatch[i]["chi"] * states["A_h"][i]) * dPatch[i]["t_hm"] / N_h

        NH_IC = 0
        for c in hPatch[i]["ic"].keys():
            NH_IC += hPatch[i]["ic"][c]

        NM_IC = 0
        for c in mPatch[i]["ic"].keys():
            NM_IC += mPatch[i]["ic"][c]        

        #Human equations
        #S_h
        dstates["S_h"][i] = hPatch[i]["L_h"]*NH_IC+ dPatch[i]["w_h"]*states["R_h"][i] + hPatch[i]["phi"]*states["V_h"][i]  - λ_h*states["S_h"][i] - (hPatch[i]["u_h"] + V(V_ON, hPatch[i]["v"]) + sum(mParam["S_h"][i,j] for j in range(n)))*states["S_h"][i] + sum(mParam["S_h"][j,i]*states["S_h"][j] for j in range(n))
        #E_h
        dstates["E_h"][i] = λ_h*states["S_h"][i] + hPatch[i]["beta"]*λ_h*states["V_h"][i] - (dPatch[i]["o_h"] + hPatch[i]["u_h"] + sum(mParam["E_h"][i,j] for j in range(n)))*states["E_h"][i] + sum(mParam["E_h"][j,i]*states["E_h"][j] for j in range(n))
        #I_h
        dstates["I_h"][i] = dPatch[i]["o_h"]*dPatch[i]["p_sy"]*states["E_h"][i] - (dPatch[i]["gI_h"] + hPatch[i]["u_h"] + dPatch[i]["d_h"] + sum(mParam["I_h"][i,j] for j in range(n)))*states["I_h"][i] + sum(mParam["I_h"][j,i]*states["I_h"][j] for j in range(n))
        #A_h
        dstates["A_h"][i] = dPatch[i]["o_h"]*(1-dPatch[i]["p_sy"])*states["E_h"][i] - (dPatch[i]["gA_h"] + hPatch[i]["u_h"] + sum(mParam["A_h"][i,j] for j in range(n)))*states["A_h"][i] + sum(mParam["A_h"][j,i]*states["A_h"][j] for j in range(n))
        #R_h
        dstates["R_h"][i] = dPatch[i]["gI_h"]*states["I_h"][i] + dPatch[i]["gA_h"]*states["A_h"][i] - (dPatch[i]["w_h"] + hPatch[i]["u_h"] + sum(mParam["R_h"][i,j] for j in range(n)))*states["R_h"][i] + sum(mParam["R_h"][j,i]*states["R_h"][j] for j in range(n))
        #V_h
        dstates["V_h"][i] = V(V_ON, hPatch[i]["v"])*states["S_h"][i] - hPatch[i]["beta"]*λ_h*states["V_h"][i] - (hPatch[i]["u_h"] + hPatch[i]["phi"] + sum(mParam["V_h"][i,j] for j in range(n)))*states["V_h"][i] + sum(mParam["V_h"][j,i]*states["V_h"][j] for j in range(n))

        #Mosquito equations
        #S_m 
        dstates["S_m"][i] =  bRed(V_ON, i, red)*0.05*NM_IC - λ_m*states["S_m"][i] - (uRed(V_ON, i, inc)*mPatch[i]["u_m"] + sum(mParam["S_m"][i,j] for j in range(n)))*states["S_m"][i] + sum(mParam["S_m"][j,i]*states["S_m"][j] for j in range(n))
        #E_m
        dstates["E_m"][i] = λ_m*states["S_m"][i] - (dPatch[i]["o_m"] + uRed(V_ON, i, inc)*mPatch[i]["u_m"] + sum(mParam["E_m"][i,j] for j in range(n)))*states["E_m"][i] + sum(mParam["E_m"][j,i]*states["E_m"][j] for j in range(n))
        #I_m
        dstates["I_m"][i] = dPatch[i]["o_m"]*states["E_m"][i] - (uRed(V_ON, i, inc)*mPatch[i]["u_m"] + sum(mParam["I_m"][i,j] for j in range(n)))*states["I_m"][i] + sum(mParam["I_m"][j,i]*states["I_m"][j] for j in range(n))

    stateVar = [states[c][i] for c in names for i in range(n)]
    dstateVar = [dstates[c][i] for c in names for i in range(n)]

    return stateVar, dstateVar

def interventionDisabled(hPatch, mPatch, dPatch, mParam, n, names):
    #Define states          
    states = {name: list(sp.symbols(f"{name}0:{n}")) for name in names}
    dstates = {name: list(sp.symbols(f"{name}0:{n}")) for name in names}

    for i in range(n):     

        N_h =states["S_h"][i] + states["E_h"][i] + states["A_h"][i] + states["I_h"][i] + states["R_h"][i]

        λ_h = mPatch[i]["b_su"] * dPatch[i]["t_mh"] * states["I_m"][i] / N_h
        λ_m = (mPatch[i]["b_sy"] * states["I_h"][i] + mPatch[i]["b_asy"] * dPatch[i]["chi"] * states["A_h"][i]) * dPatch[i]["t_hm"] / N_h

        NH_IC = 0
        for c in hPatch[i]["ic"].keys():
            NH_IC += hPatch[i]["ic"][c]

        NM_IC = 0
        for c in mPatch[i]["ic"].keys():
            NM_IC += mPatch[i]["ic"][c]        

        #Human equations
        #S_h
        dstates["S_h"][i] = hPatch[i]["L_h"]*NH_IC+ dPatch[i]["w_h"]*states["R_h"][i] - λ_h*states["S_h"][i] - (hPatch[i]["u_h"] + sum(mParam["S_h"][i,j] for j in range(n)))*states["S_h"][i] + sum(mParam["S_h"][j,i]*states["S_h"][j] for j in range(n))
        #E_h
        dstates["E_h"][i] = λ_h*states["S_h"][i] - (dPatch[i]["o_h"] + hPatch[i]["u_h"] + sum(mParam["E_h"][i,j] for j in range(n)))*states["E_h"][i] + sum(mParam["E_h"][j,i]*states["E_h"][j] for j in range(n))
        #I_h
        dstates["I_h"][i] = dPatch[i]["o_h"]*dPatch[i]["p_sy"]*states["E_h"][i] - (dPatch[i]["gI_h"] + hPatch[i]["u_h"] + dPatch[i]["d_h"] + sum(mParam["I_h"][i,j] for j in range(n)))*states["I_h"][i] + sum(mParam["I_h"][j,i]*states["I_h"][j] for j in range(n))
        #A_h
        dstates["A_h"][i] = dPatch[i]["o_h"]*(1-dPatch[i]["p_sy"])*states["E_h"][i] - (dPatch[i]["gA_h"] + hPatch[i]["u_h"] + sum(mParam["A_h"][i,j] for j in range(n)))*states["A_h"][i] + sum(mParam["A_h"][j,i]*states["A_h"][j] for j in range(n))
        #R_h
        dstates["R_h"][i] = dPatch[i]["gI_h"]*states["I_h"][i] + dPatch[i]["gA_h"]*states["A_h"][i] - (dPatch[i]["w_h"] + hPatch[i]["u_h"] + sum(mParam["R_h"][i,j] for j in range(n)))*states["R_h"][i] + sum(mParam["R_h"][j,i]*states["R_h"][j] for j in range(n))

        #Mosquito equations
        #S_m 
        dstates["S_m"][i] =  0.05*NM_IC - λ_m*states["S_m"][i] - (mPatch[i]["u_m"] + sum(mParam["S_m"][i,j] for j in range(n)))*states["S_m"][i] + sum(mParam["S_m"][j,i]*states["S_m"][j] for j in range(n))
        #E_m
        dstates["E_m"][i] = λ_m*states["S_m"][i] - (dPatch[i]["o_m"] + mPatch[i]["u_m"] + sum(mParam["E_m"][i,j] for j in range(n)))*states["E_m"][i] + sum(mParam["E_m"][j,i]*states["E_m"][j] for j in range(n))
        #I_m
        dstates["I_m"][i] = dPatch[i]["o_m"]*states["E_m"][i] - (mPatch[i]["u_m"] + sum(mParam["I_m"][i,j] for j in range(n)))*states["I_m"][i] + sum(mParam["I_m"][j,i]*states["I_m"][j] for j in range(n))

    stateVar = [states[c][i] for c in names for i in range(n)]
    dstateVar = [dstates[c][i] for c in names for i in range(n)]

    return stateVar, dstateVar

def createStateMatrix(n, names):
    return {name: list(range(k*n, (k+1)*n)) for k, name in enumerate(names)}

def findDFE(stateVars, rhsList, iNames, sNames, names, n):
    idx = createStateMatrix(n, names)
    DFESubs = {stateVars[j]: 0 for name in iNames for j in idx[name]}
    vars = [stateVars[j] for name in sNames for j in idx[name]]
    eqs = [sp.Eq(rhsList[j].subs(DFESubs), 0) for name in sNames for j in idx[name]]

    sol = sp.solve(eqs, vars, dict=True)
    DFEPoint = dict(DFESubs)
    if sol:
        DFEPoint.update(sol[0])
    return DFEPoint, sol

def printPoint(DFEPoint):
    for sym in sorted(DFEPoint.keys(), key=str):
        print(f"  {sym}* = {sp.simplify(DFEPoint[sym])}")

def checkStability(stateVars, rhsList, point):
    sv = list(stateVars)
    rhs = list(rhsList)
    J = sp.Matrix(rhs).jacobian(sp.Matrix(sv))
    Jmat = np.array(J.subs(point).evalf(), dtype=float)
    eigs = np.linalg.eigvals(Jmat)

    if bool(np.all(eigs.real < 0)):
        string = "Stable"
    else:
        string = "Unstable"    

    return string

def findEE(stateVars, rhsList, guess):
    sv = list(stateVars)
    rhs = list(rhsList)

    F = sp.lambdify(stateVars, rhsList, 'numpy')
    J_sym = sp.Matrix(rhs).jacobian(sp.Matrix(sv))
    J_func = sp.lambdify(stateVars, J_sym, 'numpy')

    def Fv(x):
        return np.array(F(*x), dtype=float)

    def Jv(x):
        return np.array(J_func(*x), dtype=float)

    x0 = np.array([guess[s] for s in stateVars], dtype=float)
    res = root(Fv, x0, jac=Jv, method='lm')

    EE = {s: v for s, v in zip(stateVars, res.x)}
    converged = res.success
    return EE, converged, res.message

def sameEquilibrium(point1, point2, stateVars, tol):
    diff = [abs(float(point1[s]) - float(point2[s])) for s in stateVars]
    maxDiff = max(diff)

    if maxDiff < tol:
        print("EE = DFE from solver")
    else:
        print("EE != DFE from solver")    