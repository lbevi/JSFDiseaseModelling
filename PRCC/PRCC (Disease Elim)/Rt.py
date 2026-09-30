import sympy as sp
import numpy as np
import model as m
from model import V_ON

# Calculating R0 for n-patch model
def calculateR0(n, hPatch, mPatch, dPatch, mParam, L):
    # Variables
    # States
    iNames = ["E_h", "I_h", "A_h", "E_m", "I_m"]
    sNames =  ["S_h", "E_h", "I_h", "A_h", "R_h", "S_m", "E_m", "I_m", "N_h", "N_m", "V_h"]
    allNames = ["E_h", "I_h", "A_h", "S_h", "R_h", "S_m", "E_m", "I_m", "V_h"] 
    states = {name: sp.symbols(f"{name}0:{n}") for name in sNames}

    # Constants
    cNames = ["b_sy", "b_asy", "b_su", "o_h", "o_m", "mu_h", "mu_m", "T_mh", "T_hm", "a", "d_h", "gI_h", "gA_h", "L_h", "L_m", "N_h", "N_m", "chi", "v", "beta", "phi"]
    params = {name: sp.symbols(f"{name}0:{n}") for name in cNames}

    # Transfer terms
    t = {}

    for c in allNames:
        t[c] = sp.Matrix(n, n, lambda i, j: 0 if i==j else sp.symbols(f"t_{c}_{i+1}_{j+1}"))

    # Substitution of params
    subVals = {}

    for i in range(n):
        subVals[params["b_sy"][i]] = mPatch[i]["b_sy"]
        subVals[params["b_asy"][i]] = mPatch[i]["b_asy"]
        subVals[params["b_su"][i]] = mPatch[i]["b_su"]
        subVals[params["T_mh"][i]] = dPatch[i]["t_hm"]
        subVals[params["T_hm"][i]] = dPatch[i]["t_mh"]
        subVals[params["o_h"][i]] = dPatch[i]["o_h"]
        subVals[params["mu_h"][i]] = hPatch[i]["u_h"]
        subVals[params["L_h"][i]] = hPatch[i]["L_h"]
        subVals[params["d_h"][i]] = dPatch[i]["d_h"]
        subVals[params["gI_h"][i]] = dPatch[i]["gI_h"]
        subVals[params["gA_h"][i]] = dPatch[i]["gA_h"]
        subVals[params["a"][i]] = dPatch[i]["p_sy"]
        subVals[params["o_m"][i]] = dPatch[i]["o_m"]
        subVals[params["mu_m"][i]] = mPatch[i]["u_m"]
        subVals[params["chi"][i]] = dPatch[i]["chi"]
        subVals[params["v"][i]] = 0
        subVals[params["beta"][i]] = hPatch[i]["beta"]
        subVals[params["phi"][i]] = hPatch[i]["phi"]
        subVals[params["L_m"][i]] = L[i]
        
        for j in range(n):
            subVals[t["E_h"][i, j]] = mParam["E_h"][i, j]
            subVals[t["I_h"][i, j]] = mParam["I_h"][i, j]
            subVals[t["A_h"][i, j]] = mParam["A_h"][i, j]
            subVals[t["E_m"][i, j]] = mParam["E_m"][i, j]
            subVals[t["I_m"][i, j]] = mParam["I_m"][i, j]
            subVals[t["S_h"][i, j]] = mParam["S_h"][i, j]
            subVals[t["V_h"][i, j]] = mParam["V_h"][i, j]
            subVals[t["S_m"][i, j]] = mParam["S_m"][i, j]

    #Finding DFE
    #Humans (after vac on)
    # Set up system of equations, S_H written first, then V_H
    NH_IC = [0 for i in range(n)]
    for i in range(n):
        for c in hPatch[i]["ic"].keys():
            NH_IC[i] += hPatch[i]["ic"][c]
            subVals[params["v"][i]] = hPatch[i]["v"]

    A_H = sp.zeros(2*n)
    b_H = sp.zeros(2*n,1)

    for i in range(n):
        b_H[i] = params["L_h"][i]*NH_IC[i]

    for i in range(n):
        b_H[n+i] = 0

    for i in range(n):
        #S_H equations
        A_H[i,i] = params["mu_h"][i] + sum(t["S_h"][i,j] for j in range(n)) + params["v"][i]  #outflow
        for j in range(n):
            if i != j:
                A_H[i,j] = - t["S_h"][j,i]  # inflow from j
        A_H[i, n+i] = -params["phi"][i] #V_H term
        
        #V_H equations    
        A_H[n+i, n+i]= params["mu_h"][i] + sum(t["V_h"][i,j] for j in range(n)) + params["phi"][i] #outflow
        for j in range(n):
            if i != j:
                A_H[n+i,n+j] = - t["V_h"][j,i]  # inflow from j
        A_H[n+i, i] = -params["v"][i] #S_H term
 
    A_H = A_H.subs(subVals)
    b_H = b_H.subs(subVals)

    sol = A_H.LUsolve(b_H)
    SH_DFE_VAC = [sol[i] for i in range(n)]
    VH_DFE_VAC = [sol[i+n] for i in range(n)]      

    # Mosquitoes
    NM_IC = [0 for i in range(n)]
    for i in range(n):
        for c in mPatch[i]["ic"].keys():
            NM_IC[i] += mPatch[i]["ic"][c]
    A_H = sp.zeros(n)
    b_H = sp.Matrix([params["L_m"][i]*NM_IC[i] for i in range(n)])

    for i in range(n):
        A_H[i,i] = params["mu_m"][i] + sum(t["S_m"][i,j] for j in range(n))  # outflow
        for j in range(n):
            if i != j:
                A_H[i,j] = - t["S_m"][j,i]  # inflow from j

    A_H = A_H.subs(subVals)
    b_H = b_H.subs(subVals)

    MH_DFE = A_H.LUsolve(b_H)  

    # F vector
    F_list = []
    for i in range(n):
        f1 = params["b_su"][i] * params["T_mh"][i] * states["I_m"][i] * states["S_h"][i] / (states["N_h"][i]) + params["beta"][i]*params["b_su"][i] * params["T_mh"][i] * states["I_m"][i] * states["V_h"][i] / (states["N_h"][i])
        f2 = 0
        f3 = 0
        f4 = (params["b_sy"][i] * states["I_h"][i] + params["b_asy"][i]* params["chi"][i] * states["A_h"][i]) * params["T_hm"][i] * states["S_m"][i] / (states["N_h"][i])
        f5 = 0
        
        F_list.extend([f1, f2, f3, f4, f5])   # append block for patch i

    F = sp.Matrix(F_list)

    # V vector
    V_list = []
    for i in range(n):
        # If 1 patch (no transfer terms)
        if n == 1:
            t_out_E_h = sp.S(0)
            t_out_I_h = sp.S(0)
            t_out_A_h = sp.S(0)
            t_out_E_m = sp.S(0)
            t_out_I_m = sp.S(0)
        else: # If n-patch
            t_out_E_h = sum(t["E_h"][i, k] for k in range(n))
            t_out_I_h = sum(t["I_h"][i, k] for k in range(n))
            t_out_A_h = sum(t["A_h"][i, k] for k in range(n))
            t_out_E_m = sum(t["E_m"][i, k] for k in range(n))
            t_out_I_m = sum(t["I_m"][i, k] for k in range(n))        

        v1 = (params["o_h"][i] + params["mu_h"][i] + t_out_E_h)*states["E_h"][i] - sum(t["E_h"][k,i]*states["E_h"][k] for k in range(n))  
        v2 = (params["d_h"][i] + params["mu_h"][i] + params["gI_h"][i] + t_out_I_h)*states["I_h"][i] - params["o_h"][i]*params["a"][i]*states["E_h"][i] - sum(t["I_h"][k,i]*states["I_h"][k] for k in range(n))
        v3 = (params["mu_h"][i] + params["gA_h"][i] + t_out_A_h)*states["A_h"][i] - params["o_h"][i]*(1-params["a"][i])*states["E_h"][i] - sum(t["A_h"][k,i]*states["A_h"][k] for k in range(n))
        v4 = (params["o_m"][i] + params["mu_m"][i] + t_out_E_m)*states["E_m"][i] - sum(t["E_m"][k,i]*states["E_m"][k] for k in range(n))
        v5 = (params["mu_m"][i] + t_out_I_m)*states["I_m"][i] - params["o_m"][i]*states["E_m"][i] - sum(t["I_m"][k,i]*states["I_m"][k] for k in range(n))
        V_list.extend([v1, v2, v3, v4, v5])

    V = sp.Matrix(V_list)

    # Create infected states list
    stateList = []
    for i in range(n):
        for c in iNames:
            stateList.append(states[c][i])

    iStates = sp.Matrix(stateList)

    JF = F.jacobian(iStates)
    JV = V.jacobian(iStates)

    # DFE (after vac)
    DFE_VAC = {}

    for i in range(n):
        DFE_VAC[states["S_h"][i]] = SH_DFE_VAC[i]
        DFE_VAC[states["V_h"][i]] = VH_DFE_VAC[i]
        DFE_VAC[states["E_h"][i]] = 0
        DFE_VAC[states["I_h"][i]] = 0
        DFE_VAC[states["A_h"][i]] = 0
        DFE_VAC[states["R_h"][i]] = 0
        DFE_VAC[states["S_m"][i]] = MH_DFE[i]
        DFE_VAC[states["E_m"][i]] = 0
        DFE_VAC[states["I_m"][i]] = 0
        DFE_VAC[states["N_h"][i]] = SH_DFE_VAC[i] + VH_DFE_VAC[i]
        DFE_VAC[states["N_m"][i]] = MH_DFE[i]

    JF_DFE_VAC = JF.subs(subVals).subs(DFE_VAC)
    JV_DFE_VAC = JV.subs(subVals).subs(DFE_VAC)
        
    JF_np = np.array(JF_DFE_VAC.tolist(), dtype=float)
    JV_np = np.array(JV_DFE_VAC.tolist(), dtype=float)

    NGM = JF_np @ np.linalg.inv(JV_np)

    EV = np.linalg.eigvals(NGM)

    sRadius = [abs(complex(i)) for i in EV]

    Rt=max(sRadius)
    
    return Rt