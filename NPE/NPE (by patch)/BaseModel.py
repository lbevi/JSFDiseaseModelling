import mjsf
import torch
import numpy as np
import model as m
import sys
sys.path.append("/data/gpfs/projects/punim2894/Parameter Estimation/mjsf")
import os
#os.chdir("D:/Storage/Work/Homework/Thesis/Code/Final Code/Model Fitting")

def MalariaModel(params):
    
    # Check if tensor
    if torch.is_tensor(params):
        params = params.detach().numpy()

    #Params
    b_su1=params[0]
    b_su2=params[1]
    b_su3=params[2]
    b_su4=params[3]
    b_su5=params[4]
    p_sy=params[5]

    # Compartments and Patch Number
    comp = ["S_h", "E_h", "I_h", "A_h", "D_h", "R_h", "S_m", "E_m", "I_m"]
    hComp = ["S", "E", "I", "A", "D", "R"]
    mComp = ["S", "E", "I"]
    k = len(comp) # number of compartments per patch

    #Human Parameters
    # S, E, I, A, D, R, (IC)
    # u_h, b_h 
    hP1 = m.createHumanPatch(25000, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4)
    hP2 = m.createHumanPatch(1000, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4)
    hP3 = m.createHumanPatch(50, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4)
    hP4 = m.createHumanPatch(5000, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4)
    hP5 = m.createHumanPatch(15, 0, 0, 0, 0, 0,
                                4.5e-5, 1.2e-4)

    hPatch = [hP1, hP2, hP3, hP4, hP5]

    n = len(hPatch); #patch size
    tot_states = n*k #total number of states

    #Mosquito Parameters
    # S, E, I (IC)
    # b_sy, b_asy, b_su, u_m

    # BITE RATES INC BY 0.2

    mP1 = m.createMosquitoPatch(50000, 0, 50,
                                0.4, 0.6, b_su1, 0.0477)
    mP2 = m.createMosquitoPatch(2000, 0, 0,
                                0.4, 0.6, b_su2, 0.0477)
    mP3 = m.createMosquitoPatch(100, 0, 0,
                                0.4, 0.6, b_su3, 0.0477)
    mP4 = m.createMosquitoPatch(10000, 0, 0,
                                0.4, 0.6, b_su4, 0.0477)
    mP5 = m.createMosquitoPatch(30, 0, 0,
                                0.4, 0.6, b_su5, 0.0477)

    mPatch = [mP1, mP2, mP3, mP4, mP5]

    #Disease Parameters
    # d_h, gI_h, gA_h w_h, o_h, o_m, t_hm, t_mh, p_sy, chi
    dP1 = m.createDiseasePatch(1.8e-3, 0.08, 0.01, 0.01, 1/15, 1/9, 0.022, 0.24, p_sy, 0.75)
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

    # Simulation Parameters
    t_max = 2000 #Max simulation time
    timeShow = 0 #Display current time every timeShow

    # State notation: (i = 0, 1, ..., 8)
    #x[0] = S_h, x[1] = E_h, x[2] = I_h, #x[3] = A_h, #x[4] = D_h, #x[5] = R_h
    #x[6] = S_m, x[7] = E_m, x[8] = I_m
    #x[i+9(n-1)] for states in the nth patch, i is the comp type

    # JSF Options
    my_opts = {
                "EnforceDo": [0]*tot_states,
                "dt": 1,
                "SwitchingThreshold": [0]*tot_states #FOR NOW DETERMINISTIC
            }

    tStep = my_opts["dt"]

    numEvents = 26 # Total number of non-migration events (count in createRatesVector function)

    # Calculate number of total events across all patches
    allEvents = numEvents*n + n*(n-1)*(k-1)  #numEvents * n + migration events (k-1 due to dead patch)

    # Create reactant and product matrix [reactant_matrix, product_matrix] (matrices need to be lists of lists, not np.array)
    rMatrix = m.createReactantMatrix(n, k, numEvents, allEvents)
    pMatrix = m.createProductMatrix(n, k, numEvents, allEvents)

    # Rainfall/temperature variaton
    rngMatrix = m.generateTempRainRNG(n, t_max/tStep, None)

    # Create rates vector for state x at time t
    rates = lambda x, t: m.createRatesVector(x, t, hPatch, mPatch, dPatch, mParam, n, k, comp, rngMatrix, tStep)

    stoich = {
            "nu": [ [a - b for a, b in zip(r1, r2)]
                    for r1, r2 in zip(pMatrix, rMatrix) ],
            "DoDisc": [1]*tot_states,
            "nuReactant": rMatrix,
            "nuProduct": pMatrix,
            }
    sim = mjsf.jsf(IC, rates, stoich, t_max, config=my_opts, timeShow=0, method="exact")
    return sim

def simulationWrapper(params):
    n = 5 # Number of patches
    
    sim = MalariaModel(params)
    iComp= [sim[0][2+9*k] for k in range(0,n)]
    iComp.extend([sim[0][3+9*k] for k in range(0,n)])
    summstats = summaryStats(iComp, n)
    return summstats
    
  
def summaryStats(data, n):
   summStats = []     
   dt = 1
   t_max = 2000
   data = np.asarray(data) 
   I = np.asarray(data[:n])
   A = np.asarray(data[n:2*n])
   
   # Find time after initial outbreak stops
   # Use the A in Patch 1 (avg rate of change over 5 days)
   A1_RoC= (A[0,5:]-A[0,:-5])/ 5
   
   # Find first index when between -5 to 5 after t=100
   idx_outbreak = t_max
   for i in range(100*dt, len(A1_RoC)):
     if -5 <= A1_RoC[i] <=5 :
       idx_outbreak = i
       break
   
   summStats.append(idx_outbreak*dt)
   
   # For A in Patch 1, calculate max of RoC and the time
   summStats.append(np.max(A1_RoC))
   summStats.append(np.argmax(A1_RoC)*dt)
   
   # For all patches, calculate average after outbreak
   for i in range(0,2*n):
     summStats.append(np.mean(data[i, idx_outbreak:]))
   
   # For all patches, calculate area under curve
   for i in range(0,2*n):
     summStats.append(np.trapz(data[i],dx=dt))  
   
   # For all large populations (only A[0,1,3] and I[0,3]) calculate min and max after outbreak time
   summStats.append(np.min(A[0, idx_outbreak:]))
   summStats.append(np.min(A[1, idx_outbreak:])) 
   summStats.append(np.min(A[3, idx_outbreak:])) 
   summStats.append(np.min(I[0, idx_outbreak:])) 
   summStats.append(np.min(I[3, idx_outbreak:]))     
   summStats.append(np.max(A[0, idx_outbreak:]))
   summStats.append(np.max(A[1, idx_outbreak:])) 
   summStats.append(np.max(A[3, idx_outbreak:])) 
   summStats.append(np.max(I[0, idx_outbreak:])) 
   summStats.append(np.max(I[3, idx_outbreak:]))    
    
   
   return summStats 