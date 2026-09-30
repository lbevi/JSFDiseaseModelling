import numpy as np
from scipy.integrate import solve_ivp
import model as m

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

#Mosquito birth function (patch 1-n)
#sin func
def L(t, rngMatrix, i, tStep):
    b_av  = 0.050 #average birth rate #OLD WAS 0.0478
    b_diff = 0.30*b_av #difference between min and max #INCREASED FROM 10%

    currTimeIdx = t/tStep - 1

    return (b_av+b_diff*np.cos(2*np.pi*t/365))*rngMatrix[i][int(currTimeIdx)]
    #return (b_av+b_diff*np.sin(2*np.pi*t/365+phi))
    #return 0.0477

def generateTempRainRNG(n, tTotalSteps, seed):

    rng = np.random.default_rng(seed)
    rho = 0.6  #Correlation
    s = 0.05  #log-normal param
    mu = -0.5*s**2 #Mean of log-normal (equals 1) 
    sigma = 0.05 #MVN sigma 
    CovMat = sigma**2*(rho*np.ones((n,n)) + (1-rho)*np.eye(n)) #CovMat (constant correl for now)
    Z = rng.multivariate_normal(mean=np.zeros(int(n)), cov=CovMat, size = int(tTotalSteps)) # MVN
    X = np.exp(Z+mu) #Convert to log-normal

    return X.T

# exp func
# def L(t):
#     return 1/9*np.exp(-t/365)

#linear func
# def L(t):
#     return 0.0477-0.0005*t

# Create initial condition vector
def createIC(hPatch, mPatch, hComp, mComp, n):
    IC = []

    for i in range(0,n):     
        IC.append(hPatch[i]["ic"]["S"])
        IC.append(hPatch[i]["ic"]["E"])
        IC.append(hPatch[i]["ic"]["I"])
        IC.append(hPatch[i]["ic"]["A"])
        IC.append(hPatch[i]["ic"]["D"])
        IC.append(hPatch[i]["ic"]["R"])
        IC.append(mPatch[i]["ic"]["S"])
        IC.append(mPatch[i]["ic"]["E"])
        IC.append(mPatch[i]["ic"]["I"])
        IC.append(hPatch[i]["ic"]["V"])
    return IC

# Vaccination switch on
V_ON = 1000
def V(t, v):
    if t <= V_ON:
        return 0
    else: 
        return v  

# Nets switch on (births)
def bRed(t, i):
    red = 0.75
    if t >= V_ON and i == 0:
        return red
    else:
        return 1

# Nets switch on (deaths)
def uRed(t, i):
    inc = 1.25
    if t >= V_ON and i == 0:
        return inc
    else:
        return 1

# Create rates vector
def createRatesVector(x, t, hPatch, mPatch, dPatch, mParam, n, k, comp, rngMatrix, tStep):
    rates = []

    # State notation: (i = 0, 1, ..., 8)
    #x[0] = S_h, x[1] = E_h, x[2] = I_h, #x[3] = A_h, #x[4] = D_h, #x[5] = R_h
    #x[6] = S_m, x[7] = E_m, x[8] = I_m
    #x[i+9(n-1)] for states in the nth patch, i is the comp type

    # All events (except migration) for patch i-1
    for i in range(0,n):
        p = k*i # shifts to the next patch
        rates.append(hPatch[i]["L_h"]*hPatch[i]["ic"]["S"]) if x[0+p] > 0 else rates.append(0)  #human birth from susceptible
        rates.append(hPatch[i]["L_h"]*hPatch[i]["ic"]["E"]) if x[1+p] > 0 else rates.append(0)  #human birth from exposed
        rates.append(hPatch[i]["L_h"]*hPatch[i]["ic"]["I"]) if x[2+p] > 0 else rates.append(0)  #human birth from sympto
        rates.append(hPatch[i]["L_h"]*hPatch[i]["ic"]["A"]) if x[3+p] > 0 else rates.append(0)  #human birth from asympto
        rates.append(hPatch[i]["L_h"]*hPatch[i]["ic"]["R"]) if x[5+p] > 0 else rates.append(0)  #human birth from recovered
        rates.append(hPatch[i]["L_h"]*hPatch[i]["ic"]["V"]) if x[9+p] > 0 else rates.append(0)  #human birth from vaccinated
        rates.append(hPatch[i]["u_h"]*x[0+p]) #human susceptible death 
        rates.append(hPatch[i]["u_h"]*x[1+p]) #human exposed death
        rates.append(hPatch[i]["u_h"]*x[2+p]) #human sympto death
        rates.append(hPatch[i]["u_h"]*x[3+p]) #human asympto death
        rates.append(hPatch[i]["u_h"]*x[5+p]) #human recovered death
        rates.append(hPatch[i]["u_h"]*x[9+p]) #human vaccinated death
        rates.append(bRed(t, i)*L(t, rngMatrix, i, tStep)*mPatch[i]["ic"]["S"]) if x[6+p] > 0 else rates.append(0) #mosquito birth from susceptible
        rates.append(bRed(t, i)*L(t, rngMatrix, i, tStep)*mPatch[i]["ic"]["E"]) if x[7+p] > 0 else rates.append(0) #mosquito birth from susceptible 
        rates.append(bRed(t, i)*L(t, rngMatrix, i, tStep)*mPatch[i]["ic"]["I"]) if x[8+p] > 0 else rates.append(0) #mosquito birth from susceptible
        rates.append(uRed(t, i)*mPatch[i]["u_m"]*x[6+p]) #mosquito susceptible death 
        rates.append(uRed(t, i)*mPatch[i]["u_m"]*x[7+p]) #mosquito exposed death 
        rates.append(uRed(t, i)*mPatch[i]["u_m"]*x[8+p]) #mosquito infected death 
        if x[0+p]+x[1+p]+x[2+p]+x[3+p]+x[5+p]+x[9+p] > 0:
            #human becomes exposed due to infected mosquito  
            rates.append(mPatch[i]["b_su"]*dPatch[i]["t_mh"]*x[0+p]*x[8+p]/(x[0+p]+x[1+p]+x[2+p]+x[3+p]+x[5+p]+x[9+p]))
        else: 
            rates.append(0) 
        rates.append(dPatch[i]["o_h"]*dPatch[i]["p_sy"]*x[1+p]) #human becomes sympto from being exposed 
        rates.append(dPatch[i]["o_h"]*(1-dPatch[i]["p_sy"])*x[1+p]) #human becomes asympto from being exposed 
        rates.append(dPatch[i]["d_h"]*x[2+p]) #human dies due to malaria 
        rates.append(dPatch[i]["gI_h"]*x[2+p]) #human sympto recoveres 
        rates.append(dPatch[i]["gA_h"]*x[3+p]) #human asympto recoveres 
        rates.append(dPatch[i]["w_h"]*x[5+p]) #human becomes susceptible
        if x[0+p]+x[1+p]+x[2+p]+x[3+p]+x[5+p]+x[9+p] > 0:
            #mosquito becomes exposed due to sympto human 
            rates.append(mPatch[i]["b_sy"]*dPatch[i]["t_hm"]*x[6+p]*x[2+p]/(x[0+p]+x[1+p]+x[2+p]+x[3+p]+x[5+p]+x[9+p])) 
        else:     
            rates.append(0)

        if x[0+p]+x[1+p]+x[2+p]+x[3+p]+x[5+p]+x[9+p] > 0:
            #mosquito becomes exposed due to asympto human 
            rates.append(mPatch[i]["b_asy"]*dPatch[i]["chi"]*dPatch[i]["t_hm"]*x[6+p]*x[3+p]/(x[0+p]+x[1+p]+x[2+p]+x[3+p]+x[5+p]+x[9+p]))
        else: 
            rates.append(0)         

        rates.append(dPatch[i]["o_m"]*x[7+p]) #mosquito becomes infected from being exposed 
        # Vaccination events
        if x[0+p]+x[1+p]+x[2+p]+x[3+p]+x[5+p]+x[9+p] > 0:
            # vaccinated human becomes exposed due to infected mosquito  
            rates.append(hPatch[i]["beta"]*mPatch[i]["b_su"]*dPatch[i]["t_mh"]*x[9+p]*x[8+p]/(x[0+p]+x[1+p]+x[2+p]+x[3+p]+x[5+p]+x[9+p]))
        else: 
            rates.append(0)
        rates.append(V(t, hPatch[i]["v"])*x[0+p]) #susceptible human becomes vaccinated
        rates.append(hPatch[i]["phi"]*x[9+p]) #vaccinated human waning immunity

    # Migration events (from patch i+1 -> patch j+1)
    comp_index = {c: idx for idx, c in enumerate(comp)} #create index matrix for each state
    for i in range(0,n):
        for j in range(0,n):
            if i == j: #ignores migration from i->i
                continue
            
            p = k*i # shifts to the next patch
            
            for c in comp:
                index = comp_index[c]
                if c != "D_h": # exclude migration for dead states    
                    rates.append(mParam[c][i,j]*x[index+p])                                 
    return rates

# Creates reactant matrix
def createReactantMatrix(n, k, numEvents, allEvents):

    # State notation: (i = 0, 1, ..., 8)
    #x[0] = S_h, x[1] = E_h, x[2] = I_h, #x[3] = A_h, #x[4] = D_h, #x[5] = R_h
    #x[6] = S_m, x[7] = E_m, x[8] = I_m
    #x[i+9(n-1)] for states in the nth patch, i is the comp type

    # Reactant matrix per patch
    rPatch = np.zeros((numEvents, k))

    # Events 
    rPatch[0,0] = 1 #human birth from susceptible
    rPatch[1,1] = 1 #human birth from exposed 
    rPatch[2,2] = 1 #human birth from sympto 
    rPatch[3,3] = 1 #human birth from asympto 
    rPatch[4,5] = 1 #human birth from recovered
    rPatch[5,9] = 1 #human birth from vaccinated  
    rPatch[6,0] = 1 #human susceptible death 
    rPatch[7,1] = 1 #human exposed death
    rPatch[8,2] = 1 #human sympto death (not due to disease) 
    rPatch[9,3] = 1 #human asympto death (not due to disease) 
    rPatch[10,5] = 1 #human recovered death
    rPatch[11,9] = 1 #human vaccinated death 
    rPatch[12,6] = 1 #mosquito birth from susceptible
    rPatch[13,7] = 1 #mosquito birth fron exposed
    rPatch[14,8] = 1 #mosquito birth from infected
    rPatch[15,6] = 1 #mosquito susceptible death 
    rPatch[16,7] = 1 #mosquito exposed death
    rPatch[17,8] = 1 #mosquito infected death
    rPatch[18,0] = 1; rPatch[18,8] = 1; #human becomes exposed due to infected mosquito
    rPatch[19,1] = 1 #human becomes sympto from being exposed 
    rPatch[20,1] = 1 #human becomes asympto from being exposed 
    rPatch[21,2] = 1 #human dies due to malaria
    rPatch[22,2] = 1 #human sympto recoveres
    rPatch[23,3] = 1 #human asympto recoveres 
    rPatch[24,5] = 1 #human becomes susceptible
    rPatch[25,2] = 1; rPatch[25,6] = 1; #mosquito becomes exposed due to sympto human
    rPatch[26,3] = 1; rPatch[26,6] = 1; #mosquito becomes exposed due to asympto human
    rPatch[27,7] = 1 #mosquito becomes infected from being exposed                     
    rPatch[28,9] = 1; rPatch[28,8] = 1; #vaccinated human becomes exposed due to infected mosquito
    rPatch[29,0] = 1; #susceptible human becomes vaccinated
    rPatch[30,9] = 1; #vaccinated human waning immunity

    # Create matrix for non-migration events
    I = np.eye(n)
    r_nonMig = np.kron(I,rPatch)

    # Checking if there are migration events
    if n==1:
        return r_nonMig.tolist()

    # Create matrix for migration events
    r_MigPatch = np.eye(k)
    r_MigPatch = np.delete(r_MigPatch, 4, axis=0) #Ensures no dead human migration

    Columns = []
    for i in range(0,n):  # By column
        colBlocks = []
        for j in range(0,n):  # By row
            if i == j:
                colBlocks.append(np.tile(r_MigPatch,(n-1, 1)))
            else:
                colBlocks.append(np.zeros(((n-1)*(k-1), k))) # k-1 due to the dead state
        Columns.append(np.vstack(colBlocks))
    
    r_Mig = np.hstack(Columns)

    # Total reactant matrix
    r = np.vstack((r_nonMig, r_Mig))
    
    return r.tolist()

# Creates product matrix
def createProductMatrix(n, k, numEvents, allEvents):

    # State notation: (i = 0, 1, ..., 8)
    #x[0] = S_h, x[1] = E_h, x[2] = I_h, #x[3] = A_h, #x[4] = D_h, #x[5] = R_h
    #x[6] = S_m, x[7] = E_m, x[8] = I_m
    #x[i+9(n-1)] for states in the nth patch, i is the comp type 

    # Product matrix per patch
    pPatch = np.zeros((numEvents, k))

    # Events 
    pPatch[0,0] = 2 #human birth from susceptible
    pPatch[1,0] = 1; pPatch[1,1] = 1 #human birth from exposed 
    pPatch[2,0] = 1; pPatch[2,2] = 1 #human birth from sympto 
    pPatch[3,0] = 1; pPatch[3,3] = 1 #human birth from asympto 
    pPatch[4,0] = 1; pPatch[4,5] = 1 #human birth from recovered
    pPatch[5,0] = 1; pPatch[5,9] = 1 #human birth from vaccinated 
    #no change #human susceptible death 
    #no change #human exposed death
    #no change #human sympto death (not due to disease) 
    #no change #human asympto death (not due to disease) 
    #no change #human recovered death
    #no change #human vaccinated death
    pPatch[12,6] = 2; #mosquito birth from susceptible
    pPatch[13,6] = 1; pPatch[13,7] = 1 #mosquito birth fron exposed
    pPatch[14,6] = 1; pPatch[14,8] = 1 #mosquito birth from infected
    #no change #mosquito susceptible death 
    #no change #mosquito exposed death
    #no change #mosquito infected death
    pPatch[18,1] = 1; pPatch[18,8] = 1; #human becomes exposed due to infected mosquito
    pPatch[19,2] = 1 #human becomes sympto from being exposed 
    pPatch[20,3] = 1 #human becomes asympto from being exposed 
    pPatch[21,4] = 1 #human dies due to malaria
    pPatch[22,5] = 1 #human sympto recoveres
    pPatch[23,5] = 1 #human asympto recoveres 
    pPatch[24,0] = 1 #human becomes susceptible
    pPatch[25,2] = 1; pPatch[25,7] = 1; #mosquito becomes exposed due to sympto human
    pPatch[26,3] = 1; pPatch[26,7] = 1; #mosquito becomes exposed due to asympto human
    pPatch[27,8] = 1 #mosquito becomes infected from being exposed
    pPatch[28,1] = 1; pPatch[28,8] = 1; #vaccinated human becomes exposed due to infected mosquito
    pPatch[29,9] = 1; #susceptible human becomes vaccinated
    pPatch[30,0] = 1; #vaccinated human waning immunity  

    # Create matrix for non-migration events
    I = np.eye(n)
    p_nonMig = np.kron(I, pPatch)

    # Checking if there are migration events
    if n==1:
        return p_nonMig.tolist()

    # Create matrix for migration events
    p_MigPatch = np.eye(k)
    p_MigPatch = np.delete(p_MigPatch, 4, axis=0) #Ensures no dead human migration

    p_Mig = np.kron(np.eye(n), p_MigPatch)
    p_Mig = np.tile(p_Mig,(n, 1))
   
    indexDel = [j+i*(n+1)*(k-1) for i in range(0,n) for j in range(0,k-1)] #k-1 due to dead states
    p_Mig= np.delete(p_Mig, indexDel, axis=0)    

    # Total reactant matrix
    p = np.vstack((p_nonMig, p_Mig))

    return p.tolist()

# Solves the deterministic ODEs
def solveODE(hPatch, mPatch, dPatch, mParam, n, k, IC, t_max, rngMatrix, tStep):

    x = solve_ivp(systemODE, (0, t_max), np.array(IC), t_eval = np.linspace(0, t_max, 100*t_max), args=(hPatch, mPatch, dPatch, mParam, n, k, rngMatrix, tStep))

    return x


# Defines ODE equation
def systemODE(t, x, hPatch, mPatch, dPatch, mParam, n, k, rngMatrix, tStep):
    dxdt = np.zeros_like(x)

    for i in range(n):
        #Define states
        S_h, E_h, I_h, A_h, D_h, R_h = x[i*k:i*k+6]
        S_m, E_m, I_m = x[i*k+6:i*k+9]
        V_h = x[i*k+9]

        N_h = S_h + E_h + I_h + A_h + R_h + V_h
        N_m = S_m + E_m + I_m

        λ_h = mPatch[i]["b_su"] * dPatch[i]["t_mh"] * I_m / (N_h+1e-12)
        λ_m = (mPatch[i]["b_sy"] * I_h + mPatch[i]["b_asy"] * dPatch[i]["chi"] * A_h) * dPatch[i]["t_hm"] / (N_h+1e-12)

        NH_IC = 0
        for c in hPatch[i]["ic"].keys():
            NH_IC += hPatch[i]["ic"][c]

        NM_IC = 0
        for c in mPatch[i]["ic"].keys():
            NM_IC += mPatch[i]["ic"][c]        

        #Human equations
        #S_h
        dxdt[i*k+0] = hPatch[i]["L_h"]*NH_IC+ dPatch[i]["w_h"]*R_h + hPatch[i]["phi"]*V_h  - λ_h*S_h - (hPatch[i]["u_h"] + V(t, hPatch[i]["v"]) + sum(mParam["S_h"][i,j] for j in range(n)))*S_h + sum(mParam["S_h"][j,i]*x[j*k+0] for j in range(n))
        #E_h
        dxdt[i*k+1] = λ_h*S_h + hPatch[i]["beta"]*λ_h*V_h - (dPatch[i]["o_h"] + hPatch[i]["u_h"] + sum(mParam["E_h"][i,j] for j in range(n)))*E_h + sum(mParam["E_h"][j,i]*x[j*k+1] for j in range(n))
        #I_h
        dxdt[i*k+2] = dPatch[i]["o_h"]*dPatch[i]["p_sy"]*E_h - (dPatch[i]["gI_h"] + hPatch[i]["u_h"] + dPatch[i]["d_h"] + sum(mParam["I_h"][i,j] for j in range(n)))*I_h + sum(mParam["I_h"][j,i]*x[j*k+2] for j in range(n))
        #A_h
        dxdt[i*k+3] = dPatch[i]["o_h"]*(1-dPatch[i]["p_sy"])*E_h - (dPatch[i]["gA_h"] + hPatch[i]["u_h"] + sum(mParam["A_h"][i,j] for j in range(n)))*A_h + sum(mParam["A_h"][j,i]*x[j*k+3] for j in range(n))
        #D_h
        dxdt[i*k+4] = dPatch[i]["d_h"]*I_h
        #R_h
        dxdt[i*k+5] = dPatch[i]["gI_h"]*I_h + dPatch[i]["gA_h"]*A_h - (dPatch[i]["w_h"] + hPatch[i]["u_h"] + sum(mParam["R_h"][i,j] for j in range(n)))*R_h + sum(mParam["R_h"][j,i]*x[j*k+5] for j in range(n))
        #V_h
        dxdt[i*k+9] = V(t, hPatch[i]["v"])*S_h - hPatch[i]["beta"]*λ_h*V_h - (hPatch[i]["u_h"] + hPatch[i]["phi"] + sum(mParam["V_h"][i,j] for j in range(n)))*V_h + sum(mParam["V_h"][j,i]*x[j*k+9] for j in range(n))

        #Mosquito equations
        #S_m 
        dxdt[i*k+6] =  bRed(t, i)*L(t, rngMatrix, i, tStep)*NM_IC - λ_m*S_m - (uRed(t, i)*mPatch[i]["u_m"] + sum(mParam["S_m"][i,j] for j in range(n)))*S_m + sum(mParam["S_m"][j,i]*x[j*k+6] for j in range(n))
        #E_m
        dxdt[i*k+7] = λ_m*S_m - (dPatch[i]["o_m"] + uRed(t, i)*mPatch[i]["u_m"] + sum(mParam["E_m"][i,j] for j in range(n)))*E_m + sum(mParam["E_m"][j,i]*x[j*k+7] for j in range(n))
        #I_m
        dxdt[i*k+8] = dPatch[i]["o_m"]*E_m - (uRed(t, i)*mPatch[i]["u_m"] + sum(mParam["I_m"][i,j] for j in range(n)))*I_m + sum(mParam["I_m"][j,i]*x[j*k+8] for j in range(n))

    return dxdt