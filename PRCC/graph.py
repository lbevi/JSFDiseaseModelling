import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Alter the directories below!

# Loading results
# no weather, no transfer
PRCCResults = pd.read_csv("D:/Storage/Work/Homework/Thesis/Code/Final Code/PRCC/PRCC (R0) (no weather + no transfer)/PRCCvals")
modelNWNT = [
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy1}$", "PRCC"].iloc[0],
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy2}$", "PRCC"].iloc[0],
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy3}$", "PRCC"].iloc[0],
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy4}$", "PRCC"].iloc[0],
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy5}$", "PRCC"].iloc[0]
]

#no weather
PRCCResults = pd.read_csv("D:/Storage/Work/Homework/Thesis/Code/Final Code/PRCC/PRCC (R0) (no weather)/PRCCvals")
modelNW = [
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy1}$", "PRCC"].iloc[0],
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy2}$", "PRCC"].iloc[0],
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy3}$", "PRCC"].iloc[0],
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy4}$", "PRCC"].iloc[0],
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy5}$", "PRCC"].iloc[0]
]

#no weather + equal transfer
PRCCResults = pd.read_csv("D:/Storage/Work/Homework/Thesis/Code/Final Code/PRCC/PRCC (R0) (no weather + equal transfer)/PRCCvals")
modelNWET = [
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy1}$", "PRCC"].iloc[0],
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy2}$", "PRCC"].iloc[0],
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy3}$", "PRCC"].iloc[0],
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy4}$", "PRCC"].iloc[0],
    PRCCResults.loc[PRCCResults["Parameter"] == "$b_{asy5}$", "PRCC"].iloc[0]
]
modelNWET = np.array(modelNWET)

# standard
ModelS = np.array([0.403359,   0.166652,  0.286542,   0.148510,   0.305841])   

PRCCval = np.concatenate([modelNWNT, modelNW, modelNWET, ModelS])

paramName= [r"$(1)$", r"$(2)$", r"$(3)$", r"$(4)$", r"$(5)$"]
x = np.arange(20)

# Print bar graph
plt.figure(figsize=(12,6))
bars = plt.bar(x, PRCCval)
plt.xticks(x, paramName * 4, fontsize=8)
plt.axhline(0, color="black", linewidth=1)
plt.xticks(fontsize=10)
plt.xlabel("Patch", fontsize = 14)
plt.ylabel("PRCC", fontsize = 14)
for bar in bars[0:5]:
    bar.set_color("red")
for bar in bars[5:10]:
    bar.set_color("purple")
for bar in bars[10:15]:
    bar.set_color("black")
for bar in bars[15:20]:
    bar.set_color("green")
plt.savefig("PRCC R0 different models", dpi=150, bbox_inches="tight")
plt.show()

tH = np.zeros((5,5)) #human transfer
tH[0,1] = 10/25000; tH[1,0] = 10/1000 #edge 1<->2
tH[0,2] = 0.1/25000; tH[2,0] = 0.1/50 #edge 1<->3
tH[0,3] = 100/25000; tH[3,0] = 100/5000 #edge 1<->4
tH[1,2] = 0.1/1000; tH[2,1] = 0.1/50 #edge 2<->3
tH[3,4] = 0.05/5000; tH[4,3] = 0.05/15 #edge 4<->5
outflow = tH.sum(axis=1)
inflow = tH.sum(axis=0)
tflow = outflow+inflow
print("outflow")
print(outflow)
print("inflow")
print(inflow)
print("total flow")
print(tflow)
# tH = np.zeros((5,5)) #human transfer
# tM = np.zeros((5,5)) #mosquito transfer

# tH[0,1] = 0.001; tH[1,0] = 0.001 #edge 1<->2
# tH[0,2] = 0.001; tH[2,0] = 0.001 #edge 1<->3
# tH[0,3] = 0.001; tH[3,0] = 0.001 #edge 1<->4
# tH[1,2] = 0.001; tH[2,1] = 0.001 #edge 2<->3
# tH[3,4] = 0.001; tH[4,3] = 0.001 #edge 4<->5
# col_sum = [tH[i, :].sum() + tH[:, i].sum() for i in range (0,5)]
# print("sums")
# print(col_sum)