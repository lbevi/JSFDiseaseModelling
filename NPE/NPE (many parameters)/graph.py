import pandas as pd
import sys
import numpy as np
import BaseModel as j
from sbi.analysis import pairplot
from sbi.utils import BoxUniform
import mjsf
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import os
import multiprocessing as mp
import corner
import matplotlib.patheffects as pe

# Recursion Limit
sys.setrecursionlimit(100000)

#ACTUAL PARAMS
# b_su = 0.6 # bite rate on susceptible
# b_sy = 0.4 # bite rate on symptomatic
# p_sy = 0.288 #probability of being symptomatic
# w_h = 0.01 # waning immunity rate
# chi = 0.75 # infectivity percentage when asymptomatic
# u_m = 0.0477 # mosquito death rate
paramTrue = np.array([0.6, 0.4, 0.288, 0.01, 0.75, 0.0477])

#Read data and posterior
data = np.loadtxt("vector.csv", delimiter=",")
data = data.tolist()
posterior = torch.load("posterior.pt", weights_only=False)

# Model params
paramNames=[r"$b_{S_{H}}$", r"$b_{I_{H}}$", r"$\alpha$", r"$\omega_{H}$", r"$\chi$", r"$\mu_{M}$"]
n = 5 # Number of patches

# Number of rounds
nRounds = 5

# Priors and Initialisation
posteriors = []
prior_min = [0.01, 0.01, 0.01, 0.0001, 0.01, 0.0001]
prior_max = [1, 1, 1, 0.1, 1, 0.1]
prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)
prior, numParameters, priorReturns = process_prior(prior)
proposal = prior

#Create observed data stats
obsSummStat = j.summaryStats(data, n)
posterior_theta = posterior.sample((100000,), x=obsSummStat)
fig, axes = analysis.pairplot(posterior_theta, labels = paramNames, points=paramTrue[None, :], points_colors="red")
plt.savefig("Pairwise plot", dpi=1000, bbox_inches="tight")

medianParam = np.median(posterior_theta.numpy(), axis=0)
print("Median param")
print(medianParam)

# Get 100 param samples
posterior_sample = posterior_theta[:100]
sampleSim = []
for x in posterior_sample:
    sampleSim.append(j.MalariaModel(medianParam))    
sampleSim = np.transpose(np.stack([s[0] for s in sampleSim]), (1,0,2))
ciVec = [95, 80, 50]

# Plot median
estSim = j.MalariaModel(medianParam)
iCompMean= [estSim[0][2+9*k] for k in range(0,n)]
iCompMean.extend([estSim[0][3+9*k] for k in range(0,n)])

fig, axs = plt.subplots(n, 2, squeeze=False, constrained_layout=True, figsize=(12, 2*n))

t = estSim[1]
for i in range(n):
    axs[i, 0].annotate( f"Patch {i+1}",xy=(-0.20, 0.5), xycoords="axes fraction",rotation=90,ha="center",va="center",fontsize=14)    
    for ci in ciVec:
        low = np.maximum(0,np.percentile(sampleSim[3+9*i], 50 - ci / 2, axis=0))
        high = np.percentile(sampleSim[3+9*i], 50 + ci / 2, axis=0)
        axs[i, 0].fill_between(t, low, high, color='r', alpha=0.2)
    
    axs[i, 0].plot(t, data[5+i], label = r"$A_{data}$", color='green')
    axs[i, 0].plot(t, np.maximum(0,iCompMean[5+i]), label = r"$A_{est}$", linestyle='--', linewidth=2, color='blue')
    if np.any(np.array(sampleSim[3+9*i]) > 500):
        axs[i, 0].axhline(y=500, color="k", linestyle="--")
    if i!=n-1:
        axs[i, 0].set_xticks([])
    axs[i, 0].tick_params(axis='both', labelsize=14)

    for ci in ciVec:
        low = np.maximum(0,np.percentile(sampleSim[2+9*i], 50 - ci / 2, axis=0))
        high = np.percentile(sampleSim[2+9*i], 50 + ci / 2, axis=0)
        axs[i, 1].fill_between(t, low, high, color='r', alpha=0.2)
        
    axs[i, 1].plot(t, data[0+i], label = r"$A_{data}$", color='green')
    axs[i, 1].plot(t, np.maximum(0,iCompMean[0+i]), label = r"$A_{est}$", linestyle='--', linewidth=2, color='blue')
    if np.any(np.array(sampleSim[2+9*i]) > 500):
        axs[i, 0].axhline(y=500, color="k", linestyle="--")
    if i!=n-1:
         axs[i, 1].set_xticks([])
    axs[i, 1].tick_params(axis='both', labelsize=14)

axs[0, 0].set_title(r"$A_{H}$", fontsize=14)
axs[0, 1].set_title(r"$I_{H}$", fontsize=14)
axs[n-1, 0].set_xlabel("Time (days)", fontsize=14)
axs[n-1, 1].set_xlabel("Time (days)", fontsize=14)
plt.savefig("Median Trajectory", dpi=500, bbox_inches="tight")
plt.show()

# Corner plot
posterior_theta_np = posterior_theta.detach().cpu().numpy()
ranges = []

for i in range(len(paramTrue)):
    samples = posterior_theta_np[:, i]

    lower = min(samples.min(), paramTrue[i])
    upper = max(samples.max(), paramTrue[i])

    # Add 5% padding
    padding = 0.05 * (upper - lower)

    ranges.append((lower - padding, upper + padding))

plt.figure()
figure = corner.corner(
    posterior_theta_np,
    labels=paramNames,
    quantiles=[0.025, 0.5, 0.975],
    show_titles=True,
    title_fmt=".2g",
    truths=paramTrue,
    truth_color="red",
    title_kwargs={"fontsize": 16},
    labelpad=0.1
)

for ax in figure.get_axes():
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.label.set_size(16)                
    ax.yaxis.label.set_size(16)                


plt.savefig("Corner Plot", dpi=150, bbox_inches="tight")

# Print summary stats
# print("Estimated stats (Mean)")
# print(j.summaryStats(iCompMean, n))
# print("Estimated stats (Mode)")
# print(j.summaryStats(iCompMode, n))
# print("Data stats")
# print(obsSummStat)