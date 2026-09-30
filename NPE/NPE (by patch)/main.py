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

# Recursion Limit
sys.setrecursionlimit(100000)

#ACTUAL PARAMS
# b_su = 0.6 # bite rate on susceptible
# p_sy = 0.288 #probability of being symptomatic

paramTrue = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.288])

#Read data
data = np.loadtxt("vector.csv", delimiter=",")
data = data.tolist()

# Model params
paramNames=[r"$b_{su1}$", r"$b_{su2}$", r"$b_{su3}$", r"$b_{su4}$", r"$b_{su5}$", r"$p_{sy}$"]
n = 5 # Number of patches

# Number of rounds
nRounds = 5

# Priors and Initialisation
posteriors = []
prior_min = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
prior_max = [1, 1, 1, 1, 1, 1]
prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)
prior, numParameters, priorReturns = process_prior(prior)
proposal = prior

#Prelim
inference = NPE(prior)
j.simulationWrapper = process_simulator(j.simulationWrapper, prior, priorReturns)
check_sbi_inputs(j.simulationWrapper, prior)
obsSummStat = j.summaryStats(data, n)
x_o = torch.as_tensor(obsSummStat, dtype=torch.float32)

for _ in range(nRounds):
    theta, x = simulate_for_sbi(j.simulationWrapper, proposal, num_simulations=50000, num_workers=100)
    density_estimator = inference.append_simulations(theta, x, proposal=proposal, exclude_invalid_x=True).train()
    posterior = inference.build_posterior(density_estimator, sample_with='mcmc')
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)

#Create observed data stats
obsSummStat = j.summaryStats(data, n)
posterior_theta = posterior.sample((100000,), x=obsSummStat)
fig, axes = analysis.pairplot(posterior_theta, labels = paramNames, points=paramTrue[None, :], points_colors="red")
plt.savefig("Pairwise plot", dpi=1000, bbox_inches="tight")

meanParam = posterior_theta.numpy().mean(axis=0)
print("Mean param")
print(meanParam)

log_probs = posterior.log_prob(posterior_theta)
modeParam = posterior_theta[log_probs.argmax()]
print("Mode param")
print(modeParam)

# Plot mean
estSim = j.MalariaModel(meanParam)
iCompMean= [estSim[0][2+9*k] for k in range(0,n)]
iCompMean.extend([estSim[0][3+9*k] for k in range(0,n)])
t = estSim[1]
plt.figure()
plt.plot(t, estSim[0][2], label=r"$I_{est}$")
plt.plot(t, data[0], label=r"$I_{data}$")
plt.plot(t, estSim[0][3], label=r"$A_{est}$")
plt.plot(t, data[5], label=r"$A_{data}$")
plt.xlabel("Time")
plt.ylabel("Patch 1 Infected/Asymptomatic Individuals")
plt.title("Estimated Patch 1 Trajectory vs Actual Trajectory")
plt.legend()
plt.savefig("Estimated vs Actual Trajectory (mean)", dpi=1000, bbox_inches="tight")

# Plot mode
estSim = j.MalariaModel(modeParam)
iCompMode= [estSim[0][2+9*k] for k in range(0,n)]
iCompMode.extend([estSim[0][3+9*k] for k in range(0,n)])
t = estSim[1]
plt.figure()
plt.plot(t, estSim[0][2], label=r"$I_{est}$")
plt.plot(t, data[0], label=r"$I_{data}$")
plt.plot(t, estSim[0][3], label=r"$A_{est}$")
plt.plot(t, data[5], label=r"$A_{data}$")
plt.xlabel("Time")
plt.ylabel("Patch 1 Infected/Asymptomatic Individuals")
plt.title("Estimated Patch 1 Trajectory vs Actual Trajectory")
plt.legend()
plt.savefig("Estimated vs Actual Trajectory (mode)", dpi=1000, bbox_inches="tight")

# Print summary stats
print("Estimated stats (Mean)")
print(j.summaryStats(iCompMean, n))
print("Estimated stats (Mode)")
print(j.summaryStats(iCompMode, n))
print("Data stats")
print(obsSummStat)

# Save posterior
torch.save(posterior, "posterior.pt")