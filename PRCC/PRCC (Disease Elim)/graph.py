import matplotlib.pyplot as plt
import numpy as np
import os

PRCCval = [0.008273, 0.003474, 0.008162, 0.014419, 0.025954, 0.016008, -0.002580, 0.001584, 0.005476, -0.005300,
        -0.005699, 0.708252, 0.016059, 0.002065, 0.078509, 0.009863, 0.258610, 0.009150, 0.006256, 0.048684,
        -0.002599, -0.376201, 0.002524, 0.004145, -0.038083, -0.011956]   

paramName= [r"$\nu_{(1)}$", r"$\nu_{(2)}$", r"$\nu_{(3)}$", r"$\nu_{(4)}$", r"$\nu_{(5)}$", r"$\beta$",
            r"$b_{I_{H}(1)}$", r"$b_{I_{H}(2)}$", r"$b_{I_{H}(3)}$", r"$b_{I_{H}(4)}$", r"$b_{I_{H}(5)}$",
            r"$b_{A_{H}(1)}$", r"$b_{A_{H}(2)}$", r"$b_{A_{H}(3)}$", r"$b_{A_{H}(4)}$", r"$b_{A_{H}(5)}$",
            r'$\eta_{(1)}$', r'$\eta_{(2)}$', r'$\eta_{(3)}$', r'$\eta_{(4)}$', r'$\eta_{(5)}$',
            r'$\mu_{M(1)}$', r'$\mu_{M(2)}$', r'$\mu_{M(3)}$', r'$\mu_{M(4)}$', r'$\mu_{M(5)}$']

# Print bar graph
plt.figure(figsize=(12,6))
bars = plt.bar(paramName, PRCCval)
plt.axhline(0, color="black", linewidth=1)
plt.xticks(fontsize=9)
plt.xlabel("Parameter", fontsize = 14)
plt.ylabel("PRCC", fontsize = 14)
for bar in bars[0:5]:
    bar.set_color("red")
bars[5].set_color("orange")
for bar in bars[11:16]:
    bar.set_color("purple")
for bar in bars[16:21]:
    bar.set_color("black")
for bar in bars[21:26]:
    bar.set_color("green")
plt.savefig("PRCC Disease Elim", dpi=150, bbox_inches="tight")
plt.show()
