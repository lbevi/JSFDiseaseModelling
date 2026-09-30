import matplotlib.pyplot as plt
import numpy as np
import os


PRCCval = [-0.024736, -0.009081, -0.013357, -0.007466, -0.015080, 0.314377,  0.002883,   0.005612,   0.001827,  
0.003537,   0.002260,   0.162493,   0.108985,  0.135910,   0.101546,   0.139611,   0.103209,   0.068489,   
0.086599,   0.063653,   0.089605,   -0.342190,   -0.262506,  -0.308102,   -0.243933,  -0.313839]   

paramName= [r"$\nu_{(1)}$", r"$\nu_{(2)}$", r"$\nu_{(3)}$", r"$\nu_{(4)}$", r"$\nu_{(5)}$", r"$\beta$",
            r"$b_{I_{H}(1)}$", r"$b_{I_{H}(2)}$", r"$b_{I_{H}(3)}$", r"$b_{I_{H}(4)}$", r"$b_{I_{H}(5)}$",
            r"$b_{A_{H}(1)}$", r"$b_{A_{H}(2)}$", r"$b_{A_{H}(3)}$", r"$b_{A_{H}(4)}$", r"$b_{A_{H}(5)}$",
            r'$\eta_{(1)}$', r'$\eta_{(2)}$', r'$\eta_{(3)}$', r'$\eta_{(4)}$', r'$\eta_{(5)}$',
            r'$\mu_{M(1)}$', r'$\mu_{M(2)}$', r'$\mu_{M(3)}$', r'$\mu_{M(4)}$', r'$\mu_{M(5)}$']

# Print bar graph
plt.figure(figsize=(12,6))
bars = plt.bar(paramName, PRCCval)
plt.axhline(0, color="black", linewidth=1)
plt.xticks(fontsize=8)
plt.xlabel("Parameter")
plt.ylabel("PRCC")
for bar in bars[0:5]:
    bar.set_color("red")
bars[5].set_color("orange")
for bar in bars[11:16]:
    bar.set_color("purple")
for bar in bars[16:21]:
    bar.set_color("black")
for bar in bars[21:26]:
    bar.set_color("green")
plt.savefig("PRCC R0", dpi=1000, bbox_inches="tight")
plt.show()
