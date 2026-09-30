import matplotlib.pyplot as plt
import numpy as np
import os

PRCCval = [-0.072699, -0.021146, -0.040607, -0.019448, -0.044611, 0.682508,
           0.010414, 0.006853, 0.007738, 0.005178, 0.007128,
           0.403359, 0.166652, 0.286542, 0.148510, 0.305841,
           0.288789, 0.122991, 0.207120, 0.106014, 0.219335,
           -0.241973, -0.118426, -0.187917, -0.101038, -0.197285]  

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
plt.savefig("PRCC R0", dpi=150, bbox_inches="tight")
plt.show()
