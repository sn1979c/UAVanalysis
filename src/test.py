from llt import LLTAnalysis as llt
import matplotlib.pyplot as plt

from scipy.interpolate import interpolate
import numpy as np
import pandas as pd
import math
import os
import sko.GA

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['font.size'] = 25
plt.rcParams['axes.unicode_minus'] = False

wing = llt('wing1_def.csv')
wing.compute_wing(0, velocity=12, altitude=500)

print(wing.aero_data)

plt.plot(wing.z, wing.Cl)
plt.show()
