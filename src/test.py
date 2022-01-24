from llt import LLTAnalysis as llt
import matplotlib.pyplot as plt

# from scipy.interpolate import interpolate
import numpy as np
import pandas as pd
from xfoil import XFoil

def cal_naca(naca_code, re):
    foil = XFoil()
    foil.print = False
    f = foil.naca(naca_code)
    foil.repanel()
    foil.max_iter = 100
    foil.Re = re
    deg, cl, cd, cm, _ = foil.aseq(-6, 10, 0.5)
    dat = pd.DataFrame({'deg':deg, 'cl':cl, 'cd':cd, 'cm':cm})
    dat = dat.dropna()
    return dat


# cal foil data
f1 = cal_naca('6412', 20e4)
f1.to_csv('src/foil_data/foil1-20w.csv', float_format='%.5f', sep=',', index=False)
f2 = cal_naca('6414', 20e4)
f2.to_csv('src/foil_data/foil2-20w.csv', float_format='%.5f', sep=',', index=False)
f3 = cal_naca('6412', 30e4)
f3.to_csv('src/foil_data/foil1-30w.csv', float_format='%.5f', sep=',', index=False)
f4 = cal_naca('6414', 30e4)
f4.to_csv('src/foil_data/foil2-30w.csv', float_format='%.5f', sep=',', index=False)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['font.size'] = 25
plt.rcParams['axes.unicode_minus'] = False

wing = llt('wing1_def.csv')
wing.compute_wing(0, velocity=12, altitude=500)

plt.plot(wing.z, wing.Cl)
plt.show()
