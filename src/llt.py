from ast import Raise
import numpy as np
import numpy.linalg as lg
import pandas as pd
import matplotlib.pyplot as plt
import os.path
from scipy.linalg import solve, det
from scipy import interpolate
from scipy import integrate
import time
import sko.GA
# s_NLLTStations + 1 points(including end point)
s_NLLTStations = 20
s_RelaxMax = 8
s_IterLim = 100


class LLTAnalysis:
    def __init__(self, wing_name):
        self.Ai = np.zeros(s_NLLTStations)
        self.upwash = np.zeros(s_NLLTStations)
        self.Cl = np.zeros(s_NLLTStations)
        self.Cd = np.zeros(s_NLLTStations)
        self.Cd_total = 0
        self.Cd_foil_pressure = np.zeros(s_NLLTStations)
        self.Cd_foil_viscous = np.zeros(s_NLLTStations)
        self.Cd_foil = np.zeros(s_NLLTStations)
        self.Cdi = np.zeros(s_NLLTStations)
        self.Cm = np.zeros(s_NLLTStations)
        self.beta = np.zeros((s_NLLTStations, s_NLLTStations))
        for m in range(1, s_NLLTStations):
            for k in range(1, s_NLLTStations):
                self.beta[m, k] = self.__beta(m, k)
        self.Cm_airfoil = np.zeros(s_NLLTStations)
        self.liftingline = np.zeros(s_NLLTStations)
        self.Cd_up = np.zeros(s_NLLTStations)
        self.aero_data = {'CL': 0, 'CD': 0, 'CDi': 0, 'CDp': 0, 'Cm': 0}

        self.twist = np.zeros(s_NLLTStations)
        self.chord = np.zeros(s_NLLTStations)
        self.offset = np.zeros(s_NLLTStations)
        self.diheral = np.zeros(s_NLLTStations)
        self.theta = np.linspace(0, np.pi, s_NLLTStations, endpoint=False)
        self.z = np.zeros(s_NLLTStations)
        self.section = pd.DataFrame()
        self.foil_names_re = {}
        self.foil_data_set = {}
        self.wing_data_set = {}
        self.span = 0
        self.atmos = {'rho': 0, 'v': 0, 'nu': 0}
        self.area = 0

        self.read_wing_settings(wing_name)
        self.read_aero_data()
        self.re = np.zeros(s_NLLTStations)

    def llt_upwash(self, x, y, z):
        rho = self.atmos['rho']
        v_inf = self.atmos['v']
        theta = list(self.theta)
        cl = self.Cl

        mu = 1.5e-5

        rc = 0.58*((x-1)/30)**0.5

        an = LLTAnalysis.fft_interpolate(theta, cl*0.5*v_inf*self.chord, 5)

        theta = np.linspace(np.pi/180, np.pi - np.pi/180, 300, endpoint=True)
        zs = self.span / 2 * np.cos(theta)
        gamma_zs = np.zeros(len(theta))
        gamma_cal = LLTAnalysis.fft_cal(theta, an)

        for i in range(len(theta)):
            for j in range(len(an)):
                gamma_zs[i] += an[j] * (j+1) * np.cos((j+1) * theta[i])
            gamma_zs[i] *= 2 / self.span / np.sin(theta[i])
        w = np.zeros(len(z))

        for i in range(len(z)):
            dw = gamma_zs * (zs - z[i]) / (4*np.pi*(y**2 + (zs - z[i])**2 + rc**2)
                                         ) * (x/(x**2 + y**2 + (z[i] - zs)**2)**0.5 + 1)
            w[i] = integrate.trapz(dw, zs)
        return w

    def iter_nonlinear_solution(self, Alpha, edgePoint='estimate'):
        s_CvPrec = 0.01
        iter = 0

        while iter < s_IterLim:
            m_Maxa = 0.0
            # plt.plot(self.theta[1:s_NLLTStations], self.Cl[1:s_NLLTStations])
            time1 = 0
            time2 = 0
            # time1 = time.time()
            for i in range(s_NLLTStations):
                self.Cl[i] = self.get_aero_data(self.z[i], self.re[i], Alpha + self.Ai[i] + self.twist[i])[0]
                # self.Cl[i] = 2*np.pi*(Alpha + self.Ai[i] + self.twist[i] - alpha_L0) / 180 * np.pi
            # time1 = time.time() - time1
            # time2 = time.time()
            for k in range(1, s_NLLTStations):
                a = self.Ai[k]
                anext = - np.sum(self.beta[1:, k] * self.Cl[1:] * self.chord[1:] / self.span)
                self.Ai[k] = a + (anext-a) / s_RelaxMax
                m_Maxa = np.max([m_Maxa, np.abs(a-anext)])
            # calculate explicitly

            # estimate edge point
            if edgePoint == 'estimate':
                self.Ai[0] = - self.get_aero_data(self.z[0], self.re[0], Alpha + self.twist[0])[0]  / (2 * np.pi) * 180 / np.pi * 0.5
                self.Cl[0] = 0
            else:
                while abs(self.Cl[0]) > 1e-5:
                    self.Ai[0] -= self.Cl[0] / (2 * np.pi) * 180 / np.pi * 0.5
                    self.Cl[0] = - self.get_aero_data(self.z[0], self.re[0], Alpha + self.Ai[0] + self.twist[0])[0]

            if (m_Maxa < s_CvPrec):
                break

            if (m_Maxa > 1e+3):
                print("divergence")
                break
            iter = iter + 1
        # print('iteration converge after {:d} steps'.format(iter))
        # for k in range(1, s_NLLTStations):
        #     print('k = {:d}, Cl = {:.5f}, Ai = {:.5f}'.format(k, self.Cl[k], self.Ai[k]))
        return iter

    def iter_twist(self, theta, cl):
        an = self.fft_interpolate(theta, cl, 30)
        # plt.plot(theta, cl)
        # plt.plot(np.linspace(0, np.pi, 100), self.fft_cal(np.linspace(0, np.pi, 100), an))
        # plt.show()
        delta_cl = np.zeros(s_NLLTStations)
        delta_cl[1:] = self.fft_cal(self.theta[1:], an) - self.Cl[1:]
        delta_alpha = delta_cl / (1 * np.pi) *180 / np.pi
        self.twist = self.twist + delta_alpha
        # plt.show()
        delta_cl[-3:] *= 0.5
        delta_cl[:2] *= 0.5
        return np.mean(abs(delta_cl[1:]))

    @staticmethod
    def compare_2_line(x1, y1, x2, y2, level):
        f = interpolate.interp1d(x1, y1, kind="linear", fill_value='extrapolate')
        # an = LLTAnalysis.fft_interpolate(x1, y1, 20)
        y2_fit = f(x2)
        distance = np.sum((y2 - y2_fit) ** 2) ** 0.5 / len(y2)
        return distance

    # x \in [0, pi]
    @staticmethod
    def fft_interpolate(x, y, number_of_an):
        Cof = np.ones((len(x), number_of_an))
        for position in range(len(x)):
            for n in range(number_of_an):
                Cof[position, n] = np.sin((n+1) * x[position])
        an = lg.solve(Cof.T.dot(Cof), Cof.T.dot(y))
        return an

    @staticmethod
    def fft_cal(x, an):
        Cof = np.ones((len(x), len(an)))
        for position in range(len(x)):
            for n in range(len(an)):
                Cof[position, n] =  np.sin((n+1) * x[position])
        y = Cof.dot(an)
        return y

    def alpha_induced(self, k):
        ai = np.sum(self.beta[1:, k] * self.Cl[1:] * self.chord[1:] / self.span)
        # ai = 0
        # for m in range(1, s_NLLTStations):
        #     ai += self.beta[m, k] * self.Cl[m] * self.chord[m] / self.span
        #     # ai += self.beta[m, k] * self.Cl[m] * self.chord[m] / self.span
        return ai

    def add_twist(self, theta, twi):
        f = interpolate.interp1d(theta, twi, kind="linear")
        self.twist[1:s_NLLTStations] = self.twist[1:s_NLLTStations] + f(self.theta[1:s_NLLTStations])
        return self.twist

    @staticmethod
    def __beta(m, k):
        r = s_NLLTStations
        beta_mk = 0
        for n in range(1, r):
            beta_mk += 1 / (4 * r * np.sin(k / r * np.pi)) * n * (
                        np.cos(n * (k - m) / r * np.pi) - np.cos(n * (k + m) / r * np.pi))
        return beta_mk * 180 / np.pi

    def plot(self):
        theta = np.linspace(1, s_NLLTStations-1, s_NLLTStations-1) / (s_NLLTStations) * np.pi
        plt.plot(theta, self.Cl[1:s_NLLTStations], label='Ai')
        plt.xlabel('theta')
        plt.ylabel('Cl')
        plt.show()

        plt.plot(theta, self.Ai[1:s_NLLTStations], label='Ai')
        plt.xlabel('theta')
        plt.ylabel('Ai')
        plt.show()

    def read_aero_data(self):
        file_names = os.listdir('./src/foil_data')
        # print(file_names)
        for name in file_names:
            foil_data = pd.read_csv('./src/foil_data/' + name)
            foil_name, re = name[:-4].split(sep='-')
            # print(self.foil_names_re)
            re = float(re[:-1]) * 10000
            if foil_name in list(self.foil_names_re.keys()):
                self.foil_names_re[foil_name].append(re)
            else:
                self.foil_names_re[foil_name] = [re]
            self.foil_data_set[name[:-4]] = foil_data

    # foil0 比截面坐标小的翼型名
    # foil1 比截面坐标大的翼型名
    # tau = d1 / (d1+ d2)
    #           d1      d2
    # ----foil0----foil----foil1----> X
    def get_aero_data(self, z, re, alpha):
        # time1 = time.time()
        foil0_index = self.section[self.section['z'] <= z].index[-1]
        foil1_index = self.section[self.section['z'] >= z].index[0]
        # time1 = time.time() - time1
        # time2 = time.time()
        foil0_data = self.get_foil_from_alpha(self.section.loc[foil0_index, 'foil'], re, alpha)
        foil1_data = self.get_foil_from_alpha(self.section.loc[foil1_index, 'foil'], re, alpha)
        # time2 = time.time() - time2
        # print('time1={},time2={}'.format(time1, time2))
        z0 = self.section.loc[foil0_index, 'z']
        z1 = self.section.loc[foil1_index, 'z']
        if z == z0:
            data = foil0_data
        elif z == z1:
            data = foil1_data
        else:
            data = (foil1_data - foil0_data) / (z1 - z0) * (z - z0) + foil0_data
        return data

    def get_foil_from_alpha(self, foil, re, alpha):
        # interpolate aero_data from re and alpha data
        # input 
        # foil: foil_name
        # re:Reynolds number to be interpolated, 
        # alpha:Reynolds number to be interpolated
        # Return:
        # np.array([cl, cd, cm])
        foil_re = self.foil_names_re[foil].copy()
        foil_re.append(re)
        foil_re.sort()
        index = foil_re.index(re)

        if (index == 0) | (index == len(foil_re)-1):
            raise ValueError('Re is out of Bounds, calculate more re points')
        else:
            data_left = self.foil_data_set[foil + '-' + str(int(foil_re[index-1]/10000))+'w']
            data_right = self.foil_data_set[foil + '-' + str(int(foil_re[index+1]/10000))+'w']
            deg_left = np.array(data_left['deg'])
            deg_right = np.array(data_right['deg'])

            f1 = interpolate.interp1d(deg_left, np.array(data_left['cl']))
            f2 = interpolate.interp1d(deg_left, np.array(data_left['cd']))
            f3 = interpolate.interp1d(deg_left, np.array(data_left['cm']))
            f4 = interpolate.interp1d(deg_right, np.array(data_right['cl']))
            f5 = interpolate.interp1d(deg_right, np.array(data_right['cd']))
            f6 = interpolate.interp1d(deg_right, np.array(data_right['cm']))
            try:
                foil_re_left = np.array([f1(alpha), f2(alpha), f3(alpha)])
                foil_re_right = np.array([f4(alpha), f5(alpha), f6(alpha)])
                foil_data = foil_re_left + (foil_re_right - foil_re_left) / (
                        foil_re[index+1] - foil_re[index-1]) * (re - foil_re[index-1])
                return foil_data
            except ValueError as e:
                print(deg_left)
                print(deg_right)
                raise ValueError('alpha={:.2f} is out of bounds'.format(alpha))


    def read_wing_settings(self, wing_name):
        wing = pd.read_csv('.\\src\\wing_def\\{}'.format(wing_name))
        index = len(wing)
        wing = wing.append(wing[wing['z'] > 0], ignore_index=True)
        wing.loc[index:, 'z'] = - wing.loc[index:, 'z']
        wing = wing.sort_values(by='z').reset_index(drop=True)
        self.section = wing
        f1 = interpolate.interp1d(wing['z'], wing['chord'])
        f2 = interpolate.interp1d(wing['z'], wing['offset'])
        f3 = interpolate.interp1d(wing['z'], wing['diheral'])
        f4 = interpolate.interp1d(wing['z'], wing['twist'])
        self.span = wing['z'].max() * 2
        self.z = np.cos(self.theta) * self.span / 2
        self.chord = f1(self.z)
        self.offset = f2(self.z)
        self.diheral = f3(self.z)
        self.twist = f4(self.z)

    def compute_wing(self, alpha, velocity=30, rho_inf=1.167362, nu=1.7849e-5, altitude=500):
        start = time.time()
        self.atmos['v'] = velocity
        if altitude == 500:
            self.atmos['rho'] = rho_inf
            self.atmos['nu'] = nu
        else:
            a = self.get_atoms(altitude)
            self.atmos['rho'] = a['density']
            self.atmos['nu'] = a['dynamicViscosity']
        self.re = self.atmos['rho'] * self.atmos['v'] * self.chord / self.atmos['nu']
        print(self.re)
        # main_start = time.time()
        # self.set_linear_solution(alpha)
        self.iter_nonlinear_solution(alpha)
        # main_end = time.time()
        # main_consume = str(main_end - main_start)
        self.Cd_foil_pressure = np.zeros(s_NLLTStations)
        self.Cd_foil_viscous = np.zeros(s_NLLTStations)
        for i in range(s_NLLTStations):
            _, self.Cd_foil[i], self.Cm_airfoil[i]= self.get_aero_data(
                self.z[i], self.re[i], alpha+self.twist[i]+self.Ai[i])

        self.Cdi = - self.Cl * np.sin(self.Ai * np.pi / 180)
        self.Cd = self.Cd_foil + self.Cdi
        self.Cd_up = - self.Cl * np.sin(self.upwash * np.pi / 180)

        CL = -(integrate.trapz(self.Cl[1:], self.z[1:]) + integrate.trapz(self.Cl[:2], self.z[:2]) +
               integrate.trapz([self.Cl[0], self.Cl[-1]], self.z[:2]))

        CDi = - (integrate.trapz(self.Cdi[1:], self.z[1:]) + integrate.trapz(self.Cdi[:2], self.z[:2]) +
                 integrate.trapz([self.Cdi[0], self.Cdi[-1]], self.z[:2]))

        CDp = - (integrate.trapz(self.Cd_foil[1:], self.z[1:]) + integrate.trapz(self.Cd_foil[:2], self.z[:2]) +
                 integrate.trapz([self.Cd_foil[0], self.Cd_foil[-1]], self.z[:2]))

        CL /= self.span
        CDi /= self.span
        CDp /= self.span
        CD = CDi + CDp
        cm = self.Cm_airfoil - self.Cl * (self.chord * 0.25 + self.offset) / self.chord
        Cm = - (integrate.trapz(cm[1:], self.z[1:]) + integrate.trapz(cm[:2], self.z[:2]) * 2)
        Cm /= self.span

        self.aero_data['CL'] = CL
        self.aero_data['CD'] = CD
        self.aero_data['CDi'] = CDi
        self.aero_data['CDp'] = CDp
        self.aero_data['Cm'] = Cm
        # print(self.aero_data['CL'], self.aero_data['CL'] / self.aero_data['CD'])

    @staticmethod
    def get_atoms(altitude):
        universal_gas_constant = 8.3144621
        dry_air_molar_mass = 0.02896442
        adiabatic_index = 1.4
        sutherlands_constant = 120
        reference_viscosity = 17.894e-6
        standard_temperature = 288.15
        standard_gravity = 9.80665
        standard_pressure = 101325
        standard_lapse_rate = 0.0065
        # <= 11000 m
        temperature = standard_temperature - altitude * standard_lapse_rate
        preesure = standard_pressure * (temperature / standard_temperature)**(
                standard_gravity * dry_air_molar_mass / universal_gas_constant / standard_lapse_rate)
        density = preesure * dry_air_molar_mass / universal_gas_constant / temperature
        dynamicViscosity = reference_viscosity * (standard_temperature + sutherlands_constant) / (
                temperature + sutherlands_constant) * (temperature / standard_temperature) ** 1.5
        speedOfSound = (adiabatic_index * universal_gas_constant * temperature / dry_air_molar_mass) ** 0.5
        return {'temperature': temperature, 'pressure': preesure, 'density': density,
                'dynamicViscosity': dynamicViscosity, 'speedOfSound': speedOfSound}
