import numpy as np
import pandas as pd


s_NLLTStations = 20
m_MatSize = 15
RFF = 10.0         # < factor used to determine if a point is at a far distance from the panel >*/

class panel:
    def __init__(self):
        self.Aera = 0


        self.m_bIsLeading     = False
        self.m_bIsTrailing    = False
        self.m_bIsInSymPlane  = False
        self.m_bIsLeftPanel   = False
        self.m_bIsWakePanel   = False

        self.m_iLA = -1                 #< index of the leading left node in the node array
        self.m_iLB = -1                 #< index of the leading right node in the node array
        self.m_iTA = -1                 #< index of the trailing left node in the node array
        self.m_iTB = -1                 #< index of the trailing right node in the node array

        self.m_iElement = -1                 #< panel identification number ; used when the panel array is re-arranged in non sequential order to reduce the matrix size in symetrical calculations */
        self.m_iPL = -1                      #< index of the panel which lies left of this panel, or -1 if none */
        self.m_iPR = -1                      #< index of the panel which lies right of this panel, or -1 if none */
        self.m_iPU = -1                      #< index of the panel which lies upstream of this panel, or -1 if none */
        self.m_iPD = -1                      #< index of the panel which lies downstream of this panel, or -1 if none */
        self.m_iWake = -1                    #< -1 if not followed by a wake panel, else equal to wake panel number */
        self.m_iWakeColumn = -1              #< index of the wake column shed by this panel, numbered from left tip to right tip, or -1 if none */

        self.Normal = np.zeros(3)            #< the unit vector normal to the panel
        self.CtrlPt = np.zeros(3)            #< the position of the control point for VLM analysis or 3D/Thin panels analysis
        self.CollPt = np.zeros(3)            #< the collocation point for 3d panel analysis
        self.VA = np.zeros(3)                #< the left end point of the bound quarter-chord vortex on this panel
        self.VB = np.zeros(3)                #< the rightt end point of the bound quarter-chord vortex on this panel

    def sourceNASA4023(C, V, phi):
        # Evaluates the influence of a uniform source at a point outside the panel.
        # 
        # Follows the method provided in the VSAERO theory Manual NASA 4023.
        # 
        # Vectorial operations are written inline to save computing times -->longer code, but 4x more efficient.
        # 
        # @param C the point where the influence is to be evaluated
        # @param V the perturbation velocity at point C
        # @param phi the potential at point C
        
        pass

    def doubletNASA4023(self, C: np.ndarray, V, phi, bWake):
        # Evaluates the influence of a doublet at a point outside the panel.
        # 
        # Follows the method provided in the VSAERO theory Manual NASA 4023.
        # 
        # Vectorial operations are written inline to save computing times -->longer code, but 4x more efficient.
        # 
        # @param C the point where the influence is to be evaluated
        # @param V the perturbation velocity at point C
        # @param phi the potential at point C
        # @param bWake true if the panel is a wake panel, false if it is a surface panel
        PJK = C - self.CollPt
        PN = (PJK * self.Normal).sum()
        pjk = (PJK ** 0.5).sum()

        if(pjk> RFF*Size):
            # // use far-field formula
            phi = PN * self.Area / (pjk**3)
            T1 = PJK * 3.0 * PN - self.Normal*(pjk**2)
            V   = T1 * self.Area /(pjk ** 5)


            return



        pass

    def setPanelFrame(self):
        smp = np.zeros(3)
        smq = np.zeros(3)
        TALB = np.zeros(3)
        LATB = np.zeros(3)
        MidA = np.zeros(3)
        MidB = np.zeros(3)

        if(abs(self.LA[1])<1.e-5 & abs(self.TA[1])<1.e-5 & abs(self.LB[1])<1.e-5 & abs(self.TB[1])<1.e-5):
            m_bIsInSymPlane = True
        else:
            m_bIsInSymPlane = False


class PanelAnalysis:
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

        self.bThinSurfaces = True
        self.m_aij      = None
        self.m_aijWake  = None
        self.m_uRHS  = None
        self.m_vRHS  = None
        self.m_wRHS  = None
        self.m_pRHS  = None
        self.m_qRHS  = None
        self.m_rRHS  = None
        self.m_cRHS  = None
        self.m_uWake = None
        self.m_wWake = None
        self.m_uVl = None
        self.m_wVl = None

    def read_wing_settings(self, wing_name):
        pass

    def read_aero_data(self):
        pass

    def alphaLoop(self):
        self.buildInfluenceMatrix()
        self.createUnitRHS()

        if self.bThinSurfaces:
            self.createWakeContribution()

        for p in range(m_MatSize):
            self.m_uRHS[p]+= self.m_uWake[p]
            self.m_wRHS[p]+= self.m_wWake[p]
            for pp in range(m_MatSize):
                self.m_aij[p*m_MatSize+pp] += self.m_aijWake[p*m_MatSize+pp]

        self.solveUnitRHS()
        self.createSourceStrength()
        self.createDoubletStrength()
        self.computeFarField()
        self.computeAeroCoefs()

    def buildInfluenceMatrix(self):
        pass

    def createSourceStrength(self):
        pass

    def createRHS(self):
        pass

    def createUnitRHS(self):
        pass

    def createWakeContribution(self):
        pass

    def computeFarField(self):
        pass

    def computeAeroCoefs(self):
        pass

    def getDoubletInfluence(self):
        pass

    def getSourceInfluence(self):
        pass

    def createDoubletStrength(self):
        pass

    def solveUnitRHS(self):
        pass
    
    @staticmethod
    def VLMGetVortexInfluence(c, pPanel: panel, V, phi, bwake, bAll):
        # Returns the perturbation velocity created at a point C by a horseshoe or quad vortex with unit circulation located on a panel pPanel
        # @param pPanel a pointer to the Panel where the vortex is located
        # @param C the point where the perrturbation is evaluated
        # @param V a reference to the resulting perturbation velocity vector
        # @param bAll true if the influence of the bound vector should be included. Not necessary in the case of a far-field evaluation.


        return



