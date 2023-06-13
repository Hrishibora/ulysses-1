###################################################################################
#                                                                                 #
#                    Primordial Black Hole induced leptogenesis.                  #
#                  High Scale Scenario, including DL = 2 scattering               #
#  Including Superradiant production of an scalar which decays into RH neutrinos  #
#                                                                                 #
###################################################################################

import ulysses
import numpy as np
from odeintw import odeintw
from scipy import interpolate
import scipy.integrate as integrate
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta
from numba import njit

import ulysses.BHProp as bh #Schwarzschild and Kerr BHs library

from numpy import sqrt, log, exp, log10, pi, arctan

from termcolor import colored

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                                                     FLRW-Boltzmann Equations                                                       #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#----------------------------------#
#   Equations before evaporation   #
#----------------------------------#

def FBEqs(x, v, ngam, M1, M2, M3, Mphi, eps, d1, w1, N1Req, dPBH1_prim, dPBH1_sec, dSR1, WashDL2, xilog10, GSR, Gphi, pphi, BR_RH):

    M     = v[0] # PBH mass in g
    ast   = v[1] # PBH angular momentum
    rRAD  = v[2] # Radiation energy density
    rPBH  = v[3] # PBH       energy density
    Tp    = v[4] # Plasma Temperature
    
    N1RT  = v[5] # nu_R thermal number density
    N1RBp = v[6] # nu_R number density from PBH evaporation -- primary
    N1RBs = v[7] # nu_R number density from PBH evaporation -- secondary
    N1RS  = v[8] # nu_R number density from Superradiance

    NphiB = v[9]  # Scalar number of particles from evaporation
    NphiS = v[10] # Scalar number of particles from Superradiance per BH
    
    NBL   = v[11] # single Flavor B-L asymmetry

    #--------------------------------------------------------------------------------#
    #                                     Parameters                                 #
    #--------------------------------------------------------------------------------#
    
    M_GeV = M/bh.GeV_in_g            # PBH mass in GeV
    nPBH  = rPBH/M_GeV               # PBH number density in GeV^3
    MPL_r = bh.mPL/np.sqrt(8.*np.pi) # Reduced Planck Mass

    TBH = bh.TBH(M, ast)       # Black Hole Temperature
     
    xff = x + xilog10          # Logarithm_10 of scale factor, xilog is the log10 of the initial scale factor

    a = 10.**xff               # Scale Factor

    eps1tt, eps1mm, eps1ee, eps1tm, eps1te, eps1me = eps  # CP violation elements

    # Evaporation functions for Black Hole mass and spin
    
    FSM  = bh.fSM(M, ast) + bh.gg * bh.phi_g(M, ast, 0.)  # SM + gravitons Evaporation contribution
    FRH1 = bh.fRH(M, ast, M1)                             # 1 RH neutrino  Evaporation contribution
    FRH2 = bh.fRH(M, ast, M2)                             # 2 RH neutrino  Evaporation contribution
    FRH3 = bh.fRH(M, ast, M3)                             # 3 RH neutrino  Evaporation contribution
    FPhi = bh.fDM(M, ast, Mphi, 0)                        # Scalar contribution
    FT   = FSM + FRH1 + FRH2 + FRH3 + FPhi                # Total Evaporation contribution

    GSM  = bh.gSM(M, ast) + bh.gg * bh.gam_g(M, ast, 0.)  # SM + gravitons Evaporation contribution
    GRH1 = bh.gRH(M, ast, M1)                             # 1 RH neutrino  Evaporation contribution
    GRH2 = bh.gRH(M, ast, M2)                             # 2 RH neutrino  Evaporation contribution
    GRH3 = bh.gRH(M, ast, M3)                             # 3 RH neutrino  Evaporation contribution
    GPhi = bh.gDM(M, ast, Mphi, 0)                        # Scalar contribution
    GT   = GSM + GRH1 + GRH2 + GRH3 + GPhi                # Total Evaporation contribution

    # Hubble parameter
    
    H = np.sqrt(8 * np.pi * bh.GCF * (rPBH*a**(-3) + rRAD*a**(-4))/3.)
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp)) # Temperature parameter


    # Thermally averaged phi decay width wrt BH temperature
    BR_SM = 1. - BR_RH

    zphi = Mphi/TBH

    from ulysses.ulsbase import my_kn2, my_kn1
    GphiBH = min([Gphi * my_kn1(zphi)/my_kn2(zphi), 1e5*H])
    pphi /= a  # Average momentum of phi particles from evaporation, a factor is included due to redshift
    Ephi  = sqrt(Mphi**2  + pphi**2)
    
    rphiBH = Ephi*NphiB*ngam # Phi Energy density from evaporation
    
    #----------------------------------------------#
    #    Radiation + PBH + Temperature equations   #
    #----------------------------------------------#

    alpha = bh.GCF * M_GeV * Mphi                # Gravitational coupling between scalar and PBH

    # Black Hole Evolution
    dMrad_GeVdx = - FT/(bh.GCF**2 * M_GeV**2)/H  # Only Hawking production -> Contribution to the SM radiation density

    dM_GeVdx    = (- FT/(bh.GCF**2 * M_GeV**2) - Mphi*GSR*NphiS)/H  
    dastdx      = (- ast * (GT - 2.*FT)/(bh.GCF**2 * M_GeV**3)
                   - 8.*np.pi*(np.sqrt(2.) - 2.*alpha*ast)*(MPL_r/M_GeV)**2*GSR*NphiS)/H
    
    # Universe Evolution
    drRADdx  = - (FSM/FT) * (dMrad_GeVdx/M_GeV) * a * rPBH  + 2.*(BR_SM*GphiBH/H) * a * rphiBH + 2.*(BR_SM*Gphi/H) * a * (Mphi*NphiS*nPBH)
    # Additional term includes scalar decay from evaporation and SR into SM radiation
    drPBHdx  = + (dM_GeVdx/M_GeV) * rPBH
    dTdx     = - (Tp/Del) * (1.0 - (bh.gstarS(Tp)/bh.gstar(Tp))*(drRADdx/(4.*rRAD)))

    #----------------------------------------#
    #       RH neutrinos and Scalar          #
    #----------------------------------------#

    d1t = min([d1, 1.e4*H])
    dPBH1_primt = min([dPBH1_prim, 1.e4*H])
    dPBH1_sect  = min([dPBH1_sec, 1.e4*H])
    dSR1t  = min([dSR1, 1.e4*H])
    
    NTH1  = (N1RT - N1Req) * d1t/H
    NBH1p = N1RBp * dPBH1_primt/H  # dPBH_prim corresponds to the direct emission from the PBH 
    NBH1s = N1RBs * dPBH1_sect/H   # dPBH_sec is from the decay of scalars produced by the evaporation
    NSR1  = N1RS  * dSR1t/H

    dN1RTdx  = -NTH1                                           # Thermal contribution
    dN1RBpdx = -NBH1p + (bh.Gamma_F(M, ast, M1)/H)*(nPBH/ngam) # Primary PBH-induced contribution, normalized wrt the initial photon density ngam
    dN1RBsdx = -NBH1s + 2.*(BR_RH * GphiBH/H)*NphiB            # Secondary PBH-induced contribution, normalized wrt the initial photon density ngam
    dN1RSdx  = -NSR1  + 2.*(BR_RH * Gphi/H)*NphiS*(nPBH/ngam)  # RH neutrinos from scalar decay  from SR

    dNphiBdx = -NphiB*GphiBH/H + (2*bh.Gamma_S(M, ast, Mphi)/H)*(nPBH/ngam) # Scalar number of particles from Evaporation
    dNphiSdx = (GSR - Gphi)*NphiS/H                                         # Scalar number of particles from Superradiance per PBH

    #----------------------------------------#
    #            Lepton asymmetries          #
    #----------------------------------------#

    dNBLdx = (eps1tt+eps1mm+eps1ee)*(NTH1 + NBH1p + NBH1s + NSR1) - (w1 + WashDL2)*NBL/H
  
    #Equations

    kappa = bh.GeV_in_g # Conversion factor to have Mass equation rate for PBH mass in g
    
    dEqsdx = [kappa * dM_GeVdx, dastdx, drRADdx, drPBHdx, dTdx, dN1RTdx, dN1RBpdx, dN1RBsdx, dN1RSdx, dNphiBdx, dNphiSdx, dNBLdx]

    return [xeq * np.log(10.) for xeq in dEqsdx]

#----------------------------------#
#    Equations after evaporation   #
#----------------------------------#

def FBEqs_aBE(x, v, ngam, M1,M2,M3, Mphi, eps, d1, w1, N1Req, nPBHi, dPBH1_prim, dPBH1_sec, dSR1, WashDL2, Gphi, pphi, BR_RH, x_ev, TBHi):

    rRAD  = v[0] # Radiation energy density
    Tp    = v[1] # Plasma Temperature
    
    N1RT  = v[2] # nu_R therma number density
    N1RBp = v[3] # nu_R number density from PBH evaporation -- primary
    N1RBs = v[4] # nu_R number density from PBH evaporation -- secondary
    N1RS  = v[5] # nu_R number density from Superradiance

    NphiB = v[6]  # Scalar number of particles from evaporation
    NphiS = v[7] # Scalar number density from Superradiance
    
    NBL   = v[8] # single Flavor B-L asymmetry

    #----------------#
    #   Parameters   #
    #----------------#

    a = 10.**x   # Scale factor

    eps1tt, eps1mm, eps1ee, eps1tm, eps1te, eps1me = eps # CP violation elements

    H   = np.sqrt(8 * np.pi * bh.GCF * ((N1RT + N1RBp + N1RBs)*M1 * a**(-3) + rRAD * a**(-4))/3.)  # Hubble parameter
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp))                                         # Temperature parameter

    BR_SM = 1. - BR_RH

    zphi = Mphi/TBHi

    from ulysses.ulsbase import my_kn2, my_kn1
    
    pphi *= 10.**(x_ev - x)             # Average momentum of phi particles from evaporation, factor included due to redshift
    Ephi  = sqrt(Mphi**2  + pphi**2)

    if NphiB >= 0.: GphiBH = min([Gphi * Mphi/Ephi, 1e3*H]) # boosted phi width (saturated at 10^3 H in order to avoid stiffness)
    else: GphiBH = 0.
    
    rphiBH = Ephi*NphiB*ngam # Mediator Energy density

    #----------------------------------------#
    #    Radiation + Temperature equations   #
    #----------------------------------------#
    
    drRADdx = 2.*(BR_SM*GphiBH/H) * a * rphiBH + 2.*(BR_SM*Gphi/H)* a * (Mphi*NphiS*nPBHi) # Term includes scalar decay into SM radiation
    dTdx    = - Tp/Del

    #----------------------------------------#
    #              RH neutrinos              #
    #----------------------------------------#

    NTH1  = (N1RT - N1Req) * d1/H
    NBH1p = N1RBp * dPBH1_prim/H   # dPBH_prim corresponds to the direct emission from the PBH while dPBH_sec is from the decay of scalars produced by the evaporation
    NBH1s = N1RBs * dPBH1_sec/H    # dPBH_prim corresponds to the direct emission from the PBH while dPBH_sec is from the decay of scalars produced by the evaporation
    NSR1  = N1RS  * dSR1/H

    dN1RTdx  = -NTH1                                             # Thermal contribution
    dN1RBpdx = -NBH1p                                            # Primary PBH-induced contribution, normalized wrt the initial photon density ngam
    dN1RBsdx = -NBH1s + 2.*(BR_RH * GphiBH/H)*NphiB*(nPBHi/ngam) # Secondary PBH-induced contribution, normalized wrt the initial photon density ngam
    dN1RSdx  = -NSR1  + 2.*(BR_RH * Gphi/H)*NphiS*(nPBHi/ngam)   # RH neutrinos from scalar decay  

    dNphiBdx = - NphiB*GphiBH/H     # Scalar number of particles from Evaporation
    dNphiSdx = - NphiS*Gphi/H       # Scalar decay into RHNs

    #----------------------------------------#
    #            Lepton asymmetries          #
    #----------------------------------------#

    dNBLdx = ((eps1tt+eps1mm+eps1ee)* (NTH1 + NBH1p + NBH1s + NSR1) - (w1 + WashDL2)*NBL/H)
    #Equations
    
    dEqsdx = [drRADdx, dTdx, dN1RTdx, dN1RBpdx, dN1RBsdx, dN1RSdx, dNphiBdx, dNphiSdx, dNBLdx]

    dEqsdx = [x * np.log(10.) for x in dEqsdx]
    
    return dEqsdx

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                           ULYSSES class for PBH-Leptogenesis with Superradiance                            #     
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

class EtaB_PBH_SR(ulysses.ULSBase):
    """
    Primordial black hole Leptogenesis including Superradiant emission of a scalar decaying into RHNs.
    One-flavoured BE with 1 Right-handed Neutrino. Including the DL=2 washout term.
    See arXiv:2010.03565, arXiv:2203.08823 and arXiv:2205.11522
    """

    def shortname(self): return "1BE1F_PBHSR"

    def evolname(self): return "a"

    def flavourindices(self): return [1, 2]

    def flavourlabels(self): return ["$N^{\\rm B-L}_{\\rm TH}$", "$N^{\\rm B-L}_{\\rm PBH}$"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        This model requires five additional parameters, PBH mass and spin and PBH initial abundance, initial gravitational coupling between scalar and PBH and scalar coupling to RHNs
        """
        self.MPBHi = None # Log10[M/1g]
        self.aPBHi = None # Initial spin factor a_*
        self.bPBHi = None # Log10[beta']
        self.alphi = None # Alpha = G * MPBH * Mphi
        self.g     = None # Coupling g
        self.BR    = None # Branching ratio of phi -> NN

        self.pnames = ['m',  'M1', 'M2', 'M3', 'delta', 'a21', 'a31', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3',
                       't12', 't13', 't23', 'MPBHi', 'aPBHi', 'bPBHi', 'alphi', 'log_g', 'BR']

        #------------------------------------------------------------#
        #            Inverse Time dilatation factor <M/E>            #
        #        Averaged wrt the Hawking emission spectrum          #
        #------------------------------------------------------------#

        import os
        data_dir = os.path.dirname(ulysses.__file__)

        # Fermions
        MEav_f   = os.path.join(data_dir, "./data/timedil.txt")
        MEav_f_Tab  = np.loadtxt(MEav_f)
        self.MEav_f = interpolate.interp1d(MEav_f_Tab[:,0],  MEav_f_Tab[:,1], bounds_error=False, fill_value=(MEav_f_Tab[0,1],  MEav_f_Tab[-1,1]))

        # Scalar

        fav_s   = os.path.join(data_dir, "./data/fN1_av.txt")
        pav_s   = os.path.join(data_dir, "./data/pN1_av.txt")
        Eav_s   = os.path.join(data_dir, "./data/EN1_av.txt")
        MEav_s   = os.path.join(data_dir, "./data/timedil_s.txt")
        
        fav_s_Tab   = np.loadtxt(fav_s)
        pav_s_Tab   = np.loadtxt(pav_s)
        Eav_s_Tab   = np.loadtxt(Eav_s)
        MEav_s_Tab  = np.loadtxt(MEav_s)
        
        self.fav_s = interpolate.interp1d(fav_s_Tab[:,0],  fav_s_Tab[:,1], bounds_error=False, fill_value=(fav_s_Tab[0,1],  fav_s_Tab[-1,1]))
        self.pav_s = interpolate.interp1d(pav_s_Tab[:,0],  pav_s_Tab[:,1], bounds_error=False, fill_value=(pav_s_Tab[0,1],  pav_s_Tab[-1,1]))
        self.Eav_s = interpolate.interp1d(Eav_s_Tab[:,0],  Eav_s_Tab[:,1], bounds_error=False, fill_value=(Eav_s_Tab[0,1],  Eav_s_Tab[-1,1]))
        self.MEav_s = interpolate.interp1d(MEav_s_Tab[:,0],  MEav_s_Tab[:,1], bounds_error=False, fill_value=(MEav_s_Tab[0,1],  MEav_s_Tab[-1,1]))

        #-----------------------------------#
        #      Fitting tables of Gamma_sr   #
        #-----------------------------------#

        Nti  = 500-1
        Ntj1 = 250-1
        Ntj2 = 150
        
        mui = np.array([0.01 + (0.5 - 0.01)*i/Nti for i in range(Nti+1)])
        
        as1 = np.array([1.e-5 + (0.9 - 1.e-5)*j/Ntj1 for j in range(Ntj1+1)])
        as2 = np.array([1. - 10.**(-1. + (-3. + 1.)*j/Ntj2) for j in range(1, Ntj2+1)])
        asj = np.concatenate((as1,as2))
        
        test_Gsr_f = os.path.join(data_dir, "./data/tab_Gammasr_l=m=1_new.txt")
        test_Gsr   = np.loadtxt(test_Gsr_f)
        
        self.fG_SR_ = interpolate.RectBivariateSpline(mui, asj, test_Gsr, kx=1, ky=1, s=0.)
        
        
    def setParams(self, pdict):
        super().setParams(pdict)
        self.MPBHi = pdict["MPBHi"]
        self.aPBHi = pdict["aPBHi"]
        self.bPBHi = pdict["bPBHi"]
        self.alphi = pdict["alphi"]
        self.log_g = pdict["log_g"]
        self.BR    = pdict["BR"]

    def ME(self,zBH):

        LzBH = np.log10(zBH)
        
        if LzBH < -4.:
            return 10.**(-0.6267457 + 0.9999617*LzBH)
        elif LzBH >= -4. and LzBH <= 2.85:
            return self.MEav_f(LzBH)
        else:
            return 1.

    def ME_phi(self, zBH):

        LzBH = np.log10(zBH)
        
        if LzBH < -4.:
            return 10.**(-0.44893591 + LzBH)
        elif LzBH >= -4. and LzBH <= 2.85:
            return self.MEav_s(LzBH)
        else:
            return 1.

    def ME_N_phi(self, M1, Mphi, TBH):

        zphi = Mphi/TBH
        z1   = M1/TBH

        eta = np.sqrt(1. - 4.*M1*M1/(Mphi*Mphi))

        LzBH = np.log10(zphi)
        
        if LzBH < -4.:

            fN_0 = 2.45691
            EN_0 = 6.90756
            pN_0 = 6.90756
            
            MEn = z1*fN_0/(EN_0 + eta * pN_0)
            
        elif LzBH >= -4. and LzBH <= 2.85:

            MEn = z1*self.fav_s(zphi)/(self.Eav_s(zphi) + eta * self.pav_s(zphi))
            
        else:

            MEn = 0.

        return MEn
    
    def StopMass(self, t, v, Mi):
        '''
        Function to stop the solver if the BH is equal than 
        1% the initial mass  or the Planck mass
        '''
        eps = 0.01
        
        if (eps*Mi > bh.MPL): Mst = eps*Mi
        else: Mst = bh.MPL
        
        return v[0] - Mst

    def Gphi(self, Mphi, M1, g):
        '''
        Scalar decay width into RH neutrinos and SM fermions, in GeV
        Assuming massless SM dofs.
        '''

        term_RH = (1. - 4*M1*M1/(Mphi*Mphi))**(3/2)
        term_SM = 1.
        
        return (g*g*Mphi/(16.*np.pi))*(term_RH + term_SM) 

    def Gamma_SR_num(self, Mphi, M, ast, fGsr):
        '''
        Superradiance Growth and Decay rate, in GeV
        '''
        
        GM = bh.GCF * (M/bh.GeV_in_g) # in GeV^-1
        
        alpha = GM*Mphi
        
        Gsr =  fGsr(alpha, ast)[0,0]
        
        if ast > 1.: Gsr *= 0.
        
        return alpha*Gsr/GM # in GeV

    def p_average_phi(self, Mi, asi, Mphi, tau, Sol_t):
        '''
        Average momentum of phi from BH evaporation
        '''

        def Integ_p(t, pars):
            
            MDM, sol = pars
            
            M   = sol(t)[0]
            ast = sol(t)[1]
            
            sPhi = 0. # Phi spin
            
            return 10.**t * log(10.) * bh.fDM(M, ast, Mphi, sPhi)/M**2
        
        def Integ_n(t, pars):
            
            MDM, sol = pars
            
            M   = sol(t)[0]
            ast = sol(t)[1]
            
            return 10.**t * log(10.) * bh.Gamma_S(M, ast, Mphi)
        
        pars = [Mphi, Sol_t]
        
        integ_p = integrate.quad(Integ_p, -10., tau, args=(pars))
        integ_n = integrate.quad(Integ_n, -10., tau, args=(pars), epsabs=1.e-07, epsrel=1.e-07)

        if(integ_n[0]!=0):
            
            return (bh.kappa * integ_p[0]/bh.GeV_in_g)/integ_n[0]
        else:
            return 0
    
        
    #********************************************************#
    #        Equations Before  PBH evaporation               #
    #********************************************************#
    def RHS(self, x, y0, eps, ngam, xilog10, Mphi, g, fG_SR, pphi, BR): # x is the Log10 of the scale factor

        MBH  = y0[0]
        ast  = y0[1]
        Tp   = y0[4]                 # Plasma Temperature
        TBH  = bh.TBH(MBH, ast)  # BH temperature
        z    = self.M1/Tp
        zBH  = np.real(self.M1/TBH)
        zphi = Mphi/TBH

        from ulysses.ulsbase import my_kn2, my_kn1

        self._d1         = np.real(self.Gamma1 * my_kn1(z)/my_kn2(z))                # Therm-av RH decay width wrt to Plasma Temperature
        self._dPBH1_prim = np.real(self.Gamma1 * self.ME(zBH))                       # Therm-av RH decay width wrt to TBH, primary component
        self._dPBH1_sec  = np.real(self.Gamma1 * self.ME_N_phi(self.M1, Mphi, TBH))  # Therm-av RH decay width wrt to TBH, secondary component
        self._dSR1       = np.real(self.Gamma1 * (2.*self.M1/Mphi))                  # Boosted RHN decay width from the Superradiant cloud, assuming phi decaying at rest

        Gam_phi     = self.Gphi(Mphi, self.M1, g)

        self._w1    = self._d1 * (my_kn2(z) * z**2/(3. * zeta(3.)))
        
        # RH neutrino equilibrium number density, normalized to initial photon density
        self._n1eq  = (10**(3*(x + xilog10)) * self.M1**2 * Tp * my_kn2(z))/(np.pi**2)/ngam 
        
        # Neutrino masses squared
        m1sq = self.SqrtDm[0,0]**4
        m2sq = self.SqrtDm[1,1]**4
        m3sq = self.SqrtDm[2,2]**4

        # Lepton number density in equilibrium, 2 factor corresponds to the number of degrees of freedom
        nleq = (3./4.) * 2 * (zeta(3)/np.pi**2) * Tp**3

        # Thermally averaged scattering 
        gD2 = (3.*Tp**6/(4.*np.pi**5*self.v**4))*(m1sq + m2sq + m3sq)

        # Washout term for the scattering DL = 2 term
        WashDL2 = gD2/nleq

        #SR
        Gam_SR = self.Gamma_SR_num(Mphi, MBH, ast, fG_SR)
            
        return FBEqs(x, y0, ngam, self.M1, self.M2, self.M3, Mphi, eps,
                     self._d1, self._w1, self._n1eq,
                     self._dPBH1_prim, self._dPBH1_sec, self._dSR1, np.real(WashDL2), xilog10, Gam_SR, Gam_phi, pphi, BR)

    #******************************************************#
    #        Equations After PBH evaporation               #
    #******************************************************#
    def RHS_aBE(self, x, y0, eps, ngam, nPBHi, MBHi, asi, Mphi, g, pphi, BR, x_ev):
        
        Tp   = y0[1]             # Plasma Temperature
        TBH  = bh.TBH(MBHi, asi) # BH final temperature
        k    = np.real(self.k1)
        z    = np.real(self.M1/Tp)
        zBH  = np.real(self.M1/TBH)
        zphi = Mphi/TBH

        from ulysses.ulsbase import my_kn2, my_kn1

        self._d1         = np.real(self.Gamma1 * my_kn1(z)/my_kn2(z))                # Therm-av RH decay width wrt to Plasma Temperature
        self._dPBH1_prim = np.real(self.Gamma1 * self.ME(zBH))                       # Therm-av RH decay width wrt to TBH, primary component
        self._dPBH1_sec  = np.real(self.Gamma1 * self.ME_N_phi(self.M1, Mphi, TBH))  # Therm-av RH decay width wrt to TBH, secondary component
        self._dSR1       = np.real(self.Gamma1 * (2.*self.M1/Mphi))                  # Boosted RHN decay width from the Superradiant cloud, assuming phi decaying at rest

        Gam_phi     = self.Gphi(Mphi, self.M1, g)
        
        self._w1    = self._d1 * (my_kn2(z) * z**2/(3. * zeta(3.)))

        # RH neutrino equilibrium number density, normalized to initial photon density
        self._n1eq  = (10**(3*x) * self.M1**2 * Tp * my_kn2(z))/(np.pi**2)/ngam
        
        # Neutrino masses squared
        m1sq = self.SqrtDm[0,0]**4
        m2sq = self.SqrtDm[1,1]**4
        m3sq = self.SqrtDm[2,2]**4

        # Lepton number density in equilibrium, 2 factor corresponds to the number of degrees of freedom
        nleq = (3./4.) * 2. * (zeta(3)/np.pi**2) * Tp**3

        # Thermally averaged scattering 
        gD2 = (3.*Tp**6/(4.*np.pi**5*self.v**4))*(m1sq + m2sq + m3sq)

        # Washout term for the scattering DL = 2 term
        WashDL2 = gD2/nleq

        return FBEqs_aBE(x, y0, ngam, self.M1,self.M2,self.M3, Mphi, eps, self._d1, self._w1, self._n1eq,
                         nPBHi,  self._dPBH1_prim, self._dPBH1_sec, self._dSR1, np.real(WashDL2), Gam_phi, pphi, BR, x_ev, TBH)

    #******************************************************#
    #                     Main Program                     #
    #******************************************************#

    @property
    def EtaB(self):
        
        Mi    = 10**(self.MPBHi) # PBH initial Mass in grams
        asi   = self.aPBHi       # PBH initial rotation a_star factor
        bi    = 10**(self.bPBHi) # Reduced Initial PBH fraction, beta^prime

        g     = 10.**self.log_g                      # coupling between scalar and RHNs

        BR    = self.BR
        
        #Mphi  = self.alphi/(bh.GCF * Mi/bh.GeV_in_g) # Scalar mass, fixed from the gravitational coupling alpha = G * MPBH * Mphi
        #self.M1 = (1. - 0.025)*0.5*Mphi
        Mphi = 2.05*self.M1#/(1. - 0.025)
        
        self.M2 = 10.*self.M1
        self.M3 = 100.*self.M1

        #print(self.M1, self.M2, self.M3)

        #print("{0:6E}".format(Mphi), g, "{0:6E}".format(self.M1), "{0:6E}".format(bh.GCF * Mi/bh.GeV_in_g * Mphi))

        assert 0. <= asi and asi < 1., colored('initial spin factor a* is not in the range [0., 1.)', 'red')
        assert bi < np.sqrt(bh.gamma), colored('initial PBH density is larger than the total Universe\'s budget', 'red')
        assert self.M1 <= 0.5*Mphi,    colored('decay of scalar into RHNs not kinematically allowed', 'red')

        Ti    = ((45./(16.*106.75*(np.pi*bh.GCF)**3.))**0.25) * np.sqrt(bh.gamma * bh.GeV_in_g/Mi) # Initial Universe temperature in GeV

        rRadi = (np.pi**2./30.) * bh.gstar(Ti) * Ti**4  # Initial radiation energy density -- assuming a radiation dominated Universe
        rPBHi = (bi/(np.sqrt(bh.gamma) -  bi))*rRadi    # Initial PBH energy density, assuming collapse of density fluctuations
        ngam  = (2.*zeta(3)/np.pi**2)*Ti**3             # Initial photon number density

        N1RTi  = 0.      # Initial condition for Thermal RH neutrino number density
        N1RBpi = 0.      # Initial condition for PBH-emitted RH neutrino number -- primary
        N1RBsi = 0.      # Initial condition for PBH-emitted RH neutrino number -- secondary
        N1RSi  = 0.      # Initial condition for Superradiance-emitted RH neutrino number
        NphiBi = 0.      # Initial condition for scalar from evaporation
        NphiSi = 1.e-10  # Initial condition for scalar Superradiance, needs to be larger than 0
        NBLi   = 0.      # Initial condition for B-L asymmetry

        #Define fixed quantities for BEs
        eps1tt = np.real(self.epsilon1ab(2,2))
        eps1mm = np.real(self.epsilon1ab(1,1))
        eps1ee = np.real(self.epsilon1ab(0,0))
        eps1tm =         self.epsilon1ab(2,1)
        eps1te =         self.epsilon1ab(2,0)
        eps1me =         self.epsilon1ab(1,0)

        eps = [eps1tt, eps1mm, eps1ee, eps1tm, eps1te, eps1me] # Array for CP violation elements

        p_phi = 0.
        
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #                                           Solving the equations                                                   #
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        Min  = Mi   # We save the initial PBH mass.
        asin = asi  # We save the initial PBH spin
        
        xilog10 = 0. # Fixing the initial scale factor to be 1

        # Defining arrays to save the solution for the different components
        
        xBE     = [] # Log10 of the scale factor
        MBHBE   = [] # PBH mass
        astBE   = [] # PBH spin
        RadBE   = [] # Radiation energy density
        PBHBE   = [] # PBH energy density
        TBE     = [] # Plasma temperature
        N1RTBE  = [] # Thermal RH neutrino number density
        N1RBpBE = [] # PBH-emitted RH neutrino number density -- primary
        N1RBsBE = [] # PBH-emitted RH neutrino number density -- secondary
        N1RSBE  = [] # Superradiance-emitted RH neutrino number density
        NphBBE  = [] # Scalar number from evaporation
        NphSBE  = [] # Scalar number from Superradiance
        NBLBE   = [] # B-L number densities

        """
        The PBH evolution is done iteratively since the solver cannot reach the Planck Mass directly.
        We evolve the Friedmann & Boltzmann equations from the initial mass to 1% of such initial mass.
        Then, we use the solutions as initial conditions for a new iteraction.
        This is repeated until the PBH mass reaches the Planck mass.
        """

        i = 0

        while Mi > 1.125*bh.MPL: # We solve the equations until the PBH mass is larger than Planck Mass

            #----------------------------------------------------------------------#
            #     Compute BH lifetime and scale factor in which PBHs evaporate     #
            #----------------------------------------------------------------------#
    
            tau_sol = solve_ivp(fun=lambda t, y: bh.ItauRH_SR(t, y, self.M1, self.M2, self.M3, Mphi), t_span = [-80, 40.], y0 = [Mi, asi], 
                                  rtol=1.e-5, atol=1.e-20, dense_output=True)

            if i == 0: # For the first iteration, we determine the PBH lifetime.
                
                tau = tau_sol.t[-1] # Log10@PBH lifetime in inverse GeV
                
                #+++++++++++++++++++++++++++#
                #      Average momentum     #
                #+++++++++++++++++++++++++++#
                
                Sol_t = tau_sol.sol
                p_phi = self.p_average_phi(Mi, asi, Mphi, tau, Sol_t)

            # We compute the Log10@scale factor, xflog10, when the PBH evaporation happens
            if bi > 1.e-19*(1.e9/Mi): # If the initial PBH density is large enough, we include all components
                xf = root(bh.afin, [40.], args = (rPBHi, rRadi, 10.**tau, 0.), method='lm', tol=1.e-40) # Scale factor
                xflog10 = xf.x[0]
                
            else: # If the initial PBH density is negligible, we consider a radiation dominated Universe to obtain xflog10
                xfw = np.sqrt(1. + 4.*10.**tau*np.sqrt(2.*np.pi*bh.GCF*rRadi/3.))
                xflog10 = np.log10(xfw)
                
            #------------------------------------------------------------#
            #                                                            #
            #               Equations Before BH evaporation              #
            #                                                            #
            #------------------------------------------------------------#

            StopM = lambda t, x:self.StopMass(t, x, Mi) # Event to stop when the mass is 1% of the initial mass
            StopM.terminal  = True
            StopM.direction = -1.

            y0 = [Mi, asi, rRadi, rPBHi, Ti, N1RTi, N1RBpi, N1RBsi, N1RSi, NphiBi, NphiSi, NBLi] # Initial condition

            # Solving Equations
            solFBE = solve_ivp(lambda t, z: self.RHS(t, z, eps, np.real(ngam), xilog10, Mphi, g, self.fG_SR_, p_phi, BR),
                               [0., xflog10], y0, method='BDF', events=StopM, rtol=1.e-8, atol=1.e-10)

            assert solFBE.t[-1] > 0., colored('Solution going backwards...', 'red')

            # Appending solutions to predefined arrays
            
            xBE    = np.append(xBE,    solFBE.t[:] + xilog10)
            MBHBE  = np.append(MBHBE,  solFBE.y[0,:])
            astBE  = np.append(astBE,  solFBE.y[1,:])
            RadBE  = np.append(RadBE,  solFBE.y[2,:])
            PBHBE  = np.append(PBHBE,  solFBE.y[3,:])
            TBE    = np.append(TBE,    solFBE.y[4,:])
            
            N1RTBE  = np.append(N1RTBE,  solFBE.y[5,:])
            N1RBpBE = np.append(N1RBpBE, solFBE.y[6,:])
            N1RBsBE = np.append(N1RBsBE, solFBE.y[7,:])
            N1RSBE  = np.append(N1RSBE,  solFBE.y[8,:])
            NphBBE  = np.append(NphBBE,  solFBE.y[9,:])
            NphSBE  = np.append(NphSBE,  solFBE.y[10,:])
            
            NBLBE  = np.append(NBLBE, solFBE.y[11,:])

            # Updating the initial conditions for next iteration

            Mi     = solFBE.y[0,-1]
            asi    = solFBE.y[1,-1]
            rRadi  = solFBE.y[2,-1]
            rPBHi  = solFBE.y[3,-1]
            Ti     = solFBE.y[4,-1]
            N1RTi  = solFBE.y[5,-1]
            N1RBpi = solFBE.y[6,-1]
            N1RBsi = solFBE.y[7,-1]
            N1RSi  = solFBE.y[8,-1]
            NphiBi = solFBE.y[9,-1]
            NphiSi = solFBE.y[10,-1]
            
            NBLi  = solFBE.y[11,-1]
           
            xilog10 += solFBE.t[-1]

            i += 1

            assert i < 100, print(colored('Loop is stuck!', 'red'), Mi, bi)

        else:
            xflog10 = xilog10  # We update the value of log10(scale factor) at which PBHs evaporate
            
        #------------------------------------------------------------#
        #                                                            #
        #                Solution after BH evaporation               #
        #                                                            #
        #------------------------------------------------------------#

        """
        If Thermal Leptogenesis occurs after PBH evaporation, we solve the second set of equations with
        initial conditions from the last evolution step, and then we join the solutions
        """

        # We determine the Log10@ scale factor for maximum value of z = M/T after PBH evaporation
        xzmax = xflog10 + np.log10(self._zmax*TBE[-1]/self.M1)
        xpmax = xflog10 + np.log10(self._zmax*TBE[-1]/Mphi)
        
        if xflog10 < xzmax:
            
            # Initial condition for second set of equations, taken from last evolution step
            y0_aBE = [RadBE[-1], TBE[-1], N1RTBE[-1], N1RBpBE[-1],  N1RBsBE[-1], N1RSBE[-1], NphBBE[-1], NphSBE[-1], NBLBE[-1]] 
            
            # Solving Equations
            solFBE_aBE = solve_ivp(lambda t, z: self.RHS_aBE(t, z, eps, np.real(ngam), rPBHi/(Min/bh.GeV_in_g), np.real(Min), asin, Mphi, g, p_phi, BR, xflog10),
                                   [xflog10, xzmax], y0_aBE, method='BDF', rtol=1.e-5, atol=1.e-10)

            npaf = solFBE_aBE.t.shape[0] # Dimension of solution array from Eqs solution

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            #       Joining the solutions before and after evaporation       #
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
            
            xBE    = np.append(xBE, solFBE_aBE.t[:])
            
            MBHBE  = np.append(MBHBE,  np.full(npaf, solFBE.y[0,-1]))
            astBE  = np.append(astBE,  np.full(npaf, solFBE.y[1,-1]))
            RadBE  = np.append(RadBE,  solFBE_aBE.y[0,:])
            PBHBE  = np.append(PBHBE,  np.zeros(npaf))
            TBE    = np.append(TBE,    solFBE_aBE.y[1,:])
            
            N1RTBE  = np.append(N1RTBE,  solFBE_aBE.y[2,:])
            N1RBpBE = np.append(N1RBpBE, solFBE_aBE.y[3,:])
            N1RBsBE = np.append(N1RBsBE, solFBE_aBE.y[4,:])
            N1RSBE  = np.append(N1RSBE,  solFBE_aBE.y[5,:])
            NphBBE  = np.append(NphBBE,  solFBE_aBE.y[6,:])
            NphSBE  = np.append(NphSBE,  solFBE_aBE.y[7,:])   
            NBLBE   = np.append(NBLBE,   solFBE_aBE.y[8,:])
       
        #------------------------------------------------------------#
        #                                                            #
        #                     Conversion to eta_B                    #
        #                                                            #
        #------------------------------------------------------------#

        gstarSrec = bh.gstarS(0.3e-9) # d.o.f. at recombination
        gstarSoff = bh.gstarS(TBE[-1])  # d.o.f. at the end of leptogenesis
     
        SMspl       = 28./79.
        zeta3       = zeta(3)
        ggamma      = 2.
        coeffNgamma = ggamma*zeta3/np.pi**2
        Ngamma      = coeffNgamma*(10**xBE*TBE)**3
        coeffsph    = (SMspl * gstarSrec)/(gstarSoff * Ngamma)

        nb = coeffsph * NBLBE * ngam

        etaB = nb[-1]

        #print(NphBBE)

        dat = np.array([self.M1/TBE, np.real(N1RTBE), np.real(N1RBpBE),  np.real(N1RBsBE),  np.real(N1RSBE), np.real(NphBBE), np.real(NphSBE), np.real(nb)])

        dat = dat.T

        np.savetxt("./test_lepto.txt",dat)
                        
        #self.setEvolDataPBH(dat)
        return etaB
