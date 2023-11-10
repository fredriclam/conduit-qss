import datetime
import logging
import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.sparse
import scipy.special
from time import perf_counter

class SplitSolver():
  ''' Solver for low-Mach number split of the mass, momentum, and composition
  equations for isothermal flow.'''

  def __init__(self, params:dict, Nt=100):
    ''' Initialize using dict. Checks schema for each parameter. '''
    self.Nt = Nt
    # Internal options
    self._use_grid_RHS_advance = False

    # Set up logger
    str_timenow = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")
    logger_name = f"lowM_split_{str_timenow}"
    logger_filename = f"{logger_name}.log"
    # logging.basicConfig(filename=logger_filename,
    #                 format='%(asctime)s %(message)s',
    #                 filemode='w')
    # self.logger = logging.getLogger()
    # self.logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(logger_filename)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    self.logger = logging.getLogger(logger_name)
    self.logger.setLevel(logging.DEBUG)
    self.logger.addHandler(handler)

    try:
      self.params = params
      self.T0 = params["T0"]
      self.conduit_radius = params["conduit_radius"]
      self.R0 = self.conduit_radius
      self.tau_d = params["tau_d"] # 2.5
      self.tau_f = params["tau_f"] # 2.5
      self.vf_g_crit = params["vf_g_crit"] # 0.8
      self.solubility_n = params["solubility_n"]
      self.solubility_k = params["solubility_k"]
      self.K = params["K"]
      self.rho_m0 = params["rho_m0"]
      self.p_m0 = params["p_m0"]
      self.R_a = 0.0
      self.R_wv = params["R_wv"]
      # External friction model
      self.F_fric_viscosity_model = params["F_fric_viscosity_model"]
    except KeyError as e:
      raise ValueError("Input params dict missing parameter. See KeyError. ") from e

  ''' Thermodynamic quantities and slopes '''

  def p_ptilde(self, rho, yWv) -> tuple:
    ''' Return pressure-like quantities.
      Includes yA formally (mass fraction of air) but this quantity is set to zero. '''
    yA = 0.0
    R_a = 0.0
    yM = 1.0 - yWv
    yRGas = (yA * R_a + yWv * self.R_wv)
    ptilde = self.K * (yM * rho / self.rho_m0 - 1.0) \
             + self.p_m0 + rho * yRGas * self.T0
    p = 0.5 * (ptilde
               + np.sqrt(ptilde * ptilde
               + 4 * rho * yRGas * self.T0 * (self.K - self.p_m0)))
    return p, ptilde

  def isothermal_sound_speed_squared(self, yWv, p, ptilde):
    yM = 1.0 - yWv
    return (yM * p * self.K / self.rho_m0
            + yWv * (p + self.K - self.p_m0) * self.R_wv * self.T0
           ) / (2*p - ptilde)
  
  def isothermal_mach_squared(self, u, yWv, p, ptilde):
    return u * u / self.isothermal_sound_speed_squared(yWv, p, ptilde)
  
  def isothermal_mach_simple(self, u, yWv, p):
    rho = self.rho_p(yWv, p)
    _, ptilde = self.p_ptilde(rho, yWv)
    return u / np.sqrt(self.isothermal_sound_speed_squared(yWv, p, ptilde))
  
  def dp_dyw(self, rho, p, ptilde):
    return (-p * (self.K * rho / self.rho_m0) 
            + (p + self.K - self.p_m0) * rho * self.R_wv * self.T0
           ) / (2*p - ptilde)

  def drho_dyw(self, rho, p):
    if p == 0:
      return np.zeros_like(rho)
    rhoWv = p / (self.R_wv * self.T0)
    vWv = 1.0 / rhoWv
    # Magma
    rhoM = self.rho_m0 * (1 + (p - self.p_m0) / self.K)
    vM = 1.0 / rhoM
    return - rho * rho * (vWv - vM)
  
  def volfrac_gas(self, rho, yWv, p):
    return 1.0 - rho * (1 - yWv) / self.rhoM_p(p)
    # Gas-based vol frac may divide by zero
    # rhoWv = p / (self.R_wv * self.T0)
    # return rho * yWv / rhoWv
    
  def yWd_target(self, yWv, yWt, yC, p):
    yM = 1.0 - yWv
    yWd = yWt - yWv
    yL = (yM - (yC + yWd))
    p_like = np.clip(p, 0, None)
    return np.clip(self.solubility_k * p_like ** self.solubility_n * yL, 0, yWt)

  def yWv_eq(self, yWt, yC, p):
    yL = (1.0 - (yC + yWt))
    p_like = np.clip(p, 0, None)
    yWd = np.clip(self.solubility_k * p_like ** self.solubility_n * yL, 0, yWt)
    return yWt - yWd
  
  def rhoM_p(self, p):
    return self.rho_m0 * (1.0 + (p - self.p_m0) / self.K)

  def vM_p(self, p):
    return 1.0 / self.rhoM_p(p)
  
  def rhoWv_p(self, p):
    return p / (self.R_wv * self.T0)
  
  def vWv_p(self, p):
    return (self.R_wv * self.T0) / p
  
  def rho_p(self, yWv, p):
    return 1.0 / (yWv * self.vWv_p(p) + (1.0 - yWv) * self.vM_p(p))

  def K_mix(self, yWv, p):
    ''' Isothermal bulk modulus: rho * c^2 '''
    rho = self.rho_p(yWv, p)
    _, ptilde = self.p_ptilde(rho, yWv)
    cT2 = self.isothermal_sound_speed_squared(yWv, p, ptilde)
    return rho * cT2

  def dvdp(self, yWv, p):
    ''' Derivative of volume w.r.t. pressure at constant yWv, T, with volume
    v(p) expressed as a function of p.'''
    rhoM = self.rhoM_p(p)
    return -(yWv * self.R_wv * self.T0 / (p * p)
            + (1 - yWv) / (rhoM * rhoM) * self.rho_m0 / self.K)
  
  def ddp_cT2(self, yWv, yWt, yC, p):
    ''' Derivative of sound speed w.r.t. pressure at constant yWv, T, yWt,...
    with sound speed expressed as a function of (p, yWv, T).
    
    Note that sound speed squared itself is a derivative of pressure as a
    function of density, so the sound speed is a composed function
    cT2( rho (p) ). '''
    rho = self.rho_p(yWv, p)
    _, ptilde = self.p_ptilde(rho, yWv)
    yM = 1.0 - yWv
    # Compute drho/dp (const yWv)
    _drhodp = - rho * rho * self.dvdp(yWv, p)
    # Compute d/dp (2p - pTilde)
    _ddp_group = 2 - (self.K / self.rho_m0 * yM
                      + self.R_wv * self.T0 * yWv) * _drhodp
    _num = (2*p - ptilde) \
      * (yM * self.K / self.rho_m0 + yWv * self.R_wv * self.T0) \
      - (yM * p * self.K / self.rho_m0 
         + yWv * (p + self.K - self.p_m0) * self.R_wv * self.T0) \
      * _ddp_group
    return _num / ((2*p - ptilde) * (2*p - ptilde))
  
  def RHS_jac(self, p, u, yWv, yWt, yC, yF, also_return_RHS=False):
    ''' Analytic Jacobian of RHS as Ainv@f, w.r.t state vector [p, u]. Only
    for tau_d, tau_f > 0.
    This function proceeds by computing:
      1. (Ainv)_ij @ (d_k f_j)
    and then
      2. (d_k (Ainv)_ij) @ f_j.

    If also_return_RHS==True, returns (jac, f)
    '''

    ''' Compute simple dependents '''
    rho = self.rho_p(yWv, p)
    _, ptilde = self.p_ptilde(rho, yWv)
    yL = 1.0 - (yWt + yC)
    p_like = np.clip(p, 1e-16, None)
    yWdHat = np.clip(self.solubility_k * p_like ** self.solubility_n * yL,
                     0, yWt)
    cT2 = self.isothermal_sound_speed_squared(yWv, p, ptilde)
    mu = self.F_fric_viscosity_model(
        self.T0, yWv, yF, yWt=yWt, yC=yC)
    F_fric = 8.0 * mu / (self.conduit_radius * self.conduit_radius) \
                  * u * np.clip(1.0 - yF/(1.0 - yWv), 0.0, 1.0)
    # Shared derivatives
    _dvdp = self.dvdp(yWv, p)
    _dp_dyw = self.dp_dyw(rho, p, ptilde)
    _drhodp = - rho * rho * _dvdp

    ''' Compute Ainv, f '''
    #   d_k(Ainv_ij) @ f_j + Ainv_ij @ df_jk
    Ainv = np.array([[u, -rho*cT2], [-1/rho, u]]) / (u*u - cT2)
    f = np.stack([
      _dp_dyw * (yWt - yWdHat - yWv) / self.tau_d,
      -F_fric/rho - 9.8,
    ], axis=-1)
    
    ''' Compute Ainv times jacobian of f (without inverse mass matrix) '''
    Juu = -8.0 * mu / rho / (self.conduit_radius * self.conduit_radius) \
                * np.clip(1.0 - yF/(1.0 - yWv), 0.0, 1.0)
    Jup = -8.0 * mu / (self.conduit_radius * self.conduit_radius) \
                * u * np.clip(1.0 - yF/(1.0 - yWv), 0.0, 1.0) \
                * _dvdp
    Jpu = np.zeros_like(yF)
    # Compute dp/dyWv group (const rho, T) with product rule terms
    _Pi = (-p * (self.K * rho / self.rho_m0) 
                + (p + self.K - self.p_m0) * rho * self.R_wv * self.T0
              ) / (2*p - ptilde)
    _g1 = -2 / (2*p - ptilde)* _Pi
    _g2 = (self.K / self.rho_m0 * (1.0 - yWv) + self.R_wv * self.T0 * yWv) \
          / (2*p - ptilde) * _drhodp * _Pi
    _g3 = 1/rho * _drhodp * _Pi
    _g4 = rho / (2*p - ptilde) * (-self.K / self.rho_m0 + self.R_wv * self.T0)
    ddp_dpdYWv = _g1 + _g2 + _g3 + _g4
    # Compute Jpp using product rule
    _h1 = ddp_dpdYWv * (yWt - yWdHat - yWv) / self.tau_d
    _h2 = -_dp_dyw / self.tau_d * (self.solubility_k
          * self.solubility_n * p_like**(self.solubility_n - 1) * yL)
    _h2 = np.where(yWt >= self.solubility_k * p_like ** self.solubility_n * yL,
                   _h2, 0.0)
    Jpp = _h1 + _h2
    # Assemble jacobian of f
    _fjac = np.zeros((p.size, 2, 2))
    _fjac[:,0,0] = Jpp
    _fjac[:,0,1] = Jpu
    _fjac[:,1,0] = Jup
    _fjac[:,1,1] = Juu
    # rank2-rank2 part: Ainv @ df
    # _prod22 = np.einsum("ij..., ...jk -> ...ik", Ainv, _fjac)
    #   but with Ainv factored out, this is not necessary.

    ''' Compute jacobian of A times f '''
    # dAinv/dp @ f == Ainv @ dA/dp @ Ainv @ f as row
    # dAinv/du @ f = Ainv @ dA/du @ Ainv @ f as row
    _dAdp_01 = _drhodp * cT2 + rho * self.ddp_cT2(yWv, yWt, yC, p)
    _dAdp_10 = _dvdp
    # Compute Ainv @ f
    _Ainvf = np.einsum("ij..., ...j -> ...i", Ainv, f)
    # Compute dA/dp @ Ainv @ f explicitly (sparse off diagonal matrix)
    _dAdp_Ainvf = np.stack([
      _dAdp_01 * _Ainvf[...,1],
      _dAdp_10 * _Ainvf[...,0],
    ], axis=-1)
    # rank3-rank1 part: dAinv @ f
    # _prod31 = np.stack([
    #   -np.einsum("ij..., ...j -> ...i", Ainv, _dAdp_Ainvf),
    #   -np.einsum("ij..., ...j -> ...i", Ainv, _Ainvf),
    # ], axis=2)
    # Factorized version: stack partial results without left-factor Ainv
    _factoredprod31 = np.stack([-_dAdp_Ainvf, -_Ainvf,], axis=-1)

    jac = np.einsum("ij..., ...jk -> ...ik", Ainv, _fjac + _factoredprod31)
    if also_return_RHS:
      return jac, _Ainvf
    else:
      return jac
    # Unfactorized version would do:
    # return _prod22 + _prod31
  
  def dphidYWv(self, yWv, p):
    ''' Volume fraction slope w.r.t. yWv at const pressure, (and const T) '''
    vWv = self.vWv_p(p)
    vM = self.vM_p(p)
    v = yWv * vWv + (1.0 - yWv) * vM
    return vWv / v * (1.0 - yWv * (vWv - vM) / (v))
  
  def smoother(self, x, scale=0.15):
      ''' Returns one-sided smoothing u(x) of a step, such that
        1. u(x < -scale) = 0
        2. u(x >= 0) = 1.
        3. u smoothly interpolates from 0 to 1 in between.
      From Quail implementation
      '''
      # Shift, scale, and clip to [-1, 0] to prevent exp overflow
      _x = np.clip(x / scale + 1, 0, 1)
      f0 = np.exp(-1/np.where(_x == 0, 1, _x))
      f1 = np.exp(-1/np.where(_x == 1, 1, 1-_x))
      # Return piecewise evaluation
      return np.where(_x >= 1, 1,
              np.where(_x <= 0, 0, 
                f0 / (f0 + f1)))
  
  def smoother_derivative(self, x, scale=0.15):
    # Rescale to reference domain
    _x = x / scale + 1.0
    # Mask array outside (0,1)
    _x = np.ma.masked_where(((_x <= 0) | (_x >= 1)), _x)
    # Compute subcomponent f
    f0 = np.exp(-1.0 / _x)
    f1 = np.exp(-1.0 / (1.0 - _x))
    # Evaluate derivative of f
    fprime0 = f0 / (_x*_x)
    fprime1 = -f1 / ((1.0 - _x)*(1.0 - _x))
    # Compute smoother
    _prefactor = 1.0 / (f0 + f1)
    # Compute derivative of smoother
    gprime = _prefactor * _prefactor * (f1 * fprime0 - f0 * fprime1) / scale
    # Return as array with mask -> 0, re-cast for scalar case
    return np.ma.masked_array(gprime).filled(0)

  def jac(self, yWv, yF, p):
    ''' Jacobian of the yWv-yF system. Takes scalar inputs. '''
    # Compute intermediates
    rho = self.rho_p(yWv, p)
    vf_g = self.volfrac_gas(rho, yWv, p)
    smoothed_ind = self.smoother(vf_g - self.vf_g_crit)
    smoothed_ind_slope = self.smoother_derivative(vf_g - self.vf_g_crit)
    # Compute Jacobian terms
    J00 = -1.0 / self.tau_d
    J11 = -smoothed_ind / self.tau_f
    J10 = J11 + (1 - yWv - yF) / self.tau_f \
          * smoothed_ind_slope * self.dphidYWv(yWv, p)
    return np.array([[J00, 0], [J10, J11]])
  
  def phirm(self, A, b, r:int=3) -> tuple:
    ''' Evaluates phi1(A) * b through phir(A) * b
      Returns tuple: (expmL, phi1*b, phi2*b, ..., phir*b),
      where expmL is the matrix exponentail of the lifted matrix. '''
    ndims = A.shape[0]
    # Lift matrix
    # [A b         ]
    # [  0 1       ]
    # [    0 1 ... ]
    L = np.zeros((ndims+r, ndims+r))
    L[:ndims, :ndims] = A
    L[:ndims, ndims]  = b
    # Ones on superdiagonal
    for i in range(1,r):
      L[ndims-1+i, ndims+i]  = 1
    # Compute matrix exp of lift matrix
    expmL = scipy.linalg.expm(L)
    return (expmL, *[expmL[:ndims, ndims+i] for i in range(r)])
  
  def u_R3_spiral(N, RHS, JAC):
    ''' Third-order Rosenbrock method (Hochbruck et al. 2009). '''
    ''' This is a copy-paste from a test notebook. '''
    raise NotImplementedError("Dont use this, just for reference")
    t_range = np.linspace(0,1,N+1)
    dt = t_range[1] - t_range[0]
    u = np.zeros((2, t_range.size))
    # Initial condition
    u[0,0] = 1.0
    
    for i in range(N):
      t = t_range[i]
      # Compute Jacobian at beginning of timestep
      Jn = JAC(u[:,i])
      # Compute RHS at beginning of timestep
      RHSn = RHS(u[:,i])
      # Predictor step
      u2 = u[:,i] + dt * phi1m(dt * Jn, RHSn).ravel()
      # Compute difference in nonlinear part of RHS
      Ndiff = RHS(u2) - RHSn - Jn @ (u2 - u[:,i])
      # Compute corrector (also error estimator)
      corr = dt * 2.0 * phirm(dt * Jn, Ndiff)[3].ravel()
      # Update using corrector
      u[:,i+1] = u2 + corr
    return t_range, u
  
  def isothermal_eq_sound_speed_squared(self, yWt, yC, p):
    # Compute phasic mass fractions, mixture density
    yL = (1.0 - (yC + yWt))
    p_like = np.clip(p, 0, None)
    yWd = np.clip(self.solubility_k * p_like ** self.solubility_n * yL, 0, yWt)
    yWv = yWt - yWd
    rhoM = self.rhoM_p(p)
    vM = 1.0 / rhoM
    vWv = self.vWv_p(p)
    rho = 1.0 / (yWv * vWv + (1.0 - yWv) * vM)
    # Compute magma volatile saturation state
    if len(np.asarray(p).shape) == 0:
      is_sat:float = float(yWt > yWd)
    else:
      is_sat:np.array = np.array(yWt > yWd).astype(float)
    # Compute intermediate expression group
    _z = yWv * self.R_wv * self.T0 / (p*p) \
        + (1.0 - yWv) * self.rho_m0 / self.K / (rhoM*rhoM) \
        + self.solubility_k * self.solubility_n * p_like**(self.solubility_n - 1.0) \
          * yL * (vWv - vM) * is_sat
    return 1.0 / (rho * rho * _z)

  def full_steady_state_RHS(self, x, qfull:np.array,
                            is_debug_mode=False) -> np.array:
    ''' Full steady state ODE system RHS. '''
    # Unpack ODE vector
    q = np.asarray(qfull)
    rho_bar   = q[0,...]
    u         = q[1,...]
    rho_prime = q[2,...]
    yWv       = q[3,...]
    yWt       = q[4,...]
    yC        = q[5,...]
    yF        = q[6,...]
    num_states = q.shape[0]

    '''Compute aliases'''
    yM = 1.0 - yWv
    yWd = yWt - yWv
    rho = rho_bar + rho_prime

    p, ptilde = self.p_ptilde(rho, yWv)
    vf_g = self.volfrac_gas(rho, yWv, p)
    # Compute mixture isothermal sound speed, squared
    cT2 = self.isothermal_sound_speed_squared(yWv, p, ptilde)
    # Compute partial of pressure w.r.t. water mass fraction (const rho, T)
    dp_dyw = self.dp_dyw(rho, p, ptilde)
    # Compute partial of density w.r.t. water mass fraction (const p, T)
    drho_dyw = self.drho_dyw(rho, p)
    # Compute solubility based on pure silicate melt mass fraction
    yHat = self.yWd_target(yWv, yWt, yC, p)

    # Compute sources
    source_yWv = (yWd - yHat) / (self.tau_d)
    source_yWt = np.zeros_like(p)
    source_yC  = np.zeros_like(p)
    # source_yF  = (yM - yF) / (self.tau_f) \
    #              * np.asarray(vf_g - self.vf_g_crit > 0.0).astype(float)
    
    # One-sided smoothing for fragmentation source
    source_yF  = (yM - yF) / (self.tau_f) * self.smoother(vf_g - self.vf_g_crit)

    # Compute friction
    mu = self.F_fric_viscosity_model(self.T0, yWv, yF, yWt=yWt, yC=yC)
    F_fric = 8.0 * mu / (self.conduit_radius * self.conduit_radius) \
             * u * np.clip(1.0 - yF/yM, 0.0, 1.0)
    source_momentum = (-rho * 9.8 - F_fric) / rho
    # Compute LHS matrix
    #   for variables [rho_bar, u, rho_prime, yWv, yWt, yC, yF]
    A = np.zeros((num_states, num_states))
    A[0,:] = [u, rho,     u,       -u* drho_dyw * 0, 0, 0, 0] # drho_dyw reduces out from yw advection
    A[1,:] = [cT2/rho, u, cT2/rho, dp_dyw/rho , 0, 0, 0]
    A[2,2] = 1.0
    A[[3,4,5,6],[3,4,5,6]] = u
    # Compute source vector
    b = np.stack([0.0, #-drho_dyw * source_yWv,
                  source_momentum, # With / rho
                  0.0,
                  source_yWv,
                  source_yWt,
                  source_yC,
                  source_yF,
                  ], axis=0)
    
    if is_debug_mode:
      return A, b, np.linalg.solve(A, b)
    return np.linalg.solve(A, b)
  
  def full_steady_state_RHS_eq(self, x, qfull:np.array,
                            is_debug_mode=False) -> np.array:
    ''' Full steady state ODE system RHS for equilibrium chemistry. 
    Incompatible with other steady state. '''
    # Unpack ODE vector
    q = np.asarray(qfull)
    p         = q[0,...]
    u         = q[1,...]
    yWt       = q[2,...]
    yC        = q[3,...]
    num_states = q.shape[0]

    '''Compute water frac '''
    yL = (1.0 - (yC + yWt))
    p_like = np.clip(p, 0, None)
    yWd = np.clip(self.solubility_k * p_like ** self.solubility_n * yL, 0, yWt)
    yWv = yWt - yWd
    rho = self.rho_p(yWv, p)
    yM = 1.0 - yWv

    _, ptilde = self.p_ptilde(rho, yWv)
    vf_g = self.volfrac_gas(rho, yWv, p)
    # Compute mixture isothermal sound speed, squared
    # cT2 = self.isothermal_sound_speed_squared(yWv, p, ptilde)
    cT2 = self.isothermal_eq_sound_speed_squared(yWt, yC, p)


    # Compute partial of pressure w.r.t. water mass fraction (const rho, T)
    dp_dyw = self.dp_dyw(rho, p, ptilde)
    # Compute partial of density w.r.t. water mass fraction (const p, T)
    drho_dyw = self.drho_dyw(rho, p)
    # Compute sources
    source_yWt = np.zeros_like(p)
    source_yC  = np.zeros_like(p)

    # Compute fragmentation state (two-way fragmentation)
    yF  = yM * np.asarray(vf_g - self.vf_g_crit > 0.0).astype(float)
    # Compute friction
    mu = self.F_fric_viscosity_model(self.T0, yWv, yF, yWt=yWt, yC=yC)
    F_fric = 8.0 * mu / (self.conduit_radius * self.conduit_radius) \
             * u * np.clip(1.0 - yF/yM, 0.0, 1.0)
    source_momentum = (-rho * 9.8 - F_fric) / rho

    # Compute LHS matrix
    #   for variables [rho_bar, u, rho_prime, yWv, yWt, yC, yF]
    A = np.zeros((num_states, num_states))
    A[0,:] = [u, rho*cT2, 0 , 0]
    A[1,:] = [1.0/rho, u, 0, 0]
    A[[2,3],[2,3]] = u
    # Compute source vector
    b = np.stack([0.0,
                  source_momentum,
                  source_yWt,
                  source_yC,
                  ], axis=0)
    # Manual inversion
    Ainvb = np.zeros((num_states,))
    Ainvb[0] = b[1] * (-rho*cT2) / (u*u - cT2)
    Ainvb[1] = b[1] * (u) / (u*u - cT2)
    Ainvb[2] = b[2] / u
    Ainvb[3] = b[3] / u

    if is_debug_mode:
      return A, b, np.linalg.solve(A, b), Ainvb
    return Ainvb
  
  def full_steady_state_RHS_p(self, x, qfull:np.array,
                            is_debug_mode=False) -> np.array:
    ''' Full steady state ODE system RHS, with p in slot [0] instead of rho '''
    # Unpack ODE vector
    q = np.asarray(qfull)
    p         = q[0,...]
    u         = q[1,...]
    rho_prime = q[2,...] # Unused
    yWv       = q[3,...]
    yWt       = q[4,...]
    yC        = q[5,...]
    yF        = q[6,...]
    num_states = q.shape[0]

    '''Compute aliases'''
    yM = 1.0 - yWv
    yWd = yWt - yWv
    rho = self.rho_p(yWv, p)

    _, ptilde = self.p_ptilde(rho, yWv)
    vf_g = self.volfrac_gas(rho, yWv, p)
    # Compute mixture isothermal sound speed, squared
    cT2 = self.isothermal_sound_speed_squared(yWv, p, ptilde)
    # Compute partial of pressure w.r.t. water mass fraction (const rho, T)
    dp_dyw = self.dp_dyw(rho, p, ptilde)
    # Compute partial of density w.r.t. water mass fraction (const p, T)
    drho_dyw = self.drho_dyw(rho, p)
    # Compute solubility based on pure silicate melt mass fraction
    yHat = self.yWd_target(yWv, yWt, yC, p)

    # Compute sources
    source_yWv = (yWd - yHat) / (self.tau_d)
    source_yWt = np.zeros_like(p)
    source_yC  = np.zeros_like(p)
    source_yF  = (yM - yF) / (self.tau_f) \
                 * np.asarray(vf_g - self.vf_g_crit > 0.0).astype(float)
    # Compute friction
    mu = self.F_fric_viscosity_model(self.T0, yWv, yF, yWt=yWt, yC=yC)
    F_fric = 8.0 * mu / (self.conduit_radius * self.conduit_radius) \
             * u * np.clip(1.0 - yF/yM, 0.0, 1.0)
    source_momentum = (-rho * 9.8 - F_fric) / rho
    # Compute LHS matrix
    #   for variables [rho_bar, u, rho_prime, yWv, yWt, yC, yF]
    A = np.zeros((num_states, num_states))
    A[0,:] = [u, rho*cT2, 0.0, 0.0, 0, 0, 0]
    A[1,:] = [1/rho, u,   0.0, 0.0 ,0, 0, 0]
    A[2,2] = 1.0
    A[[3,4,5,6],[3,4,5,6]] = u
    # Compute source vector
    b = np.stack([dp_dyw * source_yWv,
                  source_momentum,
                  0.0,
                  source_yWv,
                  source_yWt,
                  source_yC,
                  source_yF,
                  ], axis=0)
    if is_debug_mode:
      return A, b, np.linalg.solve(A, b)
    return np.linalg.solve(A, b)

  def solve_full_steady_state(self, p0=80e6, u0=0.7, yWt0=0.025, yC0=0.4,
                              yF0=0.0, xlims:tuple=(0, 5000), M_eps=1e-2,
                              method="BDF", rtol=1e-5, atol=1e-6) -> object:
    ''' Solve the full steady state. '''
    # Compute phase mass fractions from solubility
    xi = self.solubility_k * p0 ** self.solubility_n
    yWd0 = np.clip(xi *(1.0 - (yWt0 + yC0)), 0, yWt0)
    yWv0 = yWt0 - yWd0
    yM0 = 1.0 - yWv0
    # Compute densities
    rhoM0 = self.rho_m0 * (1 + (p0 - self.p_m0) / self.K)
    vM0 = 1.0 / rhoM0
    vWv0 = self.R_wv * self.T0 / p0
    v0 = yM0 * vM0 + yWv0 * vWv0
    rho0 = 1.0 / v0
    # Set zero perturbative density
    rho_prime0 = 0.0
    # Assemble initial condition
    q0 = np.array([rho0, u0, rho_prime0, yWv0, yWt0, yC0, yF0])

    event_isothermal_choked = lambda x, qfull: self.isothermal_mach_squared(
      qfull[1,...], # u
      qfull[3,...], # yWv
      *self.p_ptilde(qfull[0,...] + qfull[2,...], # rho
                      qfull[3,...])) - (1.0 - M_eps)*(1.0 - M_eps) # yWv
    event_isothermal_choked.terminal = True
    # event_isothermal_choked.direction = 1.0

    soln_steady = scipy.integrate.solve_ivp(self.full_steady_state_RHS,
            xlims,
            q0,
            method=method,
            dense_output=True,
            # max_step=1.0,
            rtol=rtol,
            atol=atol,
            events=[event_isothermal_choked,])
    return soln_steady
  
  def solve_full_steady_state_eq(self, p0=80e6, u0=0.7, yWt0=0.025, yC0=0.4,
                              xlims:tuple=(0, 5000), M_eps=1e-2,
                              method="BDF", rtol=1e-5, atol=1e-6) -> object:
    ''' Solve the full steady state for equilibrium chemistry. 
    0th index of state vector is pressure p, not density. '''
    # Assemble initial condition
    q0 = np.array([p0, u0, yWt0, yC0])

    def event_isothermal_choked(x, qfull):
      p = qfull[0,...]
      u = qfull[1,...]
      yWt = qfull[2,...]
      yC = qfull[3,...]
      yWv = yWt - np.clip(
        self.solubility_k * np.clip(p, 0, None) ** self.solubility_n 
        * (1.0 - (yC + yWt)), 0, yWt)
      rho = self.rho_p(yWv, p)
      _, ptilde = self.p_ptilde(rho, yWv)

      return self.isothermal_mach_squared(u, yWv, p, ptilde)\
        - (1.0 - M_eps)*(1.0 - M_eps)
    event_isothermal_choked.terminal = True
    # event_isothermal_choked.direction = 1.0

    soln_steady = scipy.integrate.solve_ivp(self.full_steady_state_RHS_eq,
            xlims,
            q0,
            method=method,
            dense_output=True,
            # max_step=1.0,
            rtol=rtol,
            atol=atol,
            events=[event_isothermal_choked,])
    return soln_steady

  def time_indep_RHS(self, x, q2:np.array, r_interpolants,
                     is_debug_mode=False, use_integrated_friction=True,
                     is_r_pre_eval=False):
    ''' RHS for time-independent portion of the system.
    Inputs:
      x -- independent coordinate
      q2 -- state vector of size 2 (p, u)
      r_interpolants -- interpolation for slow-time-dependent variables
          if r_pre_eval==False, r_interpolants is callable
          else, r_interpolants are np.array
      is_debug_mode (optional) -- flag for returning debug quantities
      use_integrated_friction (optional) -- use built-in friction model (faster)
    '''    
    # Interpolate solution for time-dependent quantities
    if is_r_pre_eval:
      r = r_interpolants
    else:
      r = r_interpolants(x)
    if len(r.shape) > 1:
      # Shape check
      raise ValueError("Data has more than 1 axis. This function is only for scalar inputs.")
    # Unpack quantities
    p         = q2[0]
    u         = q2[1]
    yWv       = r[1]
    yWt       = r[2]
    yC        = r[3]
    yF        = r[4]
    # Extract spatial derivative of rho prime
    # dyw_dt        = dyw_dt_interpolant(t)
    '''Compute aliases'''
    yM = 1.0 - yWv
    yWd = yWt - yWv
    yL = (yM - (yC + yWd))

    '''Compute EOS quantities'''
    v = yWv * self.vWv_p(p) + yM * self.vM_p(p)
    rho = 1.0 / v
    _, ptilde = self.p_ptilde(rho, yWv)
    # Compute mixture isothermal sound speed, squared
    cT2 = self.isothermal_sound_speed_squared(yWv, p, ptilde)
    # Compute partial of pressure w.r.t. water mass fraction (const rho, T)
    dp_dyw = self.dp_dyw(rho, p, ptilde)

    # Compute solubility based on pure silicate melt mass fraction
    p_like = max(p, 0)

    if self.solubility_n == 0.5:
      yHat = min(max(self.solubility_k * np.sqrt(p_like) * yL, 0), yWt)
    else:
      yHat = np.clip(self.solubility_k * p_like ** self.solubility_n * yL, 0, yWt)
  
    # Compute sources
    source_yWv = (yWd - yHat) / (self.tau_d)

    # Compute momentum load
    if use_integrated_friction:
      # Calculate pure melt viscosity (Hess & Dingwell 1996)
      mfWd = yWd / yL # mass concentration of dissolved water: note model is
      # maximized for a small number, diverging at 0
      mfWd = max(mfWd, 1e-8)
      log_mfWd = np.log(mfWd*100)
      log10_vis = -3.545 + 0.833 * log_mfWd
      log10_vis += (9601 - 2368 * log_mfWd) / (self.T0 - 195.7 - 32.25 * log_mfWd)
      # Melt viscosity with overflow protection
      meltVisc = 10**min(log10_vis, 300)
      # Calculate relative viscosity due to crystals (Costa 2005).
      alpha = 0.999916
      phi_cr = 0.673
      gamma = 3.98937
      delta = 16.9386
      B = 2.5
      # Compute volume fraction of crystal at equal phasic densities
      # Using crystal volume per (melt + crystal + dissolved water) volume
      phi_ratio = max((yC / yM) / phi_cr, 0.0)
      erf_term = scipy.special.erf(
        np.sqrt(np.pi) / (2 * alpha) * phi_ratio * (1 + phi_ratio**gamma))
      crysVisc = (1 + phi_ratio**delta) * ((1 - alpha * erf_term)**(-B * phi_cr))
      mu = meltVisc * crysVisc
    else:
      mu = self.F_fric_viscosity_model(self.T0, yWv, yF, yWt=yWt, yC=yC)
    
    frag_factor = min(max(1.0 - yF/yM, 0.0), 1.0)
    F_fric = 8.0 * mu / (self.conduit_radius * self.conduit_radius) * u \
      * frag_factor
    source_momentum = (-rho * 9.8 - F_fric) / rho

    # Debug output
    if not is_debug_mode:
      # Compute explicitly inverted LHS matrix
      _det = u*u - cT2
      b0 = dp_dyw * source_yWv
      b1 = source_momentum
      return np.array([(u * b0 + -rho * cT2 * b1) / _det,
                      (-v * b0 + u * b1) / _det
                      ])
    
    #   for state vector [rho_bar, u]
    Ainv = np.array([[u, -rho*cT2], [-v, u]]) / (u*u - cT2)
    # Compute RHS vector
    b = np.stack([dp_dyw * source_yWv,
                  + source_momentum,
                  ], axis=0)
    return Ainv, b, Ainv @ b
  
  def time_indep_RHS(self, x, q2:np.array, r_interpolants,
                     is_debug_mode=False, use_integrated_friction=True,
                     is_r_pre_eval=False):
    ''' RHS for time-independent portion of the system.
    Inputs:
      x -- independent coordinate
      q2 -- state vector of size 2 (p, u)
      r_interpolants -- interpolation for slow-time-dependent variables
          if r_pre_eval==False, r_interpolants is callable
          else, r_interpolants are np.array
      is_debug_mode (optional) -- flag for returning debug quantities
      use_integrated_friction (optional) -- use built-in friction model (faster)
    '''    
    # Interpolate solution for time-dependent quantities
    if is_r_pre_eval:
      r = r_interpolants
    else:
      r = r_interpolants(x)
    if len(r.shape) > 1:
      # Shape check
      raise ValueError("Data has more than 1 axis. This function is only for scalar inputs.")
    # Unpack quantities
    p         = q2[0]
    u         = q2[1]
    yWv       = r[1]
    yWt       = r[2]
    yC        = r[3]
    yF        = r[4]
    # Extract spatial derivative of rho prime
    # dyw_dt        = dyw_dt_interpolant(t)
    '''Compute aliases'''
    yM = 1.0 - yWv
    yWd = yWt - yWv
    yL = (yM - (yC + yWd))

    '''Compute EOS quantities'''
    v = yWv * self.vWv_p(p) + yM * self.vM_p(p)
    rho = 1.0 / v
    _, ptilde = self.p_ptilde(rho, yWv)
    # Compute mixture isothermal sound speed, squared
    cT2 = self.isothermal_sound_speed_squared(yWv, p, ptilde)
    # Compute partial of pressure w.r.t. water mass fraction (const rho, T)
    dp_dyw = self.dp_dyw(rho, p, ptilde)

    # Compute solubility based on pure silicate melt mass fraction
    p_like = max(p, 0)

    if self.solubility_n == 0.5:
      yHat = min(max(self.solubility_k * np.sqrt(p_like) * yL, 0), yWt)
    else:
      yHat = np.clip(self.solubility_k * p_like ** self.solubility_n * yL, 0, yWt)
  
    # Compute sources
    source_yWv = (yWd - yHat) / (self.tau_d)

    # Compute momentum load
    if use_integrated_friction:
      # Calculate pure melt viscosity (Hess & Dingwell 1996)
      mfWd = yWd / yL # mass concentration of dissolved water: note model is
      # maximized for a small number, diverging at 0
      mfWd = max(mfWd, 1e-8)
      log_mfWd = np.log(mfWd*100)
      log10_vis = -3.545 + 0.833 * log_mfWd
      log10_vis += (9601 - 2368 * log_mfWd) / (self.T0 - 195.7 - 32.25 * log_mfWd)
      # Melt viscosity with overflow protection
      meltVisc = 10**min(log10_vis, 300)
      # Calculate relative viscosity due to crystals (Costa 2005).
      alpha = 0.999916
      phi_cr = 0.673
      gamma = 3.98937
      delta = 16.9386
      B = 2.5
      # Compute volume fraction of crystal at equal phasic densities
      # Using crystal volume per (melt + crystal + dissolved water) volume
      phi_ratio = max((yC / yM) / phi_cr, 0.0)
      erf_term = scipy.special.erf(
        np.sqrt(np.pi) / (2 * alpha) * phi_ratio * (1 + phi_ratio**gamma))
      crysVisc = (1 + phi_ratio**delta) * ((1 - alpha * erf_term)**(-B * phi_cr))
      mu = meltVisc * crysVisc
    else:
      mu = self.F_fric_viscosity_model(self.T0, yWv, yF, yWt=yWt, yC=yC)
    
    frag_factor = min(max(1.0 - yF/yM, 0.0), 1.0)
    F_fric = 8.0 * mu / (self.conduit_radius * self.conduit_radius) * u \
      * frag_factor
    source_momentum = (-rho * 9.8 - F_fric) / rho

    # Debug output
    if not is_debug_mode:
      # Compute explicitly inverted LHS matrix
      _det = u*u - cT2
      b0 = dp_dyw * source_yWv
      b1 = source_momentum
      return np.array([(u * b0 + -rho * cT2 * b1) / _det,
                      (-v * b0 + u * b1) / _det
                      ])
    
    #   for state vector [rho_bar, u]
    Ainv = np.array([[u, -rho*cT2], [-v, u]]) / (u*u - cT2)
    # Compute RHS vector
    b = np.stack([dp_dyw * source_yWv,
                  + source_momentum,
                  ], axis=0)
    return Ainv, b, Ainv @ b
    
  def _primal_RHS(self, x, q2:np.array, q5:np.array, ddx_q5:np.array,
                 use_integrated_friction=True, _M2_as_u=True):
    ''' Point evaluation of RHS for time-independent portion of the system,
    as the (p, u) system. If _M2_as_u==False, uses (p,M2) system.
    Vectorized for several q2 inputs (faster numerical Jacobian) but single
    q5 input.
    Inputs:
      x -- independent coordinate
      q2 -- state vector of size 2 (p, u)
      r_interpolants -- interpolation for slow-time-dependent variables
      is_debug_mode (optional) -- flag for returning debug quantities
      use_integrated_friction (optional) -- use built-in friction model (faster)
    '''

    # Unpack quantities
    p         = q2[0,:]
    M2        = q2[1,:] # replaced with u if flagged TODO: clean up M2/u
    yWv       = q5[1]
    yWt       = q5[2]
    yC        = q5[3]
    yF        = q5[4]
    # Extract spatial derivative of rho prime
    # dyw_dt        = dyw_dt_interpolant(t)
    '''Compute aliases'''
    yM = 1.0 - yWv
    yWd = yWt - yWv
    yL = (yM - (yC + yWd))

    '''Compute EOS quantities'''
    v = yWv * self.vWv_p(p) + yM * self.vM_p(p)
    rho = 1.0 / v
    _, ptilde = self.p_ptilde(rho, yWv)
    # Compute mixture isothermal sound speed, squared
    cT2 = self.isothermal_sound_speed_squared(yWv, p, ptilde)
    if _M2_as_u:
      u = M2
      M2 = u*u / cT2
    else:
      # Take positive velocity
      u = np.sqrt(M2 * cT2)
    # Compute partial of pressure w.r.t. water mass fraction (const rho, T)
    dp_dyw = self.dp_dyw(rho, p, ptilde)
    d2rho_dp2 = -self.ddp_cT2(yWv, yWt, yC, p) / (cT2*cT2)

    # Compute solubility based on pure silicate melt mass fraction
    p_like = np.maximum(p, 0)

    if self.solubility_n == 0.5:
      yHat = np.clip(self.solubility_k * np.sqrt(p_like) * yL, 0, yWt)
    else:
      yHat = np.clip(self.solubility_k * p_like ** self.solubility_n * yL, 0, yWt)
  
    # Compute sources
    source_yWv = (yWd - yHat) / (self.tau_d)

    # Compute momentum load
    if use_integrated_friction:
      # Calculate pure melt viscosity (Hess & Dingwell 1996)
      mfWd = yWd / yL # mass concentration of dissolved water: note model is
      # maximized for a small number, diverging at 0
      mfWd = np.maximum(mfWd, 1e-8)
      log_mfWd = np.log(mfWd*100)
      log10_vis = -3.545 + 0.833 * log_mfWd
      log10_vis += (9601 - 2368 * log_mfWd) / (self.T0 - 195.7 - 32.25 * log_mfWd)
      # Melt viscosity with overflow protection
      meltVisc = 10**np.minimum(log10_vis, 300)
      # Calculate relative viscosity due to crystals (Costa 2005).
      alpha = 0.999916
      phi_cr = 0.673
      gamma = 3.98937
      delta = 16.9386
      B = 2.5
      # Compute volume fraction of crystal at equal phasic densities
      # Using crystal volume per (melt + crystal + dissolved water) volume
      phi_ratio = np.maximum((yC / yM) / phi_cr, 0.0)
      erf_term = scipy.special.erf(
        np.sqrt(np.pi) / (2 * alpha) * phi_ratio * (1 + phi_ratio**gamma))
      crysVisc = (1 + phi_ratio**delta) * ((1 - alpha * erf_term)**(-B * phi_cr))
      mu = meltVisc * crysVisc
    else:
      mu = self.F_fric_viscosity_model(self.T0, yWv, yF, yWt=yWt, yC=yC)
    
    frag_factor = np.clip(1.0 - yF/yM, 0.0, 1.0)
    F_fric = 8.0 * mu / (self.conduit_radius * self.conduit_radius) * u \
      * frag_factor
    source_momentum = (-rho * 9.8 - F_fric) / rho

    # Compute (p,u) RHS with explicitly inverted LHS matrix
    _det = u*u - cT2
    b0 = dp_dyw * source_yWv
    b1 = source_momentum
    # Implicit stack for vector inputs
    g = np.array([(u * b0 + -rho * cT2 * b1) / _det,
                  (-v * b0 + u * b1) / _det
                 ])
    
    if _M2_as_u:
      return g

    # Lower triangular transformation (p, u) -> (p, M2)
    # _B = np.array([[1, 0], [_B10, _B11]])
    _B10 = u * u * d2rho_dp2
    _B11 = 2 * M2 / u
    # Independent variable chain rule term
    _M2_plus = (self.isothermal_mach_simple(u, yWv + 0.5e-8, p) ** 2)
    _M2_minus = (self.isothermal_mach_simple(u, yWv - 0.5e-8, p) ** 2)
    _dM2_dyWv = (_M2_plus - _M2_minus ) / (1e-8)
    _dyWv_dx = ddx_q5[1]
    # Implicit stack for vector inputs
    f = np.array([g[0], _B10 * g[0] + _B11 * g[1] + _dM2_dyWv * _dyWv_dx])
    return f    

  def primal_var_RHS(self, x:np.array, Q:np.array,
                     q5_interp:callable,
                     use_integrated_friction=True):
    ''' Primal equation RHS and variational equation RHS.
      Uses a finite difference estimation for jacobian of primal equation RHS.
      Input:
        x
        Q: [q, asvector(dq/dp_inlet)]
        q5_interp: vector interpolator for slow (quasi steady state) states '''

    q5 = q5_interp(x)
    # dq5_dx = q5_interp(x, nu=1)

    # Extract data
    q2 = Q[0:2]
    Z = Q[2:4]

    _use_num_diff = False
    if _use_num_diff:
      # Central difference Jacobian estimation parameters
      e1 = np.array([1, 0])
      h1 = 0.1 # Pressure epsilon
      e2 = np.array([0, 1])
      h2 = 1e-6 # M2 epsilon
      # Distribute data for numerical Jacobian evaluation of primal RHS
      # M2 uses positive finite difference
      # IMPORTANT: make sure M_eps is large enough for this if using M2
      q_vec =  np.stack([q2,
                        q2 + 0.5 * h1 * e1,
                        q2 - 0.5 * h1 * e1,
                        q2 + 0.5 * h2 * e2,
                        q2 + h2 * e2], axis=1)
      # Compute primal RHS
      f_vec = self._primal_RHS(x, q_vec, q5, None,
                          use_integrated_friction=use_integrated_friction,
                          _M2_as_u=True)
      dfdp = (f_vec[:,1] - f_vec[:,2]) / h1
      dfdu = (-3 * f_vec[:,0] + 4 * f_vec[:,3] - f_vec[:,4]) / h2
      rhs = f_vec[:,0]
    else:
      _, yWv, yWt, yC, yF = q5
      # Evaluate Jacobian of RHS and RHS
      jac, rhs = self.RHS_jac(Q[0], Q[1],
                                yWv, yWt, yC, yF,
                                also_return_RHS=True)
      # Extract partials from Jacobian (squeeze out axis 0)
      dfdp = jac[0,:,0]
      dfdu = jac[0,:,1]

    # Evaluate matrix-vector product J*Z (aka J @ Z)
    # J = np.array([dfdp, dfdM2]).T
    JdotZ0 = dfdp[0] * Z[0] + dfdu[0] * Z[1]
    JdotZ1 = dfdp[1] * Z[0] + dfdu[1] * Z[1]

    # Evaluate RHS of primal and variational systems  
    if len(Q) == 4:
      # No estimation of d/dp_iterate of L1(p)
      dQ = np.array([*rhs, JdotZ0, JdotZ1])
    else:
      dQ = np.array([*rhs, JdotZ0, JdotZ1, Z[0]])
    return dQ

  def newton_map_pout(self, p_inlet, p_outlet_iterate, xlim, q5_interp,
                      ode_rtol=1e-6, M_eps=1e-2, use_integrated_friction=True):
    ''' ODE solve wrapper for primal_var_RHS. '''

    # Construct primal state
    yWv_outlet = q5_interp(xlim[-1])[1]
    rho_outlet = self.rho_p(yWv_outlet, p_outlet_iterate)
    cT2_outlet = self.isothermal_sound_speed_squared(yWv_outlet, *self.p_ptilde(rho_outlet, yWv_outlet))
    u_outlet = np.sqrt((1-M_eps) * (1-M_eps) * cT2_outlet)
    # Assemble boundary condition of q2 at vent with column of identity and p-integral zero
    # Q_pM2 = np.array([p_outlet_iterate, u_outlet, 1, 0, 0])
    Q_pM2 = np.array([p_outlet_iterate, u_outlet, 1, 0])

    # Flag for debugging by solving the IVP in the forward (upward) direction
    _forward_ivp_debug = False
    if _forward_ivp_debug:
      soln = scipy.integrate.solve_ivp(
        lambda x, Q: self.time_indep_RHS(x,
            Q[:,np.newaxis] * np.array([1, 1])[:,np.newaxis],
            q5_interp(x),
            use_integrated_friction=True,
           is_r_pre_eval=True).ravel(),
        (xlim[0], 1000),
        np.array([40e6, self.u0]),
        # np.array([40e6, 0.0005578129172335377**2]),
        method="BDF",
        dense_output=True,
        max_step=np.inf,
        rtol=1e-7,
        atol=1e-6,)
      soln_orig = scipy.integrate.solve_ivp(
        lambda x, Q: self.time_indep_RHS(x, Q, q5_interp,
          use_integrated_friction=True).ravel(),
        (xlim[0], 1000),
        np.array([p_inlet, self.u0]),
        method="BDF",
        dense_output=True,
        max_step=np.inf,
        rtol=1e-7,
        atol=1e-6,)
      _q5 = np.array([q5_interp(_x) for _x in soln_orig.t]).T
      _M2 = (self.isothermal_mach_simple(soln_orig.y[1,...],
                                         _q5[1:2,...],
                                         soln_orig.y[0,...]) ** 2).ravel()


      soln_primal_var = scipy.integrate.solve_ivp(
        lambda x, Q: self.primal_var_RHS(x, Q, q5_interp,
          use_integrated_friction=use_integrated_friction),
        (xlim[0], 1000),
        np.array([40e6, 0.0005578129172335377**2, 1, 0]),
        method="BDF",
        dense_output=True,
        max_step=np.inf,
        rtol=ode_rtol,
        atol=1e-4,)
      _debug = "this is for breakpoint"

    # Solve from M2 condition down to inlet
    soln = scipy.integrate.solve_ivp(
      lambda x, Q: self.primal_var_RHS(x, Q, q5_interp,
        use_integrated_friction=use_integrated_friction),
      (xlim[-1], xlim[0]),
      Q_pM2,
      method="RK45",
      dense_output=True, # TODO: request only dense output for the last output
      max_step=np.inf,
      rtol=ode_rtol,
      atol=1e-6,)
    # Interpret solution at inlet
    p_in_solved = soln.y[0,-1]
    dpin_dpout = soln.y[2,-1]
    duin_dpout = soln.y[3,-1] # Use for estimating u-sensitivity
    if soln.y.shape[0] >= 5:
      d_dpout_integralp = soln.y[4,-1]
    else:
      d_dpout_integralp = None
    uin = soln.y[1, -1]

    # Old stuff for (p, M2) system -- this needed chain rule into dyWv/dx
    # M2in = soln.y[1,-1]
    # _, yWv, yWt, yC, yF = q5_interp(xlim[0])
    # rho = self.rho_p(yWv, p_in_solved)
    # _, ptilde = self.p_ptilde(rho, yWv)
    # cT2in = self.isothermal_sound_speed_squared(yWv, p_in_solved, ptilde)
    # uin = np.sqrt(M2in * cT2in)
    # Return new iterate, inlet velocity, slope
    residual = (p_in_solved - p_inlet)
    newton_step = - residual / dpin_dpout
    # Estimate change in u_in after a Newton step (recall: want uin > 0)
    uin_change_estimate = newton_step * duin_dpout
    return p_outlet_iterate + newton_step, residual, uin, dpin_dpout, \
      duin_dpout, uin_change_estimate, d_dpout_integralp, soln

  def _time_indep_RHS_explicit(self, x, q2:np.array, r_interpolants:callable,
                     is_debug_mode=False):
    ''' Copy of RHS for time-independent portion of the system.
    Inputs:
      x -- independent coordinate
      q2 -- state vector of size 2 (p, u)
      r_interpolants -- interpolation for slow-time-dependent variables
      is_debug_mode (optional) -- flag for returning debug quantities
    '''    
    # Interpolate solution for time-dependent quantities
    r = r_interpolants(x)
    # Unpack quantities
    p         = q2[0,...]
    u         = q2[1,...]
    rho_prime = r[0,...]
    yWv       = r[1,...]
    yWt       = r[2,...]
    yC        = r[3,...]
    yF        = r[4,...]
    # Extract spatial derivative of rho prime
    # dyw_dt        = dyw_dt_interpolant(t)
    '''Compute aliases'''
    yM = 1.0 - yWv
    yWd = yWt - yWv

    '''Compute EOS quantities'''
    v = yWv * self.vWv_p(p) + yM * self.vM_p(p)
    rho = 1.0 / v
    _, ptilde = self.p_ptilde(rho, yWv)
    # Compute mixture isothermal sound speed, squared
    cT2 = self.isothermal_sound_speed_squared(yWv, p, ptilde)
    # Compute partial of pressure w.r.t. water mass fraction (const rho, T)
    dp_dyw = self.dp_dyw(rho, p, ptilde)

    # Compute solubility based on pure silicate melt mass fraction
    yHat = self.yWd_target(yWv, yWt, yC, p)
    # Compute sources
    source_yWv = (yWd - yHat) / (self.tau_d)

    # Compute momentum load
    mu = self.F_fric_viscosity_model(self.T0, yWv, yF, yWt=yWt, yC=yC)
    F_fric = 8.0 * mu / (self.conduit_radius * self.conduit_radius) * u \
      * np.clip(1.0 - yF/yM, 0.0, 1.0)
    source_momentum = (-rho * 9.8 - F_fric) / rho
    # Compute explicitly inverted LHS matrix
    #   for state vector [rho_bar, u]
    Ainv = np.array([[u, -rho*cT2], [-v, u]]) / (u*u - cT2)
    # Compute RHS vector
    b = np.stack([dp_dyw * source_yWv,
                  + source_momentum,
                  ], axis=0)
    
    if np.linalg.cond(Ainv) > 1e10:
      # print("Warning: Ainv ill-conditioned")
      pass

    if is_debug_mode:
      return Ainv, b
    return Ainv @ b
  
  def time_indep_RHS_eq(self, x, q_pu:np.array, q_t_interpolants:callable,
                     is_debug_mode=False):
    ''' RHS for time-independent portion of the system.
    Inputs:
      x -- independent coordinate
      q_pu -- state vector of size 2 (p, u)
      q_t_interpolants -- interpolation for slow-time-dependent variables
    '''
    # Interpolate solution for time-dependent quantities
    q_t = q_t_interpolants(x)
    # Unpack quantities
    p         = q_pu[0,...]
    u         = q_pu[1,...]
    yWt       = q_t[0,...]
    yC        = q_t[1,...]

    '''Compute intermediates '''
    yL = (1.0 - (yC + yWt))
    p_like = np.clip(p, 0, None)
    yWd = np.clip(self.solubility_k * p_like ** self.solubility_n * yL, 0, yWt)
    yWv = yWt - yWd
    rho = self.rho_p(yWv, p)
    yM = 1.0 - yWv
    _, ptilde = self.p_ptilde(rho, yWv)
    vf_g = self.volfrac_gas(rho, yWv, p)
    # Compute mixture isothermal sound speed, squared
    # cT2 = self.isothermal_sound_speed_squared(yWv, p, ptilde)
    cT2 = self.isothermal_eq_sound_speed_squared(yWt, yC, p)

    # Compute fragmentation state (two-way fragmentation)
    yF  = yM * np.asarray(vf_g - self.vf_g_crit > 0.0).astype(float)
    # Compute friction
    mu = self.F_fric_viscosity_model(self.T0, yWv, yF, yWt=yWt, yC=yC)
    F_fric = 8.0 * mu / (self.conduit_radius * self.conduit_radius) \
             * u * np.clip(1.0 - yF/yM, 0.0, 1.0)
    source_momentum = (-rho * 9.8 - F_fric) / rho

    # Manual inversion
    Ainvb = np.zeros((2, *np.asarray(u).shape))
    Ainvb[0,...] = source_momentum * (-rho*cT2) / (u*u - cT2)
    Ainvb[1,...] = source_momentum * (u) / (u*u - cT2)  

    return Ainvb

  def time_indep_RHS_legacy(self, t, q2:np.array, r_interpolants:callable,
                     dyw_dt_interpolant:np.array, is_debug_mode=False):
    ''' Legacy variant with splti rho_bar and rho_prime
    RHS for time-independent portion of the system.
    Inputs:
      x -- independent coordinate
      q2 -- state vector of size 2 (rho0, u)
      r_interpolants -- interpolation for slow-time-dependent variables
      dyw_dt -- values of time-derivative of yWv
      is_debug_mode (optional) -- flag for returning debug quantities
    '''    
    # Interpolate solution for time-dependent quantities
    r = r_interpolants(t)
    # Unpack quantities
    rho_bar   = q2[0,...]
    u         = q2[1,...]
    rho_prime = r[0,...]
    yWv       = r[1,...]
    yWt       = r[2,...]
    yC        = r[3,...]
    yF        = r[4,...]
    # Extract spatial derivative of rho prime
    drho_prime_dx = r_interpolants.derivative()(t)[0]
    dyw_dx        = r_interpolants.derivative()(t)[1]
    dyw_dt        = dyw_dt_interpolant(t)
    '''Compute aliases'''
    yM = 1.0 - yWv
    rho = rho_bar + rho_prime
    '''Compute EOS quantities'''
    p, ptilde = self.p_ptilde(rho, yWv)
    # Compute mixture isothermal sound speed, squared
    cT2 = self.isothermal_sound_speed_squared(yWv, p, ptilde)
    # Compute partial of pressure w.r.t. water mass fraction (const rho, T)
    dp_dyw = self.dp_dyw(rho, p, ptilde)
    # Compute partial of density w.r.t. water mass fraction (const p, T)
    drho_dyw = self.drho_dyw(rho, p)

    # Compute momentum load
    mu = self.F_fric_viscosity_model(self.T0, yWv, yF, yWt=yWt, yC=yC)
    F_fric = 8.0 * mu / (self.conduit_radius * self.conduit_radius) * u \
      * np.clip(1.0 - yF/yM, 0.0, 1.0)
    source_momentum = (-rho * 9.8 - F_fric) / rho
    # Compute explicitly inverted LHS matrix
    #   for state vector [rho_bar, u]
    Ainv = np.array([[u, -rho], [-cT2/rho, u]]) / (u*u - cT2)
    # Compute RHS vector
    b = np.stack([-u * drho_prime_dx + drho_dyw * dyw_dt,
                  -cT2 / rho * drho_prime_dx
                    - 1.0 / rho * dp_dyw * dyw_dx
                    + source_momentum,
                  ], axis=0)
    if is_debug_mode:
      return Ainv @ b
    return Ainv @ b

  def solve_time_indep_system(self,
                              r_interpolants:callable,
                              p0=80e6,
                              u0=0.7,
                              xlims:tuple=(0, 5000),
                              method="BDF",
                              max_step=np.inf,
                              M_eps=1e-2) -> object:
    # Create scaling for [p, u]
    scales = np.array([10e6, 1.0])
    # Create choking event
    def event_isothermal_choked(x, q2_scaled) -> np.array:
      ''' Returns isothermal mach squared '''
      p = q2_scaled[0,...] * scales[0]
      u = q2_scaled[1,...] * scales[1]
      yWv = r_interpolants(x)[1,...]
      rho = self.rho_p(yWv, p)
      ptilde = self.p_ptilde(rho, yWv)[1]
      return self.isothermal_mach_squared(u, yWv, p, ptilde) - (1.0 - M_eps) * (1.0 - M_eps)
    event_isothermal_choked.terminal = True
    # event_isothermal_choked.direction = 1.0
    # Solve IVP
    soln = scipy.integrate.solve_ivp(
      lambda x, q_scaled: self.time_indep_RHS(
        x, q_scaled * scales, r_interpolants, is_debug_mode=False) / scales,
      xlims,
      np.array([p0, float(u0)]) / scales,
      method=method,
      dense_output=True,
      max_step=max_step,
      rtol=1e-8,
      atol=1e-8,
      events=[event_isothermal_choked,])
    return soln, scales

  def solve_time_indep_system_eq(self,
                              r_interpolants:callable,
                              p0=80e6,
                              u0=0.7,
                              xlims:tuple=(0, 5000),
                              method="BDF",
                              max_step=np.inf,
                              M_eps=1e-2) -> object:
    # Create scaling for [p, u]
    scales = np.array([1.0, 1.0])
    # Create choking event
    def event_isothermal_choked(x, q2_scaled) -> np.array:
      ''' Returns isothermal mach squared '''
      p = q2_scaled[0,...] * scales[0]
      u = q2_scaled[1,...] * scales[1]
      yWt = r_interpolants(x)[0,...]
      yC = r_interpolants(x)[1,...]
      yL = (1.0 - (yC + yWt))
      p_like = np.clip(p, 0, None)
      yWd = np.clip(self.solubility_k * p_like ** self.solubility_n * yL, 0, yWt)
      yWv = yWt - yWd
      rho = self.rho_p(yWv, p)
      ptilde = self.p_ptilde(rho, yWv)[1]

      # return self.isothermal_mach_squared(u, yWv, p, ptilde) - (1.0 - M_eps) * (1.0 - M_eps)       
      return (u*u) / self.isothermal_eq_sound_speed_squared(yWt, yC, p) - (1.0 - M_eps) * (1.0 - M_eps)
    event_isothermal_choked.terminal = True
    # event_isothermal_choked.direction = 1.0
    # Solve IVP
    soln = scipy.integrate.solve_ivp(
      lambda x, q_scaled: self.time_indep_RHS_eq(
        x, q_scaled * scales, r_interpolants, is_debug_mode=False) / scales,
      xlims,
      np.array([p0, float(u0)]) / scales,
      method=method,
      dense_output=True,
      max_step=max_step,
      vectorized=False,
      rtol=1e-6,
      atol=1e-8,
      events=[event_isothermal_choked,])
    return soln, scales
  
  def solve_steady_choked(self, xlims, p0=80e6, yWt0=0.025,
                 yC0=0.4, yF0=0.0):
    ''' Convenience function for solving choked BC. Not fully featured. '''
    xlims_unlimited = (0, 10000)
    def solve_full_ivp_L(u0_iterate):
      ''' Full steady state problem, returning soln bunch and conduit length'''
      soln = self.solve_full_steady_state(
        p0=p0,
        u0=u0_iterate,
        yWt0=yWt0,
        yC0=yC0,
        yF0=yF0,
        xlims=xlims_unlimited)
      return soln, soln.t.max()
    # Shooting method for matching conduit length
    u0_solved, self._brentq_results = scipy.optimize.brentq(
      lambda u0: solve_full_ivp_L(u0)[1] - xlims[-1],
      0.1, 15, full_output=True)
    # Solve using u0 from shooting method
    soln = self.solve_full_steady_state(
      p0=p0,
      u0=u0_solved,
      yWt0=yWt0,
      yC0=yC0,
      yF0=yF0,
      xlims=xlims_unlimited)
    return soln
  
  def solve_steady_choked_eq(self, xlims, p0=80e6, yWt0=0.025, yC0=0.4):
    ''' Convenience function for solving choked BC. Not fully featured. '''
    xlims_unlimited = (0, 10000)
    def solve_full_ivp_L(u0_iterate):
      ''' Full steady state problem, returning soln bunch and conduit length'''
      soln = self.solve_full_steady_state_eq(
        p0=p0,
        u0=u0_iterate,
        yWt0=yWt0,
        yC0=yC0,
        xlims=xlims_unlimited)
      return soln, soln.t.max()
    # Shooting method for matching conduit length
    u0_solved, self._brentq_results = scipy.optimize.brentq(
      lambda u0: solve_full_ivp_L(u0)[1] - xlims[-1],
      0.1, 15, full_output=True)
    # Solve using u0 from shooting method
    soln = self.solve_full_steady_state_eq(
      p0=p0,
      u0=u0_solved,
      yWt0=yWt0,
      yC0=yC0,
      xlims=xlims_unlimited)
    return soln

  @staticmethod
  def slowness(self, soln:object) -> object:
    return scipy.integrate.solve_ivp(
      lambda t, x: 1.0 / soln.sol(x)[1,...],
      (soln.t.min(), soln.t.max()),
      np.array([0.0]),
      dense_output=True)

  @staticmethod
  def diff_op1(x) -> scipy.sparse.spmatrix:
    ''' First-order accurate differentiation matrix. '''
    Nx = x.size
    _main_diag = np.ones((Nx,))
    _main_diag[0] = 0.0
    _sub_diag = -np.ones((Nx-1,))
    _D = scipy.sparse.diags((_main_diag, _sub_diag,), (0,-1))
    return _D    

  def BC_steady(self, t, p0=80e6, u0=0.7, yWt0=0.025, yC0=0.4,
                yF0=0.0,):
    # Compute phase mass fractions from solubility
    xi = self.solubility_k * p0 ** self.solubility_n
    yWd0 = np.clip(xi *(1.0 - (yWt0 + yC0)), 0, yWt0)
    yWv0 = yWt0 - yWd0
    yM0 = 1.0 - yWv0
    # Compute densities
    rhoM0 = self.rho_m0 * (1 + (p0 - self.p_m0) / self.K)
    vM0 = 1.0 / rhoM0
    vWv0 = self.R_wv * self.T0 / p0
    v0 = yM0 * vM0 + yWv0 * vWv0
    rho0 = 1.0 / v0

    return np.stack((
      rho0 * np.ones_like(t),
      0 * np.ones_like(t),   # dummy data
      0.0 * np.ones_like(t), # rho_prime0 -- dependent on yWv0(t),
      yWv0 * np.ones_like(t),
      yWt0 * np.ones_like(t),
      yC0 * np.ones_like(t),
      yF0 * np.ones_like(t),
    ), axis=0)
  
  def BC_gaussian(self, t, p0=80e6, u0=0.7, yWt0=0.025, yC0=0.4,
                  yF0=0.0,):    
    # Gaussian parameters
    amp = 0.05
    t0 = 30
    sigma = 15
    # Compute perturbation
    yWt_perturbation = 0.0
    _arg = ((t - t0) / sigma)
    yC_perturbation = amp * np.exp(-_arg * _arg) / np.exp(0)
    # Add perturbation to steady boundary data
    q7 = self.BC_steady(t, p0=p0, u0=u0, yWt0=yWt0, yC0=yC0, yF0=yF0,)
    q7[4,...] += yWt_perturbation
    q7[5,...] += yC_perturbation
    return q7
  
  def BC_gaussian_crossverif(self, t, p0=40e6, u0=0.7, yWt0=0.01747572815, yC0=0.4, yF0=0.0,):
    ''' Non-equilibrium gaussian for comparison with compressible code '''
    # Gaussian parameters
    amp = 0.1
    t0 = 40   # "gaussian_tpulse"
    sigma = 8
    chi_water = 0.03
    # Compute Gaussian perturbation
    _arg = ((t - t0) / sigma)
    # Important: note /2 factor
    yC_perturbation = amp * np.exp(-_arg * _arg / 2) #/ np.exp(0)
    # Water part:
    phi_crys = yC0 * np.ones_like(t) + yC_perturbation
    yWt = chi_water * (1.0 - phi_crys) / (1 + chi_water)
    # Add perturbation to steady boundary data
    q7 = self.BC_steady(t, p0=p0, u0=u0, yWt0=yWt0, yC0=yC0, yF0=yF0,)
    q7[4,...] = yWt
    q7[5,...] = phi_crys
    return q7
  
  def BC_steady_eq(self, t, p0=80e6, u0=0.7, yWt0=0.025, yC0=0.4,):    
    return np.stack((
      p0 * np.ones_like(t),
      u0 * np.ones_like(t),
      yWt0 * np.ones_like(t),
      yC0 * np.ones_like(t),
    ), axis=0)

  def BC_gaussian_eq(self, t, p0=80e6, u0=0.7, yWt0=0.025, yC0=0.4,):    
    # Gaussian parameters
    amp = 0.05
    t0 = 30
    sigma = 15
    # Compute perturbation
    yWt_perturbation = 0.0
    _arg = ((t - t0) / sigma)
    yC_perturbation = amp * np.exp(-_arg * _arg) / np.exp(0)
    # Add perturbation to steady boundary data
    q = np.stack((
      p0 * np.ones_like(t),
      u0 * np.ones_like(t),
      yWt0 * np.ones_like(t) + yWt_perturbation,
      yC0 * np.ones_like(t) + yC_perturbation,
    ), axis=0)
    return q
  
  def BC_gaussian_eq_crossverif(self, t, p0=40e6, u0=0.7, yWt0=0.01747572815, yC0=0.4,):
    # Gaussian parameters
    amp = 0.1
    t0 = 40   # "gaussian_tpulse"
    sigma = 8
    chi_water = 0.03
    # Compute Gaussian perturbation
    _arg = ((t - t0) / sigma)
    # Important: note /2 factor
    yC_perturbation = amp * np.exp(-_arg * _arg / 2) / np.exp(0)
    # Water part:
    phi_crys = yC0 * np.ones_like(t) + yC_perturbation
    yWt = chi_water * (1.0 - phi_crys) / (1 + chi_water)
    # Add perturbation to steady boundary data
    q = np.stack((
      p0 * np.ones_like(t), # Dummy data
      u0 * np.ones_like(t), # Dummy data
      yWt,
      phi_crys,
    ), axis=0)
    return q

  def BC_sine_eq(self, t, p0=80e6, u0=0.7, yWt0=0.025, yC0=0.4,):    
    # Gaussian parameters
    amp = 0.05
    t0 = 0
    period = 30
    # Compute perturbation
    yWt_perturbation = 0.0
    _arg = (2 * np.pi * (t - t0) / period)
    yC_perturbation = amp * 0.5 * (1.0 - np.cos(_arg))
    # Add perturbation to steady boundary data
    q = np.stack((
      p0 * np.ones_like(t),
      u0 * np.ones_like(t),
      yWt0 * np.ones_like(t) + yWt_perturbation,
      yC0 * np.ones_like(t) + yC_perturbation,
    ), axis=0)
    return q
  
  def BC_sine_box_eq(self, t, p0=80e6, u0=0.7, yWt0=0.025, yC0=0.4,):    
    # Gaussian parameters
    amp = 0.05
    t0 = 0
    period = 30
    N_periods = 10
    # Compute perturbation
    yWt_perturbation = 0.0
    # Clipped phase
    _arg = np.clip(2 * np.pi * (t - t0) / period, None, N_periods * 2 * np.pi)
    yC_perturbation = amp * 0.5 * (1.0 - np.cos(_arg))
    # Add perturbation to steady boundary data
    q = np.stack((
      p0 * np.ones_like(t),
      u0 * np.ones_like(t),
      yWt0 * np.ones_like(t) + yWt_perturbation,
      yC0 * np.ones_like(t) + yC_perturbation,
    ), axis=0)
    return q

  def advect_step(self, BC_fn, x, t, dt, q5_interp, q2_interp, u0):
    ''' Semi-Lagrangian advection. '''
    # Compute advection foot point (x - ut) using RK2
    # x1 = x - dt * q2_interp(x)[1,...]
    # x2 = x - dt * q2_interp(x1)[1,...]
    # x_origin = 0.5 * (x1 + x2)
    # Compute advection foot point (x - ut) using RK4
    k1 = -q2_interp(x)[1,...]
    k2 = -q2_interp(x + 0.5*dt*k1)[1,...]
    k3 = -q2_interp(x + 0.5*dt*k2)[1,...]
    k4 = -q2_interp(x + dt*k3)[1,...]
    x_origin = x + dt * (k1 + 2*k2 + 2*k3 + k4)/6.0

    # Compute value of q5 at source
    return np.where(x_origin >= 0,
                    q5_interp(x_origin), # Interpolate q5 info at foot point
                    BC_fn(t + (0 - x_origin) / u0)[2:,...] # Source from BC at t + eps
                   )
  
  def advect_step_eq(self, BC_fn, x, t, dt, q_t_interp, q_pu_interp, u0):
    ''' Semi-Lagrangian advection. '''
    # Compute advection foot point (x - ut) using RK2
    x1 = x - dt * q_pu_interp(x)[1,...]
    x2 = x - dt * q_pu_interp(x1)[1,...]
    x_origin = 0.5 * (x1 + x2)
    # Compute value of q5 at source
    return np.where(x_origin >= 0,
                    q_t_interp(x_origin), # Interpolate q5 info at foot point
                    BC_fn(t + (0 - x_origin) / u0)[2:,...] # Source from BC at t + eps
                   )
  
  def react_step(self, x, t, dt, q5, q2):
    ''' Implicit or analytic update to reaction step. '''
    p   = q2[0,...]
    u   = q2[1,...]
    rhoprime = q5[0,...] # unused
    yWv = q5[1,...]
    yWt = q5[2,...]
    yC  = q5[3,...]
    yF  = q5[4,...]

    # Compute dependents
    yM = 1.0 - yWv
    yWd = yWt - yWv
    yL = (yM - (yC + yWd))
    yWdHat = np.clip(self.solubility_k * p ** self.solubility_n * yL, 0, yWt)
    yWvHat = yWt - yWdHat
    rho = self.rho_p(yWv, p)
    vf_g = self.volfrac_gas(rho, yWv, p)
    
    # Approximate fragmentation solution
    #   Fragmentation solution depends on yWv(t) within the step since
    #   vol frac depends on yWv(t). Here we approximate the fragmentation
    #   criterion using the fragmentation criterion at the begining of the
    #   split step.
    # For a fragmenting location, compute coefficients to yF solution
    if False:
      tau_f_effective = self.tau_f / self.smoother(vf_g - self.vf_g_crit)
      _taud_coeff = (yWvHat - yWv) / (tau_f_effective / self.tau_d - 1.0) # Warning:resonant case
      _const_term = 1.0 - yWvHat
      _tauf_coeff = yF - (_taud_coeff + _const_term)
      # One-sided smoothing for fragmentation source
      # source_yF  = (yM - yF) / (self.tau_f) * self.smoother(vf_g - self.vf_g_crit)
      yF_updated = np.where(vf_g > self.vf_g_crit,
        _taud_coeff * np.exp(-dt / self.tau_d)
          + _const_term
          + _tauf_coeff * np.exp(-dt / self.tau_f * self.smoother(vf_g - self.vf_g_crit)),
        yF)
    # RK4 substep strategy (for largest A-stability region)
    _coeff = self.smoother(vf_g - self.vf_g_crit) / self.tau_f
    # Frozen-pressure coefficient TODO: update smoother(...) in between due to yWv, although p is not updated
    yM1 = 1.0 - yWv
    k1 = (yM1 - yF) * _coeff
    yWv2 = yWv + (yWvHat - yWv) * (1 - np.exp(-0.5 * dt / self.tau_d))
    yM2 = 1.0 - yWv2
    k2 = (yM2 - (yF + 0.5 * dt * k1)) * _coeff
    k3 = (yM2 - (yF + 0.5 * dt * k2)) * _coeff
    yWv4 = yWv + (yWvHat - yWv) * (1 - np.exp(-dt / self.tau_d))
    yM4 = 1.0 - yWv4
    k4 = (yM4 - (yF + dt * k3)) * _coeff
    yF_updated = yF + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    
    # Evolution operator[dt] within the split-step
    y_updated = np.stack([rhoprime,
                          yWv + (yWvHat - yWv) * (1 - np.exp(-dt / self.tau_d)),
                          yWt,
                          yC,
                          yF_updated,
                          ], axis=0)
    return y_updated

  def advect_react_step(self, BC_fn, x, t, dt, q5_interp, q2_interp, p0, u0):
    ''' Mixed step: RK4 on outside. Reaction applied to control mass found at
     footpoint of advection, subject to local pressure conditions. '''

    ''' Compute advection path nodes
      2xRK4 to compute two points on advection path
      TODO: boundary extrapolation
    '''
    # x(t + dt) -> x(t + dt/2)
    k1 = -q2_interp(x)[1,...]
    k2 = -q2_interp(x + 0.25*dt*k1)[1,...]
    k3 = -q2_interp(x + 0.25*dt*k2)[1,...]
    k4 = -q2_interp(x + 0.5*dt*k3)[1,...]
    x1 = x + (0.5*dt) * (k1 + 2*k2 + 2*k3 + k4)/6.0
    # x(t + dt/2) -> x(t)
    k1 = -q2_interp(x1)[1,...]
    k2 = -q2_interp(x1 + 0.25*dt*k1)[1,...]
    k3 = -q2_interp(x1 + 0.25*dt*k2)[1,...]
    k4 = -q2_interp(x1 + 0.5*dt*k3)[1,...]
    x2 = x1 + (0.5*dt) * (k1 + 2*k2 + 2*k3 + k4)/6.0

    ''' Integrate reaction along advection path'''
    # RK4 t -> t + dt
    # Extraction
    def q2_at(x):
      _q = q2_interp(x)
      out_of_bounds_values = np.zeros_like(_q)
      out_of_bounds_values[0,:] = p0
      out_of_bounds_values[1,:] = u0
      return np.where(x >= 0,
                      _q,
                      out_of_bounds_values)
      # return np.where(x >= 0, # TODO: replace with xlims[0] consistenly
      #   q2_interp(x),         # Interpolate q5 info at foot point
      #   np.array([[p0], [u0]]))   # Source from BC at t + eps

    # p   = q2[0,...]
    # u   = q2[1,...]
    # Compute value of q5 at source
    q5 = np.where(x2 >= 0, # TODO: replace with xlims[0] consistenly
      q5_interp(x2), # Interpolate q5 info at foot point
      BC_fn(t + (0 - x2) / u0)[2:,...]) # Source from BC at t + eps
    rhoprime = q5[0,...] # unused
    yWv = q5[1,...]
    yWt = q5[2,...] # const
    yC  = q5[3,...] # const
    yF  = q5[4,...]
    # Compute dependents
    yL = (1.0 - (yC + yWt))

    _use_exp3 = False
    if _use_exp3:
      # Use 3rd order exponential Rosenbrock (Hochbruck et al. 2009)
      p0 = q2_at(x)[0,...]

      q_loc = q5[[1,4],...].copy()
      q_pred = q_loc.copy()
      q_updated = q_loc.copy()

      def RHS(yWv, yF, p, yWt, yL):
        ''' Wrap right-hand side in vector form -> shape (2,...)'''
        rho = self.rho_p(yWv, p)
        yM = 1.0 - yWv
        vf_g = self.volfrac_gas(rho, yWv, p)
        yWdHat = np.clip(self.solubility_k * p ** self.solubility_n * yL, 0, yWt)
        rate_yWv = (yWt - yWdHat - yWv) / self.tau_d
        rate_yF = (yM - yF) * self.smoother(vf_g - self.vf_g_crit) / self.tau_f
        return np.stack((rate_yWv, rate_yF), axis=0)
      
      # exp3 step 1 with p at location x(t) with exponential substepping
      p1 = q2_at(x2)[0,...]
      for i in range(q_loc.shape[-1]): # for loop to do expm for EACH node
        # Compute Jacobian at beginning of timestep
        Jn = self.jac(yWv[i], yF[i], p1[i])
        # Compute RHS at beginning of timestep
        RHSn = RHS(yWv[i], yF[i], p1[i], yWt[i], yL[i])
        # Predictor step
        q_pred[...,i] = q_loc[...,i] + dt * self.phirm(dt * Jn, RHSn)[1].ravel()
        # Compute difference in nonlinear part of RHS TODO: using p(t+dt). is p1 needed?
        Ndiff = RHS(q_pred[0,i], q_pred[1,i], p0[i], yWt[i], yL[i]) - RHSn \
                - np.einsum("ij, j -> i", Jn, (q_pred[...,i] - q_loc[...,i]))
        # Compute corrector (also error estimator)
        corr = dt * 2.0 * self.phirm(dt * Jn, Ndiff)[3]
        # Update using corrector
        q_updated[...,i] = q_pred[...,i] + corr

      # Unpack updated quantities
      yWv_updated, yF_updated = q_updated
      
    else:
      # RK4 substep strategy (for largest A-stability region)
      # RK4 step 1 with p at location x(t) with exponential substepping
      p = q2_at(x2)[0,...]
      rho = self.rho_p(yWv, p)
      yM = 1.0 - yWv
      vf_g = self.volfrac_gas(rho, yWv, p)
      # Update yWv
      yWdHat = np.clip(self.solubility_k * p ** self.solubility_n * yL, 0, yWt)
      rate_yWv1 = (yWt - yWdHat - yWv) / self.tau_d
      # yWv1 = yWv + (yWt - yWdHat - yWv) * (1 - np.exp(-(0.5*dt) / self.tau_d))
      rate_yF1 = (yM - yF) * self.smoother(vf_g - self.vf_g_crit) / self.tau_f
      yWv1 = yWv + (0.5 * dt) * rate_yWv1
      yF1 = yF + (0.5 * dt) * rate_yF1
      
      # RK4 step 2
      p = q2_at(x1)[0,...]
      rho = self.rho_p(yWv1, p)
      yM = 1.0 - yWv1
      vf_g = self.volfrac_gas(rho, yWv1, p)
      # Update yWv
      yWdHat = np.clip(self.solubility_k * p ** self.solubility_n * yL, 0, yWt)
      rate_yWv2 = (yWt - yWdHat - yWv1) / self.tau_d
      rate_yF2 = (yM - yF1) * self.smoother(vf_g - self.vf_g_crit) / self.tau_f
      yWv2 = yWv + (0.5 * dt) * rate_yWv2
      yF2 = yF + (0.5 * dt) * rate_yF2

      # RK4 step 3, keeping p at x1
      rho = self.rho_p(yWv2, p)
      yM = 1.0 - yWv2
      vf_g = self.volfrac_gas(rho, yWv2, p)
      # Update yWv
      yWdHat = np.clip(self.solubility_k * p ** self.solubility_n * yL, 0, yWt)
      rate_yWv3 = (yWt - yWdHat - yWv2) / self.tau_d
      rate_yF3 = (yM - yF2) * self.smoother(vf_g - self.vf_g_crit) / self.tau_f
      yWv3 = yWv + dt * rate_yWv3
      yF3 = yF + dt * rate_yF3

      # RK4 step 4
      p = q2_at(x)[0,...]
      rho = self.rho_p(yWv3, p)
      yM = 1.0 - yWv3
      vf_g = self.volfrac_gas(rho, yWv3, p)
      # Update yWv
      yWdHat = np.clip(self.solubility_k * p ** self.solubility_n * yL, 0, yWt)
      rate_yWv4 = (yWt - yWdHat - yWv3) / self.tau_d
      rate_yF4 = (yM - yF3) * self.smoother(vf_g - self.vf_g_crit) / self.tau_f
      
      yWv_updated = yWv + dt * (rate_yWv1 + 2.0*rate_yWv2 + 2.0*rate_yWv3 + rate_yWv4) / 6.0
      yF_updated = yF + dt * (rate_yF1 + 2.0*rate_yF2 + 2.0*rate_yF3 + rate_yF4) / 6.0

    
    # Evolution operator[dt] within the split-step
    y_updated = np.stack([rhoprime,
                          yWv_updated,
                          yWt,
                          yC,
                          yF_updated,
                          ], axis=0)
    return y_updated
  
  def unsteady_solve_RHS(self, BC_fn, x, t, q5:np.array, q2:np.array,
                         is_debug_mode=False) -> object:
    ''' Full gridded RHS for d/dt consistency.
     Returns advection and reaction sources separately. '''

    # Upwinded dx, repeating first element
    dx = np.concatenate([[x[1] - x[0]], x[1:] - x[:-1]])
    # Unpack ODE vector
    p         = q2[0,...]
    u         = q2[1,...]
    _         = q5[0,...]
    yWv       = q5[1,...]
    yWt       = q5[2,...]
    yC        = q5[3,...]
    yF        = q5[4,...]

    ''' Compute aliases '''
    yM = 1.0 - yWv
    yWd = yWt - yWv
    yL = (yM - (yC + yWd))
    rho = self.rho_p(yWv, p)

    p, ptilde = self.p_ptilde(rho, yWv)
    vf_g = self.volfrac_gas(rho, yWv, p)
    # Compute partial of density w.r.t. water mass fraction (const p, T)
    drho_dyw = self.drho_dyw(rho, p)
    # Compute solubility based on pure silicate melt mass fraction
    yHat = self.yWd_target(yWv, yWt, yC, p)

    RHS_advection = -(u / dx) * self.__class__.diff_op1(x).dot(q5.T).T

    # Boundary SAT at left boundary (spatial index 0)
    tau = np.abs(u)[0]
    RHS_advection[...,0] += (tau / dx[0]) * (BC_fn(t)[2:,...] - q5[...,0])
    # Replace non-advected rho_prime terms
    RHS_advection[0,...] = drho_dyw * RHS_advection[1,...]

    # Compute solubility based on pure silicate melt mass fraction
    yHat = np.clip(self.solubility_k * p ** self.solubility_n * yL, 0, yWt)
    # Compute total steady state advection RHS
    ddt_yWv = (yWd - yHat) / (self.tau_d)
    ddt_yWt = np.zeros_like(p)
    ddt_yC  = np.zeros_like(p)
    ddt_yF  = (yM - yF) / (self.tau_f) * np.asarray(vf_g - self.vf_g_crit > 0.0).astype(float)
    # Compute source vector for reaction terms
    #   [rho_prime, yWv, yWt, yC, yF]
    RHS_source = np.stack([0 * drho_dyw * ddt_yWv,
                          ddt_yWv,
                          ddt_yWt,
                          ddt_yC,
                          ddt_yF,
                          ], axis=0)
    if is_debug_mode:
      return RHS_advection + RHS_source, RHS_advection, RHS_source
    else:
      return RHS_advection, RHS_source

  def full_solve(self, xlims, Nx=1001, p0=80e6, u0=0.7, yWt0=0.025,
                 yC0=0.4, yF0=0.0,
                 CFL=0.8, ivp_method="BDF", ivp_max_step=np.inf):
    # Construct grid
    x = np.linspace(xlims[0], xlims[-1], Nx)
    self.x = x
    # Upwinded dx, repeating first element
    dx = np.concatenate([[x[1] - x[0]], x[1:] - x[:-1]])
    self.dx = dx
    
    t = 0.0

    # Solve full steady state system
    soln = self.solve_full_steady_state(
      p0=p0,
      u0=u0,
      yWt0=yWt0,
      yC0=yC0,
      yF0=yF0,
      xlims=xlims)
    self._soln_steady = soln
    # Grid restriction for initial state
    q = soln.sol(x)
    q2 = q[0:2,...]
    q5 = q[2:,...]

    # Replace density with pressure
    #   TODO: clean up by using pressure directly in steady state
    q2[0:1,...] = self.p_ptilde(q[0:1,...], q[3:4,...])[0]

    # Add hooks for external access
    self.q2 = q2
    self.q5 = q5
    # Construct initial q2 interpolant by slicing full steady state
    q2_interp = lambda x: np.where(x <= soln.t.max(), soln.sol(x), soln.y[:,-1:])[0:2,...]
    # Construct initial q5 interpolant
    q5_interp = scipy.interpolate.PchipInterpolator(x,
                                                    q5,
                                                    extrapolate=True,
                                                    axis=1)
    # Set time-dependent BC
    BC_unsteady = lambda t: self.BC_gaussian(t,p0=p0, u0=u0,
                                             yWt0=yWt0, yC0=yC0, yF0=yF0)
    # BC_unsteady = lambda t: self.BC_steady(t,p0=p0, u0=u0,
    #                                          yWt0=yWt0, yC0=yC0, yF0=yF0)

    # Set debug caches
    # self.cacheadv = []
    # self.cachesrc = []
    # self.cache2 = []
    # self.cache3 = [q5.copy()]
    # self.cachep = [q2[0,...]]
    # self.cacheu = []
    self.q5_messages = []

    # Compute dt from desired CFL condition
    u = q2[1,...]
    dt = CFL * (dx / np.abs(u)).min()
    self.dt = dt

    # Set output arrays
    Nt = self.Nt
    self.out_q2 = np.zeros((Nt+1, 2, x.shape[0]))
    self.out_q2[0,...] = q2
    self.out_q5 = np.zeros((Nt+1, 5, x.shape[0]))
    self.out_q5[0,...] = q5
    self.out_t = dt * np.arange(0, Nt+1)

    # Unsteady solve
    for i in range(Nt):
      t = i * dt
      self.t = t

      self._pre_q5 = q5.copy()

      if self._use_grid_RHS_advance:
        # Injection BC
        q5[:,0] = BC_unsteady(t)[2:,...]
        # Compute RHS from unsteady part
        unsteady_RHS_adv, unsteady_RHS_source = self.unsteady_solve_RHS(
          BC_unsteady, x, t, q5, self.q2, is_debug_mode=False)
        dq = dt * (unsteady_RHS_adv + unsteady_RHS_source)
        # Forward Euler step for q5 except at injected boundary
        q5[:,1:] += dq[:,1:]
        # Disable source above choked flow
        # += dt * np.where(x <= x_max_choked, dq, 0.0) [:,1:]
      else:
        # Compute exact advection of interpolated q5 solution
        q5_advected = self.advect_step(
          BC_unsteady, x, t, self.dt, q5_interp, q2_interp, u0)
        self.q5_advected = q5_advected
        q5[:] = q5_advected
        # q5 = np.where(x <= x_max_choked, q5_advected, q5)
        # Update source by stepping
        # _, unsteady_RHS_source = self.unsteady_solve_RHS(
          # BC_unsteady, x, t, q5, self.q2, is_debug_mode=False)
        q5_reacted = self.react_step(x, t, self.dt, q5, self.q2)
        self.q5_reacted = q5_reacted
        q5[:] = q5_reacted
        # q5 += dt * np.where(x <= x_max_choked, unsteady_RHS_source, 0.0)
        
      # Clipping mass fractions for undersaturated magma due to splitting error
      # of advection and source
      clip_eps = 1e-5
      # Check for dastardly unphysicalness
      msgs = ""
      if np.any(q5[1,...] > 1 + clip_eps) or np.any(q5[1,...] < 0 - clip_eps):
        msgs += "yWv exceeds physical range; "
      if np.any(q5[2,...] > 1 + clip_eps) or np.any(q5[2,...] < 0 - clip_eps):
        msgs += "yWt exceeds physical range; "
      if np.any(q5[3,...] > 1 + clip_eps) or np.any(q5[3,...] < 0 - clip_eps):
        msgs += "yC exceeds physical range; "
      if np.any(q5[4,...] > 1 + clip_eps) or np.any(q5[4,...] < 0 - clip_eps):
        msgs += "yF exceeds physical range; "
      if len(msgs) > 0:
        # self._dump_unsteady_RHS_source = unsteady_RHS_source
        self._dump_q5 = q5.copy()
        self._dump_p = q2[0,...]
        self._dump_u = q2[1,...]
        raise ValueError(msgs)
      # Clip to [0,1]
      q5[1:5,...] = np.clip(q5[1:5,...], 0, 1)

      # Debug caching
      # self.cacheadv.append(unsteady_RHS_adv.copy())
      # self.cachesrc.append(unsteady_RHS_source.copy())
      # self.cache3.append(q5.copy())

      # Construct monotonic interpolants for indep system
      q5_interp_raw = scipy.interpolate.PchipInterpolator(x,
                                        q5,
                                        extrapolate=True,
                                        axis=1)
      # q5_interp = lambda xq: q5_interp_raw(xq)
      
      def q5_interp(xq):
        ''' Vectorized returns '''
        return np.where(xq <= x.max(),
                        q5_interp_raw(xq),
                        q5[:,-1:])

      def q5_interp_scalar(xq):
        ''' Scalar returns'''
        return np.where(xq <= x.max(),
                        q5_interp_raw(xq),
                        q5[:,-1])
      self.q5_interp = q5_interp

      # dyw_dt_interpolant = scipy.interpolate.PchipInterpolator(
      #   x, (unsteady_RHS_adv + unsteady_RHS_source)[1,...],
      #   extrapolate=True,
      #   axis=-1)

      # Solve time-independent system
      soln, scales = self.solve_time_indep_system(q5_interp_scalar, p0=p0, u0=u0,
                                          xlims=xlims,
                                          method=ivp_method,
                                          max_step=ivp_max_step)
      # TODO: prevent undefined behaviour when called with a scalar
      q2_interp = lambda x: np.where(x <= soln.t.max(),
                                     soln.sol(x),
                                     soln.y[:,-1:]) * scales[:,np.newaxis]
      self.q2_interp = q2_interp
      
      # Grid restriction for unsteady solver
      #   with extrapolation guard: applying restriction to locations above
      #   isothermal choking returns last value instead
      self.q2 = q2_interp(x)

      # Debug caching
      # self.soln_time_indep = soln
      # if len(self.cacheu) == 0:
        # self.cacheu.append(self.q2[1,...].copy())
      # self.cache2.append(q2.copy())
      # self._unsteady_RHS_adv = unsteady_RHS_adv.copy()
      # self._unsteady_RHS_source = unsteady_RHS_source.copy()
      # self._dyw_dt_interpolant = dyw_dt_interpolant
      # self._q5_interpolants = q5_interpolants
      # self._soln_t_indep = soln
      # self.cachep.append(q2[0,...])

      # Save outputs
      self.out_q2[i+1,...] = self.q2
      self.out_q5[i+1,...] = q5
      self.q5_messages.append(soln.message)


    return q2, q5 
  
  def full_solve_choked(self, xlims, Nx=1001, p0=80e6, u0=None, yWt0=0.025,
                 yC0=0.4, yF0=0.0,
                 CFL=0.8, ivp_method="BDF", ivp_max_step=np.inf):
    ''' Solve the full QSS system with choked outflow. '''
    clock_prep = perf_counter()
    # Construct grid
    x = np.linspace(xlims[0], xlims[-1], Nx)
    self.x = x
    # Set integration domain to end specifically at choking
    xlims_unlimited = (0, 8000)
    if xlims_unlimited[1] < xlims[1]:
      self.logger.warning(f"numerical xlim modified: the physical domain is longer than the choke-limited domain")
      xlims_unlimited = 2 * xlims[1]

    # Upwinded vector of dx, repeating first element
    dx = np.concatenate([[x[1] - x[0]], x[1:] - x[:-1]])
    self.dx = dx

    # Solve full steady state system
    def solve_full_ivp_L(u0_iterate):
      ''' Full steady state problem, returning soln bunch and conduit length'''
      soln = self.solve_full_steady_state(
        p0=p0,
        u0=u0_iterate,
        yWt0=yWt0,
        yC0=yC0,
        yF0=yF0,
        xlims=xlims_unlimited)
      return soln, soln.t.max()
    # Shooting method for matching conduit length
    self.logger.debug(f"Steady state brentq called.")
    u0_solved, self._brentq_results = scipy.optimize.brentq(
      lambda u0: solve_full_ivp_L(u0)[1] - xlims[-1],
      0.1, 5.5, full_output=True)
    logging.debug(f"Steady state brentq call complete with u0 = {u0_solved}")
    self.logger.debug(f"Steady state brentq call results: {self._brentq_results}")
    # Solve using u0 from shooting method
    soln = self.solve_full_steady_state(
      p0=p0,
      u0=u0_solved,
      yWt0=yWt0,
      yC0=yC0,
      yF0=yF0,
      xlims=xlims_unlimited)
    # Shift solution so that choked flow is exactly at conduit_length
    #   This strategy extrapolates on the bottom instead of the top
    self.logger.debug(f"Steady state solve called with u0 = {u0_solved}")
    self.soln_steady_solved = soln
    self.u0_solved_steady = u0_solved
    self.u0 = u0_solved
    self.steady_interp = lambda x: soln.sol(x + (soln.t.max() - xlims[-1]))

    # Grid restriction for initial state
    q = self.steady_interp(x)
    q2 = q[0:2,...]
    q5 = q[2:,...]
    # Replace density with pressure
    #   TODO: clean up by using pressure directly in steady state
    q2[0:1,...] = self.p_ptilde(q[0:1,...], q[3:4,...])[0]
    # Cache pressure for reverse-IVP solve strategy
    self.p_vent = q2[0,-1]
    # Add hooks for external access
    self.q2 = q2
    self.q5 = q5
    # Construct initial q2 interpolant by slicing full steady state
    q2_interp = lambda x: self.steady_interp(x)[0:2,...]
    # TODO: replacing first q2_interp with pressure
    q2_interp = scipy.interpolate.PchipInterpolator(x,
                                        q2,
                                        extrapolate=True,
                                        axis=1)

    # Construct initial q5 interpolant
    q5_interp = scipy.interpolate.PchipInterpolator(x,
                                                    q5,
                                                    extrapolate=True,
                                                    axis=1)
    # Set time-dependent BC
    # BC_unsteady = lambda t: self.BC_gaussian(t,p0=p0, u0=u0,
    #                                          yWt0=yWt0, yC0=yC0, yF0=yF0)
    BC_unsteady = lambda t: self.BC_gaussian_crossverif(t,p0=p0, u0=u0,
                                             yWt0=yWt0, yC0=yC0, yF0=yF0)
    # BC_unsteady = lambda t: self.BC_steady(t,p0=p0, u0=u0,
    #                                          yWt0=yWt0, yC0=yC0, yF0=yF0)

    # Compute dt from desired CFL condition
    u = q2[1,...]
    dt = CFL * (dx / np.abs(u)).min()
    self.dt = dt
    t = 0.0

    # Set output arrays
    Nt = self.Nt
    self.out_q2 = np.zeros((Nt+1, 2, x.shape[0]))
    self.out_q2[0,...] = q2
    self.out_q5 = np.zeros((Nt+1, 5, x.shape[0]))
    self.out_q5[0,...] = q5
    self.out_t = dt * np.arange(0, Nt+1)
    self.q5_messages = []

    self._use_reverse_IVP = False
    self.logger.debug(f"Timestep loop entered with with dt = {dt}.")
    self.logger.debug(f"Setup time was {perf_counter() - clock_prep}.")

    def clip_q5(solver:object, q5, q2, clip_eps=1e-5):
      ''' Clip mass fractions for undersaturated magma due to splitting error
      of advection and source. Attaches debug state to arg:solver '''
      # Check for dastardly unphysicalness
      msgs = ""
      if np.any(q5[1,...] > 1 + clip_eps) or np.any(q5[1,...] < 0 - clip_eps):
        msgs += "yWv exceeds physical range; "
      if np.any(q5[2,...] > 1 + clip_eps) or np.any(q5[2,...] < 0 - clip_eps):
        msgs += "yWt exceeds physical range; "
      if np.any(q5[3,...] > 1 + clip_eps) or np.any(q5[3,...] < 0 - clip_eps):
        msgs += "yC exceeds physical range; "
      if np.any(q5[4,...] > 1 + clip_eps) or np.any(q5[4,...] < 0 - clip_eps):
        msgs += "yF exceeds physical range; "
      if len(msgs) > 0:
        solver._dump_q5 = q5.copy()
        solver._dump_p = q2[0,...]
        solver._dump_u = q2[1,...]
        raise ValueError(msgs)
      q5_clipped = np.clip(q5[1:5,...], 0, 1)
      solver.logger.debug(f"Clip stage: is q5 == np.clip: {np.all(q5[1:5,...] == q5_clipped)}")
      # Clip to [0,1]
      q5[1:5,...] = q5_clipped

    def get_q2_interp_and_u(solver, q5_interp_scalar, p0, xlims_unlimited, ivp_method,
                      ivp_max_step) -> tuple:
      ''' Compute and return q2 interp. arg:solver is for storing debug state only,
       and reading cached u0 value. '''
      # Solve time-independent system with shooting method
      def solve_time_indep_ivp_L(u0_iterate):
        ''' Full steady state problem, returning soln bunch and conduit length.
        Solution is only correct when `scales` is applied.'''
        soln, scales = solver.solve_time_indep_system(
          q5_interp_scalar,
          p0=p0,
          u0=u0_iterate,
          xlims=xlims_unlimited,
          method=ivp_method,
          max_step=ivp_max_step)
        return soln, soln.t.max()
      try:
        max_newton_iter = 7
        newton_residual_atol = 1e-4
        # Take p_vent as initial guess
        p_iterate = solver.p_vent
        # Debug history
        _p_iterate_hist = []
        _p_residual_hist = []
        _u_in_hist = []
        for i in range(max_newton_iter):
          # Choose local ode rtol
          # if i == 0:
          #   rtol = 1e-5
          # else:
          #   # Estimate local rtol requirement from residual of p_inlet
          #   rtol = max(min(1e-5 * np.abs(residual) / p0, 1e-5), 1e-9)
          rtol = 1e-6
          # Newton iteration
          (p_iterate, residual, u_in, slope,
           u_in_sensitivity_pout,
           uin_change_estimate,
           d_dpout_integralp,
           soln) = split_solver.newton_map_pout(
             p0, p_iterate, xlims, q5_interp_scalar,
             ode_rtol=rtol, M_eps=1e-2)
          # print(p_iterate, u_in, slope, residual/p0, rtol)
          # if (residual/p0) * (residual/p0) < newton_residual_tol * newton_residual_tol:
          #   solver.logger.debug(f"Newton tolerance reached at Newton step {i}")
          #   break

          _p_iterate_hist.append(p_iterate)
          _p_residual_hist.append(residual)
          _u_in_hist.append(u_in)

          if np.abs(uin_change_estimate) < newton_residual_atol:
            solver.logger.debug(f"Inlet velocity atol {newton_residual_atol:.2e} reached in Newton step {i}")
            break
          if i >= 1 and residual * residual > _p_residual_hist[-2] * _p_residual_hist[-2]:
            solver.logger.debug(f"Residual increased at Newton step {i}")
            break
        else:
          raise ValueError(f"Max number of newton iterations reached ({max_newton_iter}).")
        # Cache new iterate for next call
        solver.p_vent = p_iterate
        # Cache inlet velocity for alternative strategy
        solver.u0 = u_in

        finally_use_forward_system = False
        if finally_use_forward_system:
          # Solve system using solved u0 TODO: replace wiht p, M2 system with adapted output
          self.logger.debug(f"Solving forward ODE with (p,u) system with solved u_in ({u_in}).")
          soln, scales = solver.solve_time_indep_system(q5_interp_scalar, p0=p0,
                                              u0=u_in,
                                              xlims=xlims_unlimited,
                                              method=ivp_method,
                                              max_step=ivp_max_step)
          return (lambda x: np.einsum("i..., i -> i...",
                                  soln.sol(x + (soln.t.max() - xlims[-1])),
                                  scales),
                  u_in,)
        else:
          return (lambda x: soln.sol(x)[0:2,...], u_in,)
      except FileNotFoundError as e:# Exception as e: 
        solver.logger.warning(f"Error encounted in Newton-reverse-IVP. Message follows: ")
        solver.logger.warning(f"{e}")
        solver.logger.debug(f"Proceeding with bisection strategy. ")
      
      # Try neighbourhood search of u0
      try:
        # Bracket expansion factor (whatever implementation)
        exp_factor = 1.1
        bracket_L = np.array([solver.u0, solver.u0])
        bracket_mult = np.array([1/exp_factor, exp_factor])
        root_find_fn = lambda u0: solve_time_indep_ivp_L(u0)[1] - xlims[-1]
        root_table = np.zeros((5,2))
        bracket_table = np.zeros((5,2))

        for idx_expansion in range(5):
          bracket_L *= bracket_mult
          bracket_table[idx_expansion, :] = bracket_L
          # Compute function value at brackets
          root_table[idx_expansion, 0] = root_find_fn(bracket_L[0])
          root_table[idx_expansion, 1] = root_find_fn(bracket_L[1])
          if root_table[idx_expansion, 0] * root_table[idx_expansion, 1] < 0:
            # Identify best bracket from least positive and least negative evals
            bracket_L = np.array([
              bracket_table.ravel()[np.argmin(np.where(root_table > 0, root_table, np.inf).ravel())],
              bracket_table.ravel()[np.argmax(np.where(root_table < 0, root_table, -np.inf).ravel())]
            ])
            vals_L = np.array([
              root_table.ravel()[np.argmin(np.where(root_table > 0, root_table, np.inf).ravel())],
              root_table.ravel()[np.argmax(np.where(root_table < 0, root_table, -np.inf).ravel())]
            ])
            break
        else:
          raise ValueError("Unable to find an expansion bracket; using manual lims")
        # Using best bounds, secant method and terminate when bracket is on same side (due to noise)
        _x_secant_hist = [*bracket_L]
        _y_secant_hist = [*vals_L]
        atol = 1e-1
        for i in range(60):
          new_x = 0.5 * (bracket_L[0] + bracket_L[1])
          new_y = root_find_fn(new_x)
          # Use midpoint as new bracket
          if new_y * vals_L[0] > 0:
            bracket_L[0], vals_L[0] = new_x, new_y
          else:
            bracket_L[1], vals_L[1] = new_x, new_y
          # Debug value tracking
          _x_secant_hist.append(new_x)
          _y_secant_hist.append(new_y)
          # End with lost bracket
          if vals_L[0] * vals_L[1] > 0:
            break
          # Absolute tolerance condition:
          if new_y*new_y < atol*atol:
            break
        # Load new u0 value
        u0_solved = new_x
      except Exception as e:
        solver.logger.warning(f"Error encounted in expansion bracket. Message follows: ")
        solver.logger.warning(f"{e}")
        bracket_L = (solve_time_indep_ivp_L(0.1)[1],
                      solve_time_indep_ivp_L(5.500)[1])
        solver.logger.debug(f"Bracketing L values for 0.1, 5.5: {bracket_L}")        
        # Shooting method for matching conduit length 
        try:
          u0_solved, solver._brentq_results_time_indep = scipy.optimize.brentq(
            lambda u0: solve_time_indep_ivp_L(u0)[1] - xlims[-1],
            0.1, 5.5, full_output=True)
        except ValueError as e:
          raise ValueError(f"Input pressure {p0} may be insufficient to "
                            +f"produce choked flow. Implement BVP using top/bottom pressure?"
                            +f"Check also extrapolation of q5 in q2 solve.") from e
        solver.logger.debug(f"Bracketing time-independent solver (brentq) finished with u0 = {float(u0_solved)}")
      # Solve system using solved u0
      self.logger.debug(f"Steady state bisection impl. complete with u0 = {u0_solved}")
      soln, scales = solver.solve_time_indep_system(q5_interp_scalar, p0=p0,
                                          u0=u0_solved,
                                          xlims=xlims_unlimited,
                                          method=ivp_method,
                                          max_step=ivp_max_step)
      # Cache u0 for next iteration
      solver.u0 = u0_solved
      solver.p_vent = soln.y[0,-1] * scales[0]
      # Shift solution so that choked flow is exactly at conduit_length
      #   This strategy extrapolates on the bottom instead of the top
      return (lambda x: np.einsum("i..., i -> i...",
                                 soln.sol(x + (soln.t.max() - xlims[-1])),
                                 scales),
              u0_solved,)

    # Unsteady solve
    for i in range(Nt):
      t = i * dt
      self.t = t
      clock_step = perf_counter()

      # Measure and log L2 norm of solution
      q5_L2 = np.sqrt(scipy.integrate.trapz(q5*q5, x=x, axis=1))
      self.logger.debug(f"q5 L2-norm:                    {q5_L2}")
      q5[:] = self.advect_react_step(BC_unsteady, x, t, self.dt,
                                     q5_interp,
                                     q2_interp,
                                     p0,
                                     self.u0)
      # Measure and log L2 norm of solution
      q5_L2 = np.sqrt(scipy.integrate.trapz(q5*q5, x=x, axis=1))
      self.logger.debug(f"q5 L2-norm updated           : {q5_L2}")
      # Clip mass fractions
      clip_q5(self, q5, q2, clip_eps=1e-5)
      

      '''# Compute exact advection of interpolated q5 solution
      q5[:] = self.advect_step(
        BC_unsteady, x, t, self.dt, q5_interp, q2_interp, u0_solved)
      # Measure and log L2 norm of solution
      q5_L2 = np.sqrt(scipy.integrate.trapz(q5*q5, x=x, axis=1))
      self.logger.debug(f"q5 L2-norm advected:           {q5_L2}")

      # Update source by substep reaction (1/2)
      q5[:] = self.react_step(x, t, self.dt, q5, self.q2)
      # Measure and log L2 norm of solution
      q5_L2 = np.sqrt(scipy.integrate.trapz(q5*q5, x=x, axis=1))
      self.logger.debug(f"q5 L2-norm reacted      dt1/2: {q5_L2}")
      # Clip mass fractions
      clip_q5(self, q5, q2, clip_eps=1e-5)

      # Update source by analytic substep reaction (2/2)
      q5[:] = self.react_step(x, t, 0.5 * self.dt, q5, self.q2)
      # Measure and log L2 norm of solution
      q5_L2 = np.sqrt(scipy.integrate.trapz(q5*q5, x=x, axis=1))
      self.logger.debug(f"q5 L2-norm reacted      dt2/2: {q5_L2}")
      # Clip mass fractions
      clip_q5(self, q5, q2, clip_eps=1e-5)'''

      # Construct monotonic q5 interpolant
      q5_interp_raw = scipy.interpolate.PchipInterpolator(x, q5,
        extrapolate=True, axis=1)
      # TODO: prevent undefined behaviour when called with a scalar, merge following two funcs
      def q5_interp(xq, nu=0):
        ''' Vectorized returns '''
        return np.where(xq <= x.max(), q5_interp_raw(xq, nu=nu), q5[:,-1:])
      def q5_interp_scalar(xq, nu=0):
        ''' Scalar returns'''
        return np.where(xq <= x.max(), q5_interp_raw(xq, nu=nu), q5[:,-1])
      self.q5_interp = q5_interp

      # q2 solve and obtain interpolator
      q2_interp, u0_solved = get_q2_interp_and_u(self, q5_interp_scalar, p0, xlims_unlimited,
                                ivp_method, ivp_max_step)
      self.u0 = u0_solved
      self.q2_interp = q2_interp
      # Grid restriction for unsteady solver
      self.q2 = q2_interp(x)

      # Save outputs
      self.logger.debug(f"Solved step {i} (t = {t:.5f} s) in {perf_counter() - clock_step} s.")

      self.out_q2[i+1,...] = self.q2
      self.out_q5[i+1,...] = q5
      self.q5_messages.append(soln.message)

    self.logger.debug(f"Solve complete.")
    return q2, q5

  def full_solve_equilibrum(self, xlims, Nx=2001, p0=80e6, yWt0=0.025,
                yC0=0.4, CFL=0.8, ivp_method="BDF", ivp_max_step=np.inf) -> tuple:
    ''' Solve the QSS system with choked outflow and eq. exsoln, frag. '''
    clock_prep = perf_counter()
    # Construct grid
    x = np.linspace(xlims[0], xlims[-1], Nx)
    self.x = x
    # Set integration domain to end specifically at choking
    xlims_unlimited = (0, 8000)
    if xlims_unlimited[1] < xlims[1]:
      self.logger.warning(f"numerical xlim modified: the physical domain is longer than the choke-limited domain")
      xlims_unlimited = 2 * xlims[1]

    # Upwinded vector of dx, repeating first element
    dx = np.concatenate([[x[1] - x[0]], x[1:] - x[:-1]])
    self.dx = dx

    u_bracket = (0.1, 12.5)

    # Solve full steady state system
    def solve_full_ivp_L(u0_iterate):
      ''' Full steady state problem, returning soln bunch and conduit length'''
      soln = self.solve_full_steady_state_eq(
        p0=p0,
        u0=u0_iterate,
        yWt0=yWt0,
        yC0=yC0,
        xlims=xlims_unlimited)
      return soln, soln.t.max()
    # Shooting method for matching conduit length
    self.logger.debug(f"Steady state brentq called.")
    u0_solved, self._brentq_results = scipy.optimize.brentq(
      lambda u0: solve_full_ivp_L(u0)[1] - xlims[-1],
      *u_bracket, full_output=True)
    logging.debug(f"Steady state brentq call complete with u0 = {u0_solved}")
    self.logger.debug(f"Steady state brentq call results: {self._brentq_results}")
    # Solve using u0 from shooting method
    soln = self.solve_full_steady_state_eq(
      p0=p0,
      u0=u0_solved,
      yWt0=yWt0,
      yC0=yC0,
      xlims=xlims_unlimited)
    # Shift solution so that choked flow is exactly at conduit_length
    #   This strategy extrapolates on the bottom instead of the top
    self.logger.debug(f"Steady state solve called with u0 = {u0_solved}")
    self.soln_steady_solved = soln
    self.u0_solved_steady = u0_solved
    self.u0 = u0_solved
    self.steady_interp = lambda x: soln.sol(x + (soln.t.max() - xlims[-1]))

    # Grid restriction for initial state
    q = self.steady_interp(x)
    q_pu = q[0:2,...] # p, u
    q_t = q[2:,...]   # yWt, yC
    # Add hooks for external access
    self.q_pu = q_pu
    self.q_t = q_t
    # Construct initial q2 interpolant by slicing full steady state
    q_pu_interp = lambda x: self.steady_interp(x)[0:2,...]
    # Construct initial q5 interpolant
    q_t_interp = scipy.interpolate.PchipInterpolator(x,
                                                    self.q_t,
                                                    extrapolate=True,
                                                    axis=1)
    # Set time-dependent BC
    # BC_unsteady = lambda t, u0: self.BC_sine_box_eq(t, p0=p0, u0=u0,
                                            # yWt0=yWt0, yC0=yC0)
    BC_unsteady = lambda t, u0: self.BC_gaussian_eq_crossverif(t, p0=p0, u0=u0,
                                            yWt0=yWt0, yC0=yC0)
    # BC_unsteady = lambda t: self.BC_steady_eq(t,p0=p0, u0=u0,
    #                                          yWt0=yWt0, yC0=yC0)

    # Compute dt from desired CFL condition
    u = q_t[1,...]
    dt = CFL * (dx / np.abs(u)).min()
    self.dt = dt
    t = 0.0

    # Set output arrays
    Nt = self.Nt
    self.out_q_pu = np.zeros((Nt+1, 2, x.shape[0]))
    self.out_q_pu[0,...] = q_pu
    self.out_q_t = np.zeros((Nt+1, 2, x.shape[0]))
    self.out_q_t[0,...] = q_t
    self.out_t = dt * np.arange(0, Nt+1)
    self.q_t_messages = []

    self._use_reverse_IVP = False
    self.logger.debug(f"Timestep loop entered with with dt = {dt}.")
    self.logger.debug(f"Setup time was {perf_counter() - clock_prep}.")

    def clip_q_t(solver:object, q_t, q_pu, clip_eps=1e-5):
      ''' Clip mass fractions for undersaturated magma due to splitting error
      of advection and source. Attaches debug state to arg:solver '''
      # Check for dastardly unphysicalness
      msgs = ""
      if np.any(q_t[0,...] > 1 + clip_eps) or np.any(q_t[0,...] < 0 - clip_eps):
        msgs += "yWt exceeds physical range; "
      if np.any(q_t[1,...] > 1 + clip_eps) or np.any(q_t[1,...] < 0 - clip_eps):
        msgs += "yC exceeds physical range; "
      if len(msgs) > 0:
        solver._dump_q_t = q_t.copy()
        solver._dump_p = q_pu[0,...]
        solver._dump_u = q_pu[1,...]
        raise ValueError(msgs)
      q_t_clipped = np.clip(q_t, 0, 1)
      solver.logger.debug(f"Clip stage: is q_t == np.clip: {np.all(q_t == q_t_clipped)}")
      # Clip to [0,1]
      q_t[...] = q_t_clipped

    def get_q_pu_interp_and_u(solver, q_t_interp_scalar, p0, xlims_unlimited, ivp_method,
                      ivp_max_step) -> tuple:
      ''' Compute and return q2 interp. arg:solver is for storing debug state only.
        Returns explicitly solved u0. '''
      # Solve time-independent system with shooting method
      def solve_time_indep_ivp_L(u0_iterate):
        ''' Full steady state problem, returning soln bunch and conduit length.
        Solution is only correct when `scales` is applied.'''
        soln, scales = solver.solve_time_indep_system_eq(
          q_t_interp_scalar,
          p0=p0,
          u0=u0_iterate,
          xlims=xlims_unlimited,
          method=ivp_method,
          max_step=ivp_max_step)
        return soln, soln.t.max() 
            
      bracket_L = (solve_time_indep_ivp_L(u_bracket[0])[1],
                    solve_time_indep_ivp_L(u_bracket[1])[1])
      solver.logger.debug(f"Bracketing L values for {u_bracket}: {bracket_L}")
      # Shooting method for matching conduit length
      try:
        u0_solved, solver._brentq_results_time_indep = scipy.optimize.brentq(
          lambda u0: solve_time_indep_ivp_L(u0)[1] - xlims[-1],
          *u_bracket, full_output=True, rtol=1e-3)
      except ValueError as e:
        raise ValueError(f"Input pressure {p0} may be insufficient to "
                          +f"produce choked flow. Implement BVP using top/bottom pressure?"
                          +f"Check also extrapolation of q_t in q_pu solve.") from e
      solver.logger.debug(f"Bracketing time-independent solver (brentq) finished with u0 = {float(u0_solved)}")
      # Solve system using solved u0
      soln, scales = solver.solve_time_indep_system_eq(q_t_interp_scalar, p0=p0,
                                          u0=u0_solved,
                                          xlims=xlims_unlimited,
                                          method=ivp_method,
                                          max_step=ivp_max_step)
      # Cache u0 for next iteration
      solver.u0 = u0_solved
      # Shift solution so that choked flow is exactly at conduit_length
      #   This strategy extrapolates on the bottom instead of the top
      return (lambda x: np.einsum("i..., i -> i...",
                                soln.sol(x + (soln.t.max() - xlims[-1])),
                                scales),
              u0_solved,)

    # Unsteady solve
    for i in range(Nt):
      t = i * dt
      self.t = t
      clock_step = perf_counter()

      # Measure and log L2 norm of solution
      q_t_L2 = np.sqrt(scipy.integrate.trapz(q_t*q_t, x=x, axis=1))
      self.logger.debug(f"q_t L2-norm:                   {q_t_L2}")

      # Compute exact advection of interpolated q5 solution
      q_t[:] = self.advect_step_eq(
        lambda t: BC_unsteady(t, u0_solved), x, t, self.dt, q_t_interp, q_pu_interp, u0_solved)
      # Measure and log L2 norm of solution
      q_t_L2 = np.sqrt(scipy.integrate.trapz(q_t*q_t, x=x, axis=1))
      self.logger.debug(f"q_t L2-norm advected:          {q_t_L2}")

      # Construct monotonic q5 interpolant
      q_t_interp_raw = scipy.interpolate.PchipInterpolator(x, q_t,
        extrapolate=True, axis=1)      
      # TODO: prevent undefined behaviour when called with a scalar, merge following two funcs
      def q_t_interp(xq):
        ''' Vectorized returns '''
        return np.where(xq <= x.max(), q_t_interp_raw(xq), q_t[:,-1:])
      def q_t_interp_scalar(xq):
        ''' Scalar returns'''
        return np.where(xq <= x.max(), q_t_interp_raw(xq), q_t[:,-1])
      self.q_t_interp = q_t_interp

      # q2 solve and obtain interpolator
      q_pu_interp, u_solved = get_q_pu_interp_and_u(self, q_t_interp_scalar,
                                                    p0, xlims_unlimited,
                                                    ivp_method, ivp_max_step)
      self.q_pu_interp = q_pu_interp
      # Grid restriction for unsteady solver
      self.q_pu = q_pu_interp(x)

      # Save outputs
      self.logger.debug(f"Solved step {i} (t = {t:.5f} s) in {perf_counter() - clock_step} s.")

      self.out_q_pu[i+1,...] = self.q_pu
      self.out_q_t[i+1,...] = q_t
      self.q_t_messages.append(soln.message)

    self.logger.debug(f"Solve complete.")
    return q_pu, q_t

def solve_full_ivp_L(u0, rtol=1e-5, atol=1e-6):
  ''' Compute full steady state IVP output domain length. '''
  soln = split_solver.solve_full_steady_state(p0=100e6,
                                              u0=u0,
                                              yWt0=0.025,
                                              yC0=0.4,
                                              yF0=0.0,
                                              xlims=(0, 10000),
                                              M_eps=1e-2,
                                              method="BDF",
                                              rtol=rtol, atol=atol)
  return soln, soln.t.max()

if __name__ == "__main__":

  from scipy.special import erf

  # Set up arbitrary SteadyState object for borrowing viscosity model
  # PC locale:
  locale = r"C:\Users\Fredric\Documents\Volcano\quail_volcano\src\compressible_conduit_steady"
  import sys
  sys.path.append(locale)
  import steady_state

  x_km_range = np.linspace(0, 15000, 100)
  f = steady_state.SteadyState(x_km_range, 1e5, 100e6, input_type="p",
                              override_properties={
                                "tau_f": 1e0, # 1e-2,
                                "tau_d": 1e0, # 1e-2,
                                #  "yWt": 0.05,
                                "yC": 0.4,
                                "rho0_magma": 2.6e3,
                              }, skip_rootfinding=True)
  
  

  def loc_F_fric_viscosity_model(T, y, yF, yWt=0.03, yC=0.4):
    ''' Calculates the viscosity as a function of dissolved
    water and crystal content (assumes crystal phase is incompressible)/
    Does not take into account fragmented vs. not fragmented (avoiding
    double-dipping the effect of fragmentation).
    '''
    # Calculate pure melt viscosity (Hess & Dingwell 1996)
    yWd = yWt - y
    yM = 1.0 - y
    yL = yM - (yWd + yC)
    mfWd = yWd / yL # mass concentration of dissolved water
    mfWd = np.where(mfWd <= 0.0, 1e-8, mfWd)
    log_mfWd = np.log(mfWd*100)
    log10_vis = -3.545 + 0.833 * log_mfWd
    log10_vis += (9601 - 2368 * log_mfWd) / (T - 195.7 - 32.25 * log_mfWd)
    # Prevent overflowing float
    log10_vis = np.where(log10_vis > 300, 300, log10_vis)
    meltVisc = 10**log10_vis
    # Calculate relative viscosity due to crystals (Costa 2005).
    alpha = 0.999916
    phi_cr = 0.673
    gamma = 3.98937
    delta = 16.9386
    B = 2.5
    # Compute volume fraction of crystal at equal phasic densities
    # Using crystal volume per (melt + crystal + dissolved water) volume
    phi_ratio = np.clip((yC / yM) / phi_cr, 0.0, None)
    erf_term = erf(
      np.sqrt(np.pi) / (2 * alpha) * phi_ratio * (1 + phi_ratio**gamma))
    crysVisc = (1 + phi_ratio**delta) * ((1 - alpha * erf_term)**(-B * phi_cr))
    
    viscosity = meltVisc * crysVisc
    return viscosity
  
  # Cross-verification case
  params = {
    "T0": 1050,
    "conduit_radius": 50,
    "tau_d": 10.0, # CHECK, and Euler stability
    "tau_f": 1.0, # TODO: add resonant case for tau_f == tau_d
    "vf_g_crit": 0.75,
    "solubility_n": 0.5,
    "solubility_k": 5e-06,
    "K": 10e9,
    "rho_m0": 2.6e3,
    "p_m0": 36e6,
    "R_wv": f.mixture.waterEx.R,
    "F_fric_viscosity_model": loc_F_fric_viscosity_model,
  }
  # 'R_wv': 461.3762486126526, 'c_p': 2288.0
  # 'c_v_magma': 3000.0,

  #Base inlet state:
  # yC: 0.4
  yWt0 = 0.03 * (1.0 - 0.4) / (1.0 + 0.03)

  # Last:
  # split_solver = SplitSolver(params, Nt=2000)
  # q_pu, q_t = split_solver.full_solve_choked((0,1000), Nx=2001, p0=40e6, yWt0=yWt0,
  #                 yC0=0.4, CFL=200, ivp_method="BDF", ivp_max_step=np.inf)
  split_solver = SplitSolver(params, Nt=2000)
  q_pu, q_t = split_solver.full_solve_choked((0,1000), Nx=2001, p0=40e6, yWt0=yWt0,
                  yC0=0.4, CFL=200, ivp_method="RK45", ivp_max_step=np.inf)
  print(f"Final t: {split_solver.t} s")
  np.save("crossverif_tauf_run15_q2", split_solver.out_q2)
  np.save("crossverif_tauf_run15_q5", split_solver.out_q5)
  np.save("crossverif_tauf_run15_t", split_solver.out_t)
  np.save("crossverif_tauf_run15_x", split_solver.x)

  # 13: first run using Newton
  # 14: Newton with analytic Jacobian, RK45, rtol=1e-6 (< 1 s per step)

  # split_solver = SplitSolver(params, Nt=200)
  # q_pu, q_t = split_solver.full_solve_choked((0,1000), Nx=1001, p0=40e6, yWt0=yWt0,
  #                 yC0=0.4, CFL=0.25*1000, ivp_method="BDF", ivp_max_step=np.inf)
  # print(f"Final t: {split_solver.t} s")
  # np.save("crossverif_tauf_run3_q2", split_solver.out_q2)
  # np.save("crossverif_tauf_run3_q5", split_solver.out_q5)
  # np.save("crossverif_tauf_run3_t", split_solver.out_t)
  # np.save("crossverif_tauf_run3_x", split_solver.x)

  # Eq case
  # split_solver = SplitSolver(params, Nt=400)
  # q_pu, q_t = split_solver.full_solve_equilibrum((0,1000), Nx=1001, p0=40e6, yWt0=yWt0,
  #                 yC0=0.4, CFL=0.25, ivp_method="BDF", ivp_max_step=np.inf)
  # print(f"Final t: {split_solver.t} s")
  # np.save("crossverif_tauf_1_q_pu", split_solver.out_q_pu)
  # np.save("crossverif_tauf_1_q_t", split_solver.out_q_t)
  # np.save("crossverif_tauf_1_t", split_solver.out_t)
  # np.save("crossverif_tauf_1_x", split_solver.x)

  if False:
    params = {
      "T0": 950+273.15,
      "conduit_radius": 10,
      "tau_d": 5.0, # CHECK, and Euler stability
      # "tau_f": .25, # TODO: add resonant case for tau_f == tau_d
      "tau_f": 2.5, # TODO: add resonant case for tau_f == tau_d
      "vf_g_crit": 0.8,
      "solubility_n": f.solubility_n,
      "solubility_k": f.solubility_k,
      "K": 10e9,
      "rho_m0": 2.7e3,
      "p_m0": 5e6,
      "R_wv": f.mixture.waterEx.R,
      "F_fric_viscosity_model": loc_F_fric_viscosity_model,
    }

    # split_solver = SplitSolver(params, Nt=600)
    # Strang-split solve with choked boundary, non-instantaneous reaction
    # q2, q5 = split_solver.full_solve_choked((0,2100), Nx=1001, p0=80e6, u0=0.7, yWt0=0.025,
    #                 yC0=0.4, yF0=0.0, CFL=100)
    # print(f"Final t: {split_solver.t} s")
    # np.save("out_q2_strang_2_5-2p5", split_solver.out_q2)
    # np.save("out_q5_strang_2_5-2p5", split_solver.out_q5)
    # np.save("out_t_strang_2_5-2p5", split_solver.out_t)
    # np.save("out_x_strang_2_5-2p5", split_solver.x)

    # Equilibrium chemistry model with choked boundary
    split_solver = SplitSolver(params, Nt=400)
    q_pu, q_t = split_solver.full_solve_equilibrum((0,2100), Nx=1001, p0=80e6, yWt0=0.025,
                    yC0=0.4, CFL=0.25, ivp_method="BDF", ivp_max_step=np.inf)
    print(f"Final t: {split_solver.t} s")
    np.save("placeholder_out_q_pu_sinebox2", split_solver.out_q_pu)
    np.save("placeholder_out_q_t_sinebox2", split_solver.out_q_t)
    np.save("placeholder_out_t_sinebox2", split_solver.out_t)
    np.save("placeholder_out_x_sinebox2", split_solver.x)

    # sinebox1: 1x refinement (100 @ 1 CFL)
    # sinebox2: 4x refinement (400 @ 0.25 CFL)
    # sinebox3: 8x refinement (800 @ 0.125 CFL)

    split_solver = SplitSolver(params, Nt=800)
    q_pu, q_t = split_solver.full_solve_equilibrum((0,2100), Nx=1001, p0=80e6, yWt0=0.025,
                    yC0=0.4, CFL=0.125, ivp_method="BDF", ivp_max_step=np.inf)
    print(f"Final t: {split_solver.t} s")
    np.save("placeholder_out_q_pu_sinebox3", split_solver.out_q_pu)
    np.save("placeholder_out_q_t_sinebox3", split_solver.out_q_t)
    np.save("placeholder_out_t_sinebox3", split_solver.out_t)
    np.save("placeholder_out_x_sinebox3", split_solver.x)