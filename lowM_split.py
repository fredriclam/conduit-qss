import datetime
import logging
import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.sparse
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
  
  def isothermal_mach_squared_simple(self, u, yWv, p):
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
    A[0,:] = [u, rho, u, -u* drho_dyw, 0, 0, 0]
    A[1,:] = [cT2/rho, u, cT2/rho, dp_dyw/rho , 0, 0, 0]
    A[2,2] = 1.0
    A[[3,4,5,6],[3,4,5,6]] = u
    # Compute source vector
    b = np.stack([-drho_dyw * source_yWv,
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

  def time_indep_RHS(self, x, q2:np.array, r_interpolants:callable,
                     is_debug_mode=False):
    ''' RHS for time-independent portion of the system.
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
      return Ainv @ b
    return Ainv @ b

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
      rtol=1e-6,
      atol=1e-8,
      events=[event_isothermal_choked,])
    return soln, scales

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
      u0 * np.ones_like(t),
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

  def advect_step(self, BC_fn, x, t, dt, q5_interp, q2_interp, u0):
    ''' Semi-Lagrangian advection. '''
    # Compute advection foot point (x - ut) using RK2
    x1 = x - dt * q2_interp(x)[1,...]
    x2 = x - dt * q2_interp(x1)[1,...]
    x_origin = 0.5 * (x1 + x2)
    # Compute value of q5 at source
    return np.where(x_origin >= 0,
                    q5_interp(x_origin), # Interpolate q5 info at foot point
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
    _taud_coeff = (yWvHat - yWv) / (self.tau_f / self.tau_d - 1.0)
    _const_term = 1.0 - yWvHat
    _tauf_coeff = yF - (_taud_coeff + _const_term)
    yF_updated = np.where(vf_g > self.vf_g_crit,
                          _taud_coeff * np.exp(-dt / self.tau_d)
                            + _const_term
                            + _tauf_coeff * np.exp(-dt / self.tau_f),
                          yF)
    # Evolution operator[dt] within the split-step
    y_updated = np.stack([rhoprime,
                          yWv + (yWvHat - yWv) * (1 - np.exp(-dt / self.tau_d)),
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
    # Add hooks for external access
    self.q2 = q2
    self.q5 = q5
    # Construct initial q2 interpolant by slicing full steady state
    q2_interp = lambda x: self.steady_interp(x)[0:2,...]
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

    def get_q2_interp(solver, q5_interp_scalar, p0, xlims_unlimited, ivp_method,
                      ivp_max_step) -> callable:
      ''' Compute and return q2 interp. arg:solver is for storing debug state only. '''
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
      soln, scales = solver.solve_time_indep_system(q5_interp_scalar, p0=p0,
                                          u0=u0_solved,
                                          xlims=xlims_unlimited,
                                          method=ivp_method,
                                          max_step=ivp_max_step)
      # Cache u0 for next iteration
      solver.u0 = u0_solved
      # Shift solution so that choked flow is exactly at conduit_length
      #   This strategy extrapolates on the bottom instead of the top
      return lambda x: np.einsum("i..., i -> i...",
                                 soln.sol(x + (soln.t.max() - xlims[-1])),
                                 scales)

    # Unsteady solve
    for i in range(Nt):
      t = i * dt
      self.t = t
      clock_step = perf_counter()

      # Measure and log L2 norm of solution
      q5_L2 = np.sqrt(scipy.integrate.trapz(q5*q5, x=x, axis=1))
      self.logger.debug(f"q5 L2-norm:                    {q5_L2}")

      # Update source by analytic substep reaction (1/2)
      q5[:] = self.react_step(x, t, 0.5 * self.dt, q5, self.q2)
      # Measure and log L2 norm of solution
      q5_L2 = np.sqrt(scipy.integrate.trapz(q5*q5, x=x, axis=1))
      self.logger.debug(f"q5 L2-norm reacted      dt1/2: {q5_L2}")
      # Clip mass fractions
      clip_q5(self, q5, q2, clip_eps=1e-5)

      # Construct monotonic q5 interpolant
      q5_interp_raw = scipy.interpolate.PchipInterpolator(x, q5,
        extrapolate=True, axis=1)      
      # TODO: prevent undefined behaviour when called with a scalar, merge following two funcs
      def q5_interp(xq):
        ''' Vectorized returns '''
        return np.where(xq <= x.max(), q5_interp_raw(xq), q5[:,-1:])
      def q5_interp_scalar(xq):
        ''' Scalar returns'''
        return np.where(xq <= x.max(), q5_interp_raw(xq), q5[:,-1])
      self.q5_interp = q5_interp

      # q2 solve and obtain interpolator
      q2_interp = get_q2_interp(self, q5_interp_scalar, p0, xlims_unlimited,
                                ivp_method, ivp_max_step)
      self.q2_interp = q2_interp
      # Grid restriction for unsteady solver
      self.q2 = q2_interp(x)

      # Compute exact advection of interpolated q5 solution
      q5[:] = self.advect_step(
        BC_unsteady, x, t, self.dt, q5_interp, q2_interp, u0)
      # Measure and log L2 norm of solution
      q5_L2 = np.sqrt(scipy.integrate.trapz(q5*q5, x=x, axis=1))
      self.logger.debug(f"q5 L2-norm advected:           {q5_L2}")

      # Construct monotonic interpolants for indep system
      q5_interp_raw = scipy.interpolate.PchipInterpolator(x, q5,
        extrapolate=True, axis=1)      
      # TODO: prevent undefined behaviour when called with a scalar, merge following two funcs
      def q5_interp(xq):
        ''' Vectorized returns '''
        return np.where(xq <= x.max(), q5_interp_raw(xq), q5[:,-1:])
      def q5_interp_scalar(xq):
        ''' Scalar returns'''
        return np.where(xq <= x.max(), q5_interp_raw(xq), q5[:,-1])
      self.q5_interp = q5_interp

      # q2 solve and obtain interpolator
      q2_interp = get_q2_interp(self, q5_interp_scalar, p0, xlims_unlimited,
                                ivp_method, ivp_max_step)
      self.q2_interp = q2_interp
      # Grid restriction for unsteady solver
      self.q2 = q2_interp(x)

      # Update source by analytic substep reaction (2/2)
      q5[:] = self.react_step(x, t, 0.5 * self.dt, q5, self.q2)
      # Measure and log L2 norm of solution
      q5_L2 = np.sqrt(scipy.integrate.trapz(q5*q5, x=x, axis=1))
      self.logger.debug(f"q5 L2-norm reacted      dt2/2: {q5_L2}")
      # Clip mass fractions
      clip_q5(self, q5, q2, clip_eps=1e-5)

      # Construct monotonic interpolants for indep system
      q5_interp_raw = scipy.interpolate.PchipInterpolator(x, q5,
        extrapolate=True, axis=1)      
      # TODO: prevent undefined behaviour when called with a scalar, merge following two funcs
      def q5_interp(xq):
        ''' Vectorized returns '''
        return np.where(xq <= x.max(), q5_interp_raw(xq), q5[:,-1:])
      def q5_interp_scalar(xq):
        ''' Scalar returns'''
        return np.where(xq <= x.max(), q5_interp_raw(xq), q5[:,-1])
      self.q5_interp = q5_interp

      # q2 solve and obtain interpolator
      q2_interp = get_q2_interp(self, q5_interp_scalar, p0, xlims_unlimited,
                                ivp_method, ivp_max_step)
      self.q2_interp = q2_interp
      # Grid restriction for unsteady solver
      self.q2 = q2_interp(x)

      # Save outputs
      self.logger.debug(f"Solved step {i} in {perf_counter() - clock_step} s.")

      self.out_q2[i+1,...] = self.q2
      self.out_q5[i+1,...] = q5
      self.q5_messages.append(soln.message)

    self.logger.debug(f"Solve complete.")
    return q2, q5

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
  
  params = {
    "T0": 950+273.15,
    "conduit_radius": 10,
    "tau_d": 5.0, # CHECK, and Euler stability
    "tau_f": .25, # TODO: add resonant case for tau_f == tau_d
    "vf_g_crit": 0.8,
    "solubility_n": f.solubility_n,
    "solubility_k": f.solubility_k,
    "K": 10e9,
    "rho_m0": 2.7e3,
    "p_m0": 5e6,
    "R_wv": f.mixture.waterEx.R,
    "F_fric_viscosity_model": loc_F_fric_viscosity_model,
  }

  split_solver = SplitSolver(params, Nt=600)
  q2, q5 = split_solver.full_solve_choked((0,2100), Nx=1001, p0=80e6, u0=0.7, yWt0=0.025,
                  yC0=0.4, yF0=0.0, CFL=100)
  print(f"Final t: {split_solver.t} s")
  np.save("out_q2_strang_1", split_solver.out_q2)
  np.save("out_q5_strang_1", split_solver.out_q5)
  np.save("out_t_strang_1", split_solver.out_t)
  np.save("out_x_strang_1", split_solver.x)
