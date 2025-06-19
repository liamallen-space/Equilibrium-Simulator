# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:56:53 2025

@author: Liam Allen 46988601
"""

#%% IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import math
import json
#import multiprocessing as mp
from scipy.optimize import fsolve, newton, bisect
from scipy.interpolate import interp1d
import time

#%% CONSTANT DEFINITIONS
numSpecies = 5
calToJoules = 4.184     # [J/cal]
atmToPa = 101325.       # [Pa/atm]
Ru = 8.314              # [J/kg/mol]
noseRadius = 0.25 # [m]
diameter = 1.     # [m]
conicalShockWaveAngle = 48.4 * np.pi / 180 # [rad]

T1 = 269.4                                       # temperature [K]
P1 = 130.7126                                    # pressure [Pa]
rho1 = 1.69e-3                                   # density [kg/m3]
u1 = 4440.                                       # velocity [m/s]
X1 = [0., 0.79, 0., 0., 0.21]                    # moll fractions
gamma1 = 1.4                                     # specific weight

#%% GENERAL HELPER FUNCTIONS
def init_data(filename: str) -> dict:
    with open(filename, "r") as inputFile:
        data = json.load(inputFile)
    return data

def solve_specific_gas_constants_for_species(gasProperties: dict) -> np.array:
    moleMasses = gasProperties["moleMasses"]
    Rs = Ru / moleMasses
    return Rs

def partial_pressure_equations_for_air( # units some order of [atm]
        partialPressures: np.array, reactions: np.array, Kp: np.array, P2: float) -> list:
    pN, pN2, pNO, pO, pO2 = partialPressures.T
    eq1 = np.prod(np.power(partialPressures.T, reactions[0])) - Kp[0]
    eq2 = np.prod(np.power(partialPressures.T, reactions[1])) - Kp[1]
    eq3 = np.prod(np.power(partialPressures.T, reactions[2])) - Kp[2]
    eq4 = np.sum(partialPressures.T) - P2 / atmToPa
    eq5 = (2 * pN2 + pN + pNO) / (2 * pO2 + pO + pNO) - (79 / 21)

    #eq1 = pN ** 2 / pN2 - Kp[0]
    #eq2 = pO ** 2 / pO2 - Kp[1]
    #eq3 = pNO / (pN2 ** 0.5 * pO2 ** 0.5) - Kp[2]
   
    #print([eq1, eq2, eq3, eq4, eq5])
    #print(partialPressures.T)
    return [eq1, eq2, eq3, eq4, eq5]

def solve_Kp(data: dict, T2: float, gasProperties: dict) -> np.array:
    species = gasProperties["species"]
    reactions = gasProperties["reactions"]
    temperatureData = np.array(data["temperatures"])
    HtData = np.zeros((len(species), len(temperatureData)))
    FtMinusHzeroData = np.zeros_like(HtData)
    for i in range(len(species)):
        HtData[i] = data[species[i]]["Ht_zero"]
        FtMinusHzeroData[i] = data[species[i]]["Ft_minus_Hzero"]
    solveHt = interp1d(temperatureData, HtData)
    solveFtMinusHzero = interp1d(temperatureData, FtMinusHzeroData)
    Ht0 = solveHt(0.) # Ht @ T = 0K
    FtMinusHzero = solveFtMinusHzero(T2)
    G = (-FtMinusHzero + Ht0) * calToJoules # Gibbs 
    dG = np.dot(reactions, G)
    Kp = np.exp(-dG / (Ru * T2)) # uses atm, not Pa
    return Kp

def solve_postshock_mole_fractions(partialPressures: np.array, P2: float) -> np.array:
    moleFractions = partialPressures / P2
    return moleFractions

def solve_mass_fractions(moleFractions: np.array, gasProperties: dict) -> np.array:
    moleMasses = gasProperties["moleMasses"]
    molarMass = np.dot(moleFractions, moleMasses)
    massFractions = moleFractions * moleMasses / molarMass
    return massFractions

def solve_specific_gas_constant(massFractions: np.array, Rs: np.array) -> float:
    R = np.dot(massFractions, Rs)
    return R

def solve_specific_gas_constant2(moleFractions: np.array, moleMasses: np.array) -> float:
    molarMass = np.dot(moleFractions, moleMasses)
    R = Ru / molarMass
    return R

def solve_enthalpies(data: dict, T: float, gasProperties: dict) -> np.array:
    species = gasProperties["species"]
    moleMasses = gasProperties["moleMasses"]
    temperatureData = np.array(data["temperatures"])
    HtData = np.zeros((len(species), len(temperatureData)))
    for i in range(len(species)):
        HtData[i] = data[species[i]]["Ht_zero"]
    solveHt = interp1d(temperatureData, HtData)
    Hts = solveHt(T)
    hs = Hts / moleMasses * calToJoules
    return hs

def solve_enthalpy(massFractions: np.array, hs: np.array) -> float:
    h = np.dot(massFractions, hs)
    return h

def solve_sound_speed(gamma, R, T) -> float:
    a = np.sqrt(gamma * R * T)
    return a

def solve_mach_number(u, a) -> float:
    M = u / a
    return M

def minimise_h_error(Tguess, hPrev: float, P2: float, partialPressuresGuess: np.array, 
                     data: dict, gasProperties: dict) -> float:
    #Tguess = Tguess.item()
    h2, R2, X2, c2 = solve_post_shock_equilibrium(data, Tguess, P2, partialPressuresGuess, 
                                                  gasProperties)
    return h2 - hPrev

# SHOCK SHAPE HELPER FUNCTIONS
def solve_shock_radius_at_vertex(noseRadius: float, M1: float) -> float:
    Rc = 1.143 * noseRadius * np.exp(0.54 / (M1 - 1.) ** 1.2)
    return Rc # [m]
    
def solve_shock_standoff(noseRadius: float, M1: float) -> float:
    shockStandoff = 0.143 * noseRadius * np.exp(3.24 / M1 ** 2)
    return shockStandoff # [m]
    
def solve_shock_xs(y: np.array, noseRadius: float, Rc: float, shockStandoff: float, 
                   conicalShockWaveAngle: float) -> np.array:
    xs = noseRadius + shockStandoff - Rc * np.tan(conicalShockWaveAngle) ** -2 * (
            np.sqrt(1 + y ** 2 * np.tan(conicalShockWaveAngle) ** 2 / noseRadius ** 2) - 1)
    return xs # [m]
    
def solve_oblique_shock_angle(ys: np.array, noseRadius: float, Rc: float, 
                              shockStandoff: float, conicalShockWaveAngle: float,):
    dydx = -1. * (Rc * np.sqrt(np.square(ys * np.tan(conicalShockWaveAngle) / Rc) + 1)) / ys
    beta = np.arctan(np.abs(dydx))
    #print(beta)
    return beta # [rads]
    
def solve_preshock_normal_velocity(u1: np.array, beta: np.array) -> np.array:
    u1n = u1 * np.sin(beta)
    return u1n

def solve_preshock_normal_mach_number(M1: float, beta: np.array) -> np.array:
    M1n = M1 * np.sin(beta)
    return M1n

def solve_postshock_normal_velocity(u1n: np.array, gamma1: float, M1: float, 
                                    beta: np.array) -> np.array:
    B = M1 * M1 * np.square(np.sin(beta))
    u2n = u1n * (2 + (gamma1 - 1) * B) / ((gamma1 + 1) * B)
    return u2n

def solve_post_shock_velocity(u1: float, h1: float, h2: np.array) -> np.array:
    u2 = np.sqrt(2 * (-1 * h2 + h1) + u1 ** 2)
    return u2

#%% EQUILIBRIUM SOLVER

def solve_post_shock_equilibrium(data: dict, T2: float, P2: float, 
                                 partialPressuresGuess: np.array, gasProperties: dict):
    #print(f"T2 eq: {T2}")
    equations = gasProperties["equations"]
    reactions = gasProperties["reactions"]
    moleMasses = gasProperties["moleMasses"]
    Kp = solve_Kp(data, T2, airProperties)
    partialPressures, info, ier, mesg = fsolve(equations, partialPressuresGuess, 
                                               args=(reactions, Kp, P2), full_output=1)
    #print(f"{ier}: partialPressures error: {partialPressures - partialPressuresGuess}", flush=True)
    if ier != 1:
        print(f"err {ier}: partialPressures", flush=True)
    X2 = solve_postshock_mole_fractions(partialPressures * atmToPa, P2)
    c2 = solve_mass_fractions(X2, airProperties)
    #Rs2 = solve_specific_gas_constants_for_species(airProperties)
    #R2 = solve_specific_gas_constant(c2, Rs2)
    R2 = solve_specific_gas_constant2(X2, moleMasses)
    hs2 = solve_enthalpies(data, T2, gasProperties)
    h2 = solve_enthalpy(c2, hs2)
    return h2, R2, X2, c2

def solve_post_shock_state(Tguess, P1: float, rho1: float, u1: float, h1: float,
        u2: float, rho2: float, partialPressuresGuess: np.array, data: dict, 
        gasProperties: dict) -> float:
    moleMasses = gasProperties["moleMasses"]
    Pmom = P1 + rho1 * u1 * u1
    Htot = h1 + 0.5 * u1 * u1
    T2 = Tguess
    Tprev = 0.
    while not math.isclose(Tprev, T2):
    #for i in range(5):
        P2 = Pmom - rho2 * u2 * u2
        h2 = Htot - 0.5 * u2 * u2
        #print('-'*60)
        #print(f"P2: {P2}")
        #print(f"h2: {h2}")
        #print(f"rho2: {rho2}")
        #print(f"u2: {u2}")
        #print(f"partialPressures: {partialPressuresGuess}")
        #T2 = (fsolve(minimise_h_error, Tguess, args=(h2, P2, data, gasProperties))).item()
        T2, info = newton(minimise_h_error, x0=Tguess, x1=Tguess*0.95, 
                          args=(h2, P2, partialPressuresGuess, data, gasProperties), 
                          full_output=True, disp=True)
        h2, R2, X2, c2 = solve_post_shock_equilibrium(data, T2, P2, partialPressuresGuess, 
                                                      gasProperties)
        rho2 = P2 / (R2 * T2)
        u2 = rho1 * u1 / rho2
        Tprev = Tguess
        Tguess = T2
        partialPressuresGuess = X2 * P2 / atmToPa
        #print(f"\nT2 =\t{T2} K", flush=True)
        #print(f"Tprev =\t{Tprev} K", flush=True)
        #print(f"converged = {info.converged}\n", flush=True)
    return T2, P2, rho2, u2, h2, R2, X2, c2

def solve_post_shock_state_arr(Tguess, P1: float, rho1: float, u1n: np.array, h1: float,
        u2n: np.array, rho2: np.array, partialPressuresGuess: np.array, 
        data: dict, gasProperties: dict) -> float:
    species = gasProperties["species"]
    T2 = np.zeros_like(u1n)
    P2 = np.zeros_like(u1n)
    h2 = np.zeros_like(u1n)
    R2 = np.zeros_like(u1n)
    X2 = np.zeros((len(species), len(u1n)))
    c2 = np.zeros((len(species), len(u1n)))
    for i in range(len(u1n)):
        if i != 0:
            Tguess = T2[i - 1]
            partialPressuresGuess = X2[:,i - 1] * P2[i - 1] / atmToPa
            u2n[i] = u2n[i - 1]
            rho2[i] = rho2[i - 1]
        T2[i], P2[i], rho2[i], u2n[i], h2[i], R2[i], X2[:,i], c2[:,i] = solve_post_shock_state(
            Tguess, P1, rho1, u1n[i], h1, u2n[i], rho2[i], partialPressuresGuess, data, gasProperties)
    return T2, P2, rho2, u2n, h2, R2, X2, c2

def solve_shock_shape(elements: int, noseRadius: float, conicalShockWaveAngle: float, 
                      gamma1: float, M1: float, u1: float) -> None:
    Rc = solve_shock_radius_at_vertex(noseRadius, M1)
    shockStandoff = solve_shock_standoff(noseRadius, M1)
    minY = 0.
    maxY = .75
    ys = np.linspace(minY, maxY, elements)
    xs = solve_shock_xs(ys, noseRadius, Rc, shockStandoff, conicalShockWaveAngle)
    #u2n = solve_postshock_normal_velocity(u1n, gamma1, M1, beta)
    #theta = self.__solve_thetas(obliqueShockAngle)
    #M = self.__solve_post_oblique_shock_mach_number(obliqueShockAngle, theta)
    return xs, ys

#%% IN DEVELOPMENT - DO NOT USE

def solve_ideal_post_shock_state(T1: float, gamma1: float, rho1: float, u1n: np.array, 
                                 beta: np.array) -> np.array:
    M1n = solve_preshock_normal_mach_number(M1, beta)
    M1nsquared = np.square(M1n)
    T2ideal = T1 * (2 * gamma1 * M1nsquared - (gamma1 - 1)) * (
        (gamma1 - 1) * M1nsquared + 2) / ((gamma1 - 1) ** 2 * M1nsquared)
    P2ideal = P1 * (2 * gamma1 * M1nsquared - (gamma1 - 1)) / (gamma1 + 1)
    M2n = np.sqrt(((gamma1 - 1) * M1nsquared + 2) / (2 * gamma1 * M1nsquared - (gamma1 - 1)))
    
    rho2ideal = P2ideal / (R2 * T2ideal)
    u2nideal = rho1 * u1n / rho2ideal
    return T2ideal, u2nideal, rho2ideal
    
def solve_post_shock_state_arr_v2(Tguess: np.array, P1: float, rho1: float, 
                                  u1n: np.array, h1: float, u2n: np.array, 
                                  rho2: np.array, partialPressuresGuess: np.array, 
                                  data: dict, gasProperties: dict) -> float:
    species = gasProperties["species"]
    T2 = np.zeros_like(u1n)
    P2 = np.zeros_like(u1n)
    h2 = np.zeros_like(u1n)
    R2 = np.zeros_like(u1n)
    X2 = np.zeros((len(species), len(u1n)))
    c2 = np.zeros((len(species), len(u1n)))

    T2, P2, rho2, u2n, h2, R2, X2, c2 = solve_post_shock_state(
        Tguess, P1, rho1, u1n, h1, u2n, rho2, partialPressuresGuess, data, gasProperties)
    return T2, P2, rho2, u2n, h2, R2, X2, c2

#%% PLOTTING ASSIST
def make_sphere_cone(elements: int, noseRadius: float) -> list:
    # REF: The following code was generated using ai and then modified by me
    # REF: to simplify the process of drawing the capsule geomtry in the shock
    # REF: shape figure.
    
    # Parameters
    Rn = noseRadius # nose radius in meters
    theta_deg = 45 # cone half-angle
    theta_rad = np.radians(theta_deg)

    # Define tangent point between sphere and cone
    x_tangent = Rn * np.sin(theta_rad)
    y_tangent = Rn * (1 - np.cos(theta_rad))

    # Generate spherical nose
    phi = np.linspace(0, theta_rad, elements)
    x_sphere = Rn * np.sin(phi)
    y_sphere = Rn * (1 - np.cos(phi))

    # Generate cone section
    cone_length = 1  # arbitrary extension
    x_cone = np.linspace(x_tangent, x_tangent + cone_length * np.cos(theta_rad), elements)
    y_cone = y_tangent + (x_cone - x_tangent) * np.tan(theta_rad)

    # Combine x and y into full coordinates
    x_full = np.concatenate((x_sphere, x_cone))
    y_full = np.concatenate((y_sphere, y_cone))

    # Apply 45° counter-clockwise rotation
    rotation_angle = np.radians(90)
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)

    xCone = cos_a * x_full - sin_a * y_full
    yCone = sin_a * x_full + cos_a * y_full
    
    xClosed = np.concatenate((xCone, xCone[::-1]))
    yClosed = np.concatenate((yCone, np.zeros_like(yCone[::-1])))    
        
    return xCone, yCone, xClosed, yClosed

#%% EXECUTE PROGRAM
if __name__ == "__main__":
    start = time.time()
    
    # Define air properties in dictionary
    airProperties = dict(
        species = ["N", "N2", "NO", "O", "O2"],
        moleMasses = np.array([0.014, 0.028, 0.030, 0.016, 0.032]).T,   # [kg/mol]
        reactions = np.array([[2., -1. ,  0., 0.,  0. ],     # N2->2N
                              [0.,  0. ,  0., 2., -1. ],     # O2->2O
                              [0., -0.5,  1., 0., -0.5]]),   # .5N2+.5O2->NO
                              #[-1.,  0. , 1., -1.,  0. ]]),
        equations = partial_pressure_equations_for_air)

    # Initialise data
    inputFilename = "mcbride_data.json"
    data = init_data(inputFilename)
    
    # Calculate pre shock state
    c1 = solve_mass_fractions(X1, airProperties)
    R1 = solve_specific_gas_constant2(X1, airProperties["moleMasses"])
    hs1 = solve_enthalpies(data, T1, airProperties)
    h1 = solve_enthalpy(c1, hs1)
    a1 = solve_sound_speed(gamma1, R1, T1)
    M1 = solve_mach_number(u1, a1)
    
    # Define guess for post shock partial pressures [atm], informed from CEA
    partialPressuresGuess = np.array([.016*.3, .7*.3, .015*.3, .22*.3, .001*.3]).T
    
    # Solve post shock equilibrium with Bertin conditions
    BERTIN = False
    if BERTIN:
        T2 = 4770.
        P2 = 30380.771
        rho2 = 18.37e-3
        u2 = 408.426
        
        h2, R2, X2, c2 = solve_post_shock_equilibrium(
            data, T2, P2, partialPressuresGuess, airProperties)
        print(f"final h2 = {h2 * 1e-3:.2f} kJ/kg")
        print(f"final R2 = {R2:.3f} J/kgK")
        np.set_printoptions(precision=4)
        print(f"final X2 = {X2}")
        print(f"final c2 = {c2}")

    # Solve post shock temperature iteratively
    STAGNATION = False
    if STAGNATION:
        Tguess = 4000.

        u2guess = 0.
        rho2guess = 0.

        T2, P2, rho2, u2, h2, R2, X2, c2 = solve_post_shock_state(
            Tguess, P1, rho1, u1, h1, u2guess, rho2guess, 
            partialPressuresGuess, data, airProperties)
        print(f"final T2 = {T2:.2f} K")
        print(f"final P2 = {P2 * 1e-3:.3f} kPa")
        print(f"final rho2 = {rho2:.7f} kg/m3")
        print(f"final u2 = {u2:.3f} m/s")
        print(f"final h2 = {h2 * 1e-3:.2f} kJ/kg")
        print(f"final R2 = {R2:.3f} J/kgK")
        np.set_printoptions(precision=7)
        print(f"final X2 = {X2}")
        print(f"final c2 = {c2}")
    
    # Solve conditions along entire shock length
    SHOCK = True
    if SHOCK:
        elements = 100
        Tguess = 4000.

        Rc = solve_shock_radius_at_vertex(noseRadius, M1)
        shockStandoff = solve_shock_standoff(noseRadius, M1)
        xs, ys = solve_shock_shape(elements, noseRadius, conicalShockWaveAngle, 
                                   gamma1, M1, u1)
        beta = solve_oblique_shock_angle(ys, noseRadius, Rc, shockStandoff, 
                                         conicalShockWaveAngle)
        u1n = solve_preshock_normal_velocity(u1, beta)
        
        u2n = np.zeros_like(u1n)
        rho2 = np.zeros_like(u1n)
        T2, P2, rho2, u2n, h2, R2, X2, c2 = solve_post_shock_state_arr(
            Tguess, P1, rho1, u1n, h1, u2n, rho2, partialPressuresGuess, data, 
            airProperties)
        
        TEST = False
        if TEST:
            T2guess, rho2guess, u2nguess = solve_ideal_post_shock_state(
                T1, gamma1, rho1, u1n, beta)
            T2, P2, rho2, u2n, h2, R2, X2, c = solve_post_shock_state_v2(
                T2guess, P1, rho1, u1n, h1, u2nguess, rho2guess, 
                partialPressuresGuess, data, airProperties)

        u2 = solve_post_shock_velocity(u1, h1, h2)
        
        # Define ratios for plotting
        Tratio = T2 / T1
        Pratio = P2 / P1
        rhoratio = rho2 / rho1
        uratio = u2 / u1
        
        # Set x limit
        xlim = [-0.25, 0.3]
        
        # Plot property ratios
        fig1, axs1 = plt.subplots(1, 1)
        axs1.plot(xs, Tratio, label='T2 / T1')
        axs1.plot(xs, Pratio, label='P2 / P1')
        axs1.plot(xs, rhoratio, label='rho2 / rho1')
        axs1.plot(xs, uratio, label='u2 / u1')
        axs1.set_yscale('log')
        axs1.set_xlim(xlim)
        axs1.legend()
        axs1.set_xlabel("x [m]")
        
        # Plot Mole Fractions
        fig2, axs2 = plt.subplots(1, 1)
        axs2.plot(xs, X2.T, label=airProperties["species"])
        axs2.set_yscale('log')
        axs2.set_xlim(xlim)
        axs2.legend()
        axs2.set_xlabel("x [m]")

        # Plot shock shape
        xCone, yCone, xClosed, yClosed = make_sphere_cone(elements, noseRadius)
        fig3, axs3 = plt.subplots(1, 1)
        axs3.plot(xs, ys)
        axs3.plot(xCone, yCone, color='gray')
        axs3.fill(xClosed, yClosed, color='black', alpha=0.05)
        axs3.set_aspect('equal', adjustable='box')
        axs3.set_xlim(xlim)
        axs3.set_ylim([0, 0.8])
        axs3.set_xlabel("x [m]") # [\SI{1}{m/s}]$
        axs3.set_ylabel("y [m]")
        
        fig4, axs4 = plt.subplots(1, 1)
        axs4.plot(xs, Tratio, label='T2 / T1')
        axs4.set_xlim(xlim)
        axs4.set_xlabel("x [m]")

        fig5, axs5 = plt.subplots(1, 1)
        axs5.plot(xs, Pratio, label='P2 / P1')
        axs5.set_xlim(xlim)
        axs5.set_xlabel("x [m]")

        fig6, axs6 = plt.subplots(1, 1)
        axs6.plot(xs, rhoratio, label='rho2 / rho1')
        axs6.set_xlim(xlim)
        axs6.set_xlabel("x [m]")

        fig7, axs7 = plt.subplots(1, 1)
        axs7.plot(xs, uratio, label='u2 / u1')
        axs7.set_xlim(xlim)
        axs7.set_xlabel("x [m]")

        plt.show()

    end = time.time()
    
    print(f"ellapsed time: {end - start} s")
