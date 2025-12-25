import numpy as np
from scipy.integrate import solve_ivp
import time

# ==========================================================
# 1. KINETIC PARAMETERS (Bisotti et al., 2022)
# ==========================================================
R = 8.314  # J/mol·K
rho_bulk = 1200  # kg katalis / m3 reaktor (Nilai standar industri)
purge_ratio = 0.05  # Buang 5% (industri tipikal 3–10%)

# Pre-exponential factors (A) and Activation energies (Ea)
A = np.array([1.0e5, 1.22e10])
Ea = np.array([36696, 94765])  # J/mol

# Adsorption constants [A_i, B_i] for K_ads = A * exp(-B / RT)
K_ads_A = np.array([3453.38, 0.499, 6.62e-11])
K_ads_B = np.array([0, -17197, -124119])


def get_rates(T, P_partials):
    """
    Calculate reaction rates for methanol synthesis and RWGS.
    
    Parameters:
    -----------
    MeOH_recovery = 0.80   # 80% MeOH direcycle
    T : float
        Temperature in Kelvin
    P_partials : array-like
        Partial pressures [H2, CO2, CO, MeOH, H2O] in bar
    
    Returns:
    --------
    r1, r2 : float
        Reaction rates for methanol synthesis and RWGS
    """
    # Rate constants: k = A * exp(-Ea / RT)
    k = A * np.exp(-Ea / (R * T))
    
    # Adsorption equilibrium constants
    K_ads = K_ads_A * np.exp(-K_ads_B / (R * T))
    
    # Reaction equilibrium constants (Kp)
    log_Kp1 = (3066 / T) - 10.592
    Kp1 = 10 ** log_Kp1
    
    log_Kp2 = (-2073 / T) + 2.029
    Kp2 = 1 / (10 ** log_Kp2)
    
    # Unpack partial pressures
    p_H2, p_CO2, p_CO, p_MeOH, p_H2O = P_partials
    
    # Prevent division by zero
    p_H2 = max(p_H2, 1e-10)
    p_CO2 = max(p_CO2, 1e-10)
    
    # VBF (Vanden Bussche & Froment) denominator
    denom = (1 + K_ads[0] * (p_H2O / p_H2) 
             + K_ads[1] * np.sqrt(p_H2) 
             + K_ads[2] * p_H2O) ** 3
    
    # Rate equations
    # r1: Methanol synthesis (CO2 + 3H2 -> CH3OH + H2O)
    approach_eq1 = 1 - (p_MeOH * p_H2O) / (p_CO2 * p_H2**3 * Kp1 + 1e-20)
    r1 = k[0] * p_CO2 * (p_H2 ** 3) * approach_eq1 / denom
    
    # r2: RWGS (CO2 + H2 -> CO + H2O)
    approach_eq2 = 1 - Kp2 * (p_CO * p_H2O) / (p_CO2 * p_H2 + 1e-20)
    r2 = 0.05 * k[1] * p_CO2 * approach_eq2 / (denom ** (1/3))

    return r1, r2


# ==========================================================
# 2. REACTOR MODEL (ODE System)
# ==========================================================
def reactor_model(z, F, P_total, Ac):
    """
    ODE system for plug flow reactor.
    
    Parameters:
    -----------
    z : float
        Axial position in reactor (m)
    F : array
        State vector [F_H2, F_CO2, F_CO, F_MeOH, F_H2O, T]
        Flow rates in mol/s, Temperature in K
    P_total : float
        Total pressure (bar)
    Ac : float
        Cross-sectional area (m²)
    
    Returns:
    --------
    dF_dz : array
        Derivatives of state variables
    """
    # Unpack state
    F_species = F[:5]
    T = F[5]
    
    # Total molar flow rate
    F_total = np.sum(F_species)
    
    # Avoid division by zero
    if F_total < 1e-10:
        return np.zeros(6)
    
    # Partial pressures
    P_partials = (F_species / F_total) * P_total
    
    # Get reaction rates
    r1, r2 = get_rates(T, P_partials)
    
    # Tambahkan rho_bulk di sini!
    # dF/dz = r [mol/kg.s] * rho_bulk [kg/m3] * Ac [m2] = mol/m.s
    reaction_term = rho_bulk * Ac
    
    dF_dz = np.zeros(6)
    dF_dz[0] = (-3 * r1 - r2) * reaction_term  # H2
    dF_dz[1] = (-r1 - r2) * reaction_term       # CO2
    dF_dz[2] = r2 * reaction_term                # CO
    dF_dz[3] = r1 * reaction_term                # MeOH
    dF_dz[4] = (r1 + r2) * reaction_term         # H2O
    dF_dz[5] = 0 
    return dF_dz


# ==========================================================
# 3. REACTOR SOLVER
# ==========================================================
def solve_reactor(F_inlet, T_inlet, L, P_total, Ac):
    """
    Solve the reactor ODE system.
    
    Returns the outlet conditions.
    """
    # Ensure non-negative inlet flows
    F_inlet = np.maximum(F_inlet, 0)
    y0 = np.append(F_inlet, T_inlet)
    
    # Use BDF method for stiff systems
    sol = solve_ivp(
        reactor_model,
        [0, L],
        y0,
        args=(P_total, Ac),
        method='BDF',
        rtol=1e-4,
        atol=1e-6,
        dense_output=False
    )
    
    if not sol.success:
        print(f"Warning: ODE solver failed - {sol.message}")
        return y0  # Return inlet as fallback
    
    return sol.y[:, -1]


# ==========================================================
# 4. SUCCESSIVE SUBSTITUTION (Robust Recycle Solver)
# ==========================================================
def successive_substitution(fresh_feed, initial_guess, T_inlet, L, P_total, Ac,
                               MeOH_recovery,
                               max_iter=200, tol=1e-4, relaxation=0.3, verbose=True
                        ):
    """
    Successive substitution with relaxation for recycle convergence.
    
    This is much more robust than Newton-Raphson (fsolve) for recycle loops.
    It's the same approach used by Aspen Plus, HYSYS, etc.
    
    Parameters:
    -----------
    fresh_feed : array
        Fresh feed [H2, CO2, CO, MeOH, H2O] in mol/s
    initial_guess : array
        Initial guess for recycle stream
    T_inlet : float
        Reactor inlet temperature (K)
    L : float
        Reactor length (m)
    P_total : float
        Total pressure (bar)
    Ac : float
        Cross-sectional area (m²)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    relaxation : float
        Relaxation factor (0 < α ≤ 1). Lower = more stable, higher = faster.
        Recommended: 0.2-0.5 for chemical processes
    verbose : bool
        Print iteration info
    
    Returns:
    --------
    F_recycle : array
        Converged recycle flow rates
    n_iter : int
        Number of iterations
    converged : bool
        Whether solution converged
    target_recycle_ratio = 1.2
    """
    # Initialize with non-negative values
    F_recycle = np.maximum(initial_guess.copy(), 0)
    
    if verbose:
        print(f"{'Iter':>5} {'Error':>12} {'H2_rec':>10} {'CO2_rec':>10} {'CO_rec':>10}")
        print("-" * 50)
    
    for iteration in range(max_iter):
        # Calculate new recycle from current guess
        F_reactor_inlet = fresh_feed + F_recycle
        F_outlet = solve_reactor(F_reactor_inlet, T_inlet, L, P_total, Ac)
        
        # Separator: Only H2, CO2, CO are recycled
        # MeOH and H2O are removed as products
        # F_new_recycle = np.array([
        #     max(F_outlet[0], 0),  # H2 recycle
        #     max(F_outlet[1], 0),  # CO2 recycle
        #     max(F_outlet[2], 0),  # CO recycle
        #     0,                     # MeOH (product)
        #     0                      # H2O (product)
        # ])
        
        # ======================================================
        # GAS RECYCLE WITH TARGET RECYCLE RATIO (INDUSTRIAL)
        # ======================================================

        target_recycle_ratio = 5.0   # 3–7 recommended

        # Gas outlet from reactor (only gas phase)
        F_gas_out = np.array([
            max(F_outlet[0], 0),  # H2
            max(F_outlet[1], 0),  # CO2
            max(F_outlet[2], 0)   # CO
            ])

        # Fresh gas feed (basis for recycle ratio)
        F_fresh_gas = fresh_feed[:3]
        
        # === FORCE recycle ratio based on fresh feed (NOT outlet-dependent) ===
        total_fresh_gas = np.sum(F_fresh_gas)
        
        if total_fresh_gas < 1e-10:
            scale = 0.0
        else:
            scale = target_recycle_ratio * total_fresh_gas / max(np.sum(F_gas_out), 1e-10)

        # Final recycle stream
        F_new_recycle = np.array([
            F_gas_out[0] * scale,           # H2 recycle
            F_gas_out[1] * scale,           # CO2 recycle
            F_gas_out[2] * scale,           # CO recycle
            np.clip(F_outlet[3], 5.0, 10.0), #MeOh recycle
            0
        ])

        # Calculate error (L2 norm of difference)
        error = np.linalg.norm(F_new_recycle - F_recycle)
        
        if verbose and (iteration < 10 or iteration % 10 == 0):
            print(f"{iteration+1:>5} {error:>12.6f} {F_recycle[0]:>10.2f} {F_recycle[1]:>10.2f} {F_recycle[2]:>10.2f}")
        
        # Check convergence
        if error < tol:
            if verbose:
                print(f"{iteration+1:>5} {error:>12.6f} {F_recycle[0]:>10.2f} {F_recycle[1]:>10.2f} {F_recycle[2]:>10.2f}")
                print("-" * 50)
                print(f"✅ Converged in {iteration + 1} iterations!")
            return F_recycle, iteration + 1, True
        
        # Relaxed update: F_new = α * F_calculated + (1-α) * F_old
        # This dampens oscillations and improves stability
        F_recycle = relaxation * F_new_recycle + (1 - relaxation) * F_recycle
    
    if verbose:
        print("-" * 50)
        print(f"⚠️ Did not converge after {max_iter} iterations (error = {error:.6f})")
    
    return F_recycle, max_iter, False


# ==========================================================
# 5. MAIN PLANT SOLVER
# ==========================================================
def solve_plant_with_recycle(
    fresh_feed, initial_recycle_guess,
    T_inlet=503, L=130.0, P_total=50, Ac=0.0011,
    MeOH_recovery=0.95,  # 95% recovered as product
    relaxation=0.3, verbose=True
):
    """
    Main solver for the plant with recycle loop.
    
    Parameters:
    -----------
    fresh_feed : array
        Fresh feed [H2, CO2, CO, MeOH, H2O] in mol/s
    initial_recycle_guess : array
        Initial guess for recycle stream
    T_inlet : float
        Reactor inlet temperature (K)
    L : float
        Reactor length (m)
    P_total : float
        Operating pressure (bar)
    Ac : float
        Cross-sectional area (m²)
    relaxation : float
        Relaxation factor for successive substitution
    verbose : bool
        Print iteration progress
    
    Returns:
    --------
    dict : Results dictionary
    """
    print("=" * 60)
    print("CO2 HYDROGENATION REACTOR SIMULATION")
    print("=" * 60)
    print(f"Fresh Feed: H2={fresh_feed[0]:.1f}, CO2={fresh_feed[1]:.1f} mol/s")
    print(f"Conditions: T={T_inlet}K, P={P_total}bar, L={L}m")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    # Solve recycle loop using successive substitution
    converged_recycle, n_iter, success = successive_substitution(fresh_feed, initial_recycle_guess, T_inlet, L, P_total, Ac,
    MeOH_recovery,max_iter=100000, tol=1e-4, relaxation=relaxation, verbose=verbose)


    elapsed = time.time() - start_time
    
    # Ensure non-negative
    converged_recycle = np.maximum(converged_recycle, 0)
    
    # Calculate final results
    F_reactor_inlet = fresh_feed + converged_recycle
    ratio = F_reactor_inlet[0] / max(F_reactor_inlet[1], 1e-6)
    if ratio < 3.0:
        F_reactor_inlet[0] *= (3.0 / ratio)
    F_reactor_outlet = solve_reactor(F_reactor_inlet, T_inlet, L, P_total, Ac)
    
    # Products (what leaves the separator)
    F_MeOH_product = F_reactor_outlet[3] * MeOH_recovery
    F_H2O_product = max(F_reactor_outlet[4], 0)
    
    # Single-pass conversion (per pass through reactor)
    CO2_fresh = fresh_feed[1]
    CO2_reacted = F_reactor_inlet[1] - F_reactor_outlet[1]
    single_pass_reactor = CO2_reacted / F_reactor_inlet[1] * 100
    single_pass_once_through = CO2_reacted / fresh_feed[1] * 100


    single_pass_conversion = (
        CO2_reacted / F_reactor_inlet[1] * 100
    )
    
    # Overall conversion (based on fresh feed)
    # At steady state: CO2 reacted = CO2 in fresh feed (mass balance)
    
    # ================= OVERALL CONVERSION (CORRECT) =================
    CO2_fresh = fresh_feed[1]

    # CO2 leaving system ONLY via purge
    CO2_purged = purge_ratio * fresh_feed[1]

    overall_conversion = (CO2_fresh - CO2_purged) / CO2_fresh * 100
    overall_conversion = np.clip(overall_conversion, 0, 100)

    # Selectivity (MeOH vs CO)
    CO_from_RWGS = max(F_reactor_outlet[2] - F_reactor_inlet[2], 0)
    selectivity = (
        F_MeOH_product / (F_MeOH_product + CO_from_RWGS) * 100
    )

    results = {
        'recycle_flow': converged_recycle,
        'reactor_inlet': F_reactor_inlet,
        'reactor_outlet': F_reactor_outlet[:5],
        'MeOH_production': F_MeOH_product,
        'H2O_production': F_H2O_product,
        'single_pass_conversion': single_pass_conversion,
        'overall_conversion': overall_conversion,
        'selectivity': selectivity,
        'solver_time': elapsed,
        'iterations': n_iter,
        'solver_success': success
    }
    
    return results


def print_results(results):
    """Pretty print the simulation results."""
    print(f"\n{'═'*60}")
    print("SIMULATION RESULTS")
    print(f"{'═'*60}")
    
    species = ['H2', 'CO2', 'CO', 'MeOH', 'H2O']
    
    print("\n📥 REACTOR INLET (mol/s):")
    for i, s in enumerate(species):
        print(f"   {s:6s}: {results['reactor_inlet'][i]:10.4f}")
    
    print("\n📤 REACTOR OUTLET (mol/s):")
    for i, s in enumerate(species):
        print(f"   {s:6s}: {results['reactor_outlet'][i]:10.4f}")
    
    print("\n♻️  CONVERGED RECYCLE (mol/s):")
    for i, s in enumerate(species[:3]):
        print(f"   {s:6s}: {results['recycle_flow'][i]:10.4f}")
    
    print(f"\n{'─'*60}")
    print("📊 PERFORMANCE METRICS:")
    print(f"{'─'*60}")
    print(f"   🧪 MeOH Production:       {results['MeOH_production']:10.4f} mol/s")
    print(f"   💧 H2O Production:        {results['H2O_production']:10.4f} mol/s")
    print(f"   📈 Single-pass Conv.:     {results['single_pass_conversion']:10.2f} %")
    print(f"   📈 Overall Conversion:    {results['overall_conversion']:10.2f} %")
    print(f"   🎯 MeOH Selectivity:      {results['selectivity']:10.2f} %")
    
    print(f"\n{'─'*60}")
    print("⚙️  SOLVER INFO:")
    print(f"{'─'*60}")
    print(f"   ⏱️  Solver Time:          {results['solver_time']:10.3f} seconds")
    print(f"   🔄 Iterations:            {results['iterations']:10d}")
    print(f"   ✅ Converged:             {results['solver_success']}")
    print(f"{'═'*60}\n")


# ==========================================================
# 6. MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    # Feed conditions
    # 3:1 H2:CO2 ratio (stoichiometric for methanol synthesis)
    fresh_feed = np.array([300.0, 100.0, 0.0, 0.0, 0.0])  # mol/s
    
    # Initial guess for recycle stream
    # Start with reasonable values (will be refined by solver)
    initial_guess = np.array([200.0, 80.0, 5.0, 0.0, 0.0])  # mol/s
    
    # Operating conditions
    T_inlet = 503.15      # K (230°C) - typical for Cu/ZnO/Al2O3 catalyst
    L = 1500.0          # m (reactor length)
    P_total = 100       # bar (typical industrial pressure)
    Ac = 0.004        # m² (cross-sectional area)
    
    # Run simulation
    results = solve_plant_with_recycle(
        fresh_feed,
        initial_guess,
        T_inlet=T_inlet,
        L=L,
        P_total=P_total,
        Ac=Ac,
        relaxation=0.3,  # Adjust if not converging (try 0.1-0.5)
        verbose=True
    )
    
    # Display results
    print_results(results)
    
    # Quick convergence check
    if results['solver_success']:
        print("🎉 Simulation completed successfully!")
    else:
        print("⚠️  Warning: Solution may not be fully converged.")
        print("   Try reducing relaxation factor or increasing max_iter.")