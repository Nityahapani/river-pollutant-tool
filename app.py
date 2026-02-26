import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. APP CONFIGURATION & UI ---
st.set_page_config(page_title="River Pollutant Digital Twin", layout="wide")
st.title("🌊 River Pollutant Life Cycle Simulator")
st.markdown("Interactive stochastic model of multi-species pollutant transport and decay.")

# Sidebar for User Inputs
st.sidebar.header("River Conditions")
velocity = st.sidebar.slider("Average Flow Velocity (m/s)", 0.1, 2.0, 0.6, 0.1)
temp = st.sidebar.slider("Water Temperature (°C)", 0.0, 35.0, 22.0, 1.0)

st.sidebar.header("Spill Characteristics")
conc_A = st.sidebar.number_input("Initial Ammonia (mg/L)", min_value=0.0, value=15.0)
conc_B = st.sidebar.number_input("Initial Nitrate (mg/L)", min_value=0.0, value=0.5)

st.sidebar.header("Simulation Parameters")
iterations = st.sidebar.slider("Monte Carlo Iterations", 10, 500, 100, 10)
reg_limit = st.sidebar.number_input("Regulatory Limit (mg/L)", value=2.0)

# Constants
DIST_MAX = 50000  # Simulate up to 50 km downstream
X_EVAL = np.linspace(0, DIST_MAX, 500)

# --- 2. MATHEMATICAL MODEL ---
def simulate_river(t, C, v, current_temp):
    """
    Solves the Advection-Reaction differential equations.
    """
    # Arrhenius temperature correction for biological/chemical decay
    k1 = (0.4 / 86400) * (1.047**(current_temp - 20)) 
    k2 = (0.15 / 86400) * (1.047**(current_temp - 20))
    
    dA_dx = (-k1 / v) * C[0]
    dB_dx = ((k1 / v) * C[0]) - ((k2 / v) * C[1])
    return [dA_dx, dB_dx]

# --- 3. MONTE CARLO ENGINE ---
with st.spinner('Running probabilistic simulations...'):
    all_results_A = []
    all_results_B = []
    
    for _ in range(iterations):
        # Introduce 10% natural variability to velocity and temperature
        v_rand = velocity * np.random.uniform(0.9, 1.1)
        temp_rand = temp * np.random.uniform(0.9, 1.1)
        
        sol = solve_ivp(
            simulate_river, 
            (0, DIST_MAX), 
            [conc_A, conc_B], 
            t_eval=X_EVAL, 
            args=(v_rand, temp_rand)
        )
        
        all_results_A.append(sol.y[0])
        all_results_B.append(sol.y[1])

# Calculate Statistics
A_mean = np.mean(all_results_A, axis=0)
A_std = np.std(all_results_A, axis=0)
B_mean = np.mean(all_results_B, axis=0)
B_std = np.std(all_results_B, axis=0)

# --- 4. DATA VISUALIZATION ---
st.subheader(f"Downstream Plume Analysis (Simulated up to {DIST_MAX/1000} km)")

fig, ax = plt.subplots(figsize=(10, 5))
plt.style.use('bmh')

# Plot Pollutant A
ax.plot(X_EVAL/1000, A_mean, color='#d62728', label='Ammonia (Mean)', lw=2)
ax.fill_between(X_EVAL/1000, np.maximum(0, A_mean - 2*A_std), A_mean + 2*A_std, 
                color='#d62728', alpha=0.2, label='95% Confidence (Ammonia)')

# Plot Pollutant B
ax.plot(X_EVAL/1000, B_mean, color='#1f77b4', label='Nitrate (Mean)', lw=2)
ax.fill_between(X_EVAL/1000, np.maximum(0, B_mean - 2*B_std), B_mean + 2*B_std, 
                color='#1f77b4', alpha=0.2, label='95% Confidence (Nitrate)')

# Formatting
ax.axhline(reg_limit, color='black', linestyle='--', alpha=0.7, label='Regulatory Limit')
ax.set_xlabel("Distance Downstream (km)", fontsize=12)
ax.set_ylabel("Concentration (mg/L)", fontsize=12)
ax.legend(loc='upper right')

# Render plot in Streamlit
st.pyplot(fig)

# --- 5. KEY METRICS ---
st.subheader("Actionable Insights")
col1, col2 = st.columns(2)

# Calculate where the primary pollutant drops below the regulatory limit
safe_indices = np.where(A_mean < reg_limit)[0]
if len(safe_indices) > 0:
    safe_distance = X_EVAL[safe_indices[0]] / 1000
    col1.metric("Distance to Safe Levels (Ammonia)", f"{safe_distance:.2f} km")
else:
    col1.metric("Distance to Safe Levels (Ammonia)", "> 50 km (Exceeds limit)")

peak_nitrate = np.max(B_mean)
col2.metric("Peak Nitrate Concentration", f"{peak_nitrate:.2f} mg/L")
