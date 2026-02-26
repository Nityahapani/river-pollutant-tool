import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# --- 1. APP CONFIGURATION & UI ARCHITECTURE ---
st.set_page_config(page_title="Advanced River Pollutant Simulator", layout="wide", page_icon="🌊")

st.title("🌊 Advanced River Pollutant Simulator")
st.markdown("A probabilistic, multi-species transport model using continuous stirred-tank reactor (CSTR) kinetics.")

# Sidebar: Grouped Inputs for cleaner UI
with st.sidebar:
    st.header("Model Parameters")
    
    with st.expander("💧 Hydrological Conditions", expanded=True):
        velocity = st.slider("Flow Velocity (m/s)", 0.1, 2.0, 0.6, 0.1, help="Average stream velocity.")
        temp = st.slider("Water Temperature (°C)", 0.0, 35.0, 22.0, 1.0, help="Influences kinetic reaction rates.")

    with st.expander("🧪 Spill Characteristics", expanded=True):
        conc_A = st.number_input("Initial Ammonia (mg/L)", min_value=0.0, value=19.0)
        conc_B = st.number_input("Initial Nitrate (mg/L)", min_value=0.0, value=0.5)

    with st.expander("⚙️ Stochastic Engine", expanded=False):
        iterations = st.slider("Monte Carlo Iterations", 10, 500, 150, 10)
        reg_limit = st.number_input("Regulatory Limit (mg/L)", value=2.0)

# Constants
DIST_MAX = 50000  # 50 km
X_EVAL = np.linspace(0, DIST_MAX, 500)

# --- 2. MATHEMATICAL MODEL ---
def simulate_river(t, C, v, current_temp):
    # Arrhenius temperature correction
    k1 = (0.4 / 86400) * (1.047**(current_temp - 20)) 
    k2 = (0.15 / 86400) * (1.047**(current_temp - 20))
    
    dA_dx = (-k1 / v) * C[0]
    dB_dx = ((k1 / v) * C[0]) - ((k2 / v) * C[1])
    return [dA_dx, dB_dx]

# --- 3. MONTE CARLO ENGINE ---
with st.spinner('Calculating stochastic matrices...'):
    all_results_A = []
    all_results_B = []
    
    for _ in range(iterations):
        v_rand = velocity * np.random.uniform(0.9, 1.1)
        temp_rand = temp * np.random.uniform(0.9, 1.1)
        
        sol = solve_ivp(
            simulate_river, (0, DIST_MAX), [conc_A, conc_B], 
            t_eval=X_EVAL, args=(v_rand, temp_rand)
        )
        all_results_A.append(sol.y[0])
        all_results_B.append(sol.y[1])

A_mean, A_std = np.mean(all_results_A, axis=0), np.std(all_results_A, axis=0)
B_mean, B_std = np.mean(all_results_B, axis=0), np.std(all_results_B, axis=0)

# --- 4. DASHBOARD TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Interactive Visualizer", "🧮 Model Physics", "📥 Data Export"])

with tab1:
    # Actionable Insights at the top
    col1, col2, col3 = st.columns(3)
    safe_indices = np.where(A_mean < reg_limit)[0]
    safe_dist = X_EVAL[safe_indices[0]] / 1000 if len(safe_indices) > 0 else "> 50"
    
    col1.metric("Distance to Compliance", f"{safe_dist} km", delta="Ammonia Limit", delta_color="off")
    col2.metric("Peak Nitrate Load", f"{np.max(B_mean):.2f} mg/L", delta="Byproduct", delta_color="inverse")
    col3.metric("Max Reaction Temp", f"{temp}°C")

    # Interactive Plotly Chart
    fig = go.Figure()

    # Ammonia Trace & Confidence Bands
    fig.add_trace(go.Scatter(x=X_EVAL/1000, y=A_mean + 2*A_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=X_EVAL/1000, y=np.maximum(0, A_mean - 2*A_std), mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(214, 39, 40, 0.2)', name='95% CI (Ammonia)'))
    fig.add_trace(go.Scatter(x=X_EVAL/1000, y=A_mean, mode='lines', line=dict(color='#d62728', width=3), name='Ammonia (Mean)'))

    # Nitrate Trace & Confidence Bands
    fig.add_trace(go.Scatter(x=X_EVAL/1000, y=B_mean + 2*B_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=X_EVAL/1000, y=np.maximum(0, B_mean - 2*B_std), mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(31, 119, 180, 0.2)', name='95% CI (Nitrate)'))
    fig.add_trace(go.Scatter(x=X_EVAL/1000, y=B_mean, mode='lines', line=dict(color='#1f77b4', width=3), name='Nitrate (Mean)'))

    # Regulatory Limit Line
    fig.add_hline(y=reg_limit, line_dash="dash", line_color="black", annotation_text="EPA Limit", annotation_position="bottom right")

    fig.update_layout(
        title="Probabilistic Downstream Plume Analysis",
        xaxis_title="Distance Downstream (km)",
        yaxis_title="Concentration (mg/L)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Governing Differential Equations")
    st.markdown("The system relies on a coupled set of ordinary differential equations (ODEs) derived from the advection-reaction framework. Distance ($x$) is parameterized by flow velocity ($v$).")
    
    st.latex(r""" \frac{dA}{dx} = -\left(\frac{k_1}{v}\right)A """)
    st.latex(r""" \frac{dB}{dx} = \left(\frac{k_1}{v}\right)A - \left(\frac{k_2}{v}\right)B """)
    
    st.markdown("**Where:**")
    st.markdown("* $A$ = Concentration of Primary Pollutant (Ammonia)\n* $B$ = Concentration of Secondary Pollutant (Nitrate)\n* $k_1, k_2$ = Temperature-corrected decay constants")

with tab3:
    st.markdown("### Simulation Data")
    df = pd.DataFrame({
        "Distance_km": X_EVAL / 1000,
        "Ammonia_Mean_mgL": A_mean,
        "Ammonia_StdDev": A_std,
        "Nitrate_Mean_mgL": B_mean,
        "Nitrate_StdDev": B_std
    })
    st.dataframe(df, use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV for Excel/GIS",
        data=csv,
        file_name='river_simulation_results.csv',
        mime='text/csv',
    )
