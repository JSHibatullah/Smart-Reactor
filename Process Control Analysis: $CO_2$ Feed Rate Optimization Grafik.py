import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

t_open = np.array([0, 5, 10, 20, 30, 45, 60])
flow_open = np.array([0.0000, 1.8000, 4.9500, 3.8500, 4.6500, 4.4200, 4.4010])

t_mpc = np.array([0, 5, 10, 20, 30, 40, 50, 60])
flow_mpc = np.array([0.0000, 1.1002, 2.4205, 3.9609, 4.4010, 4.4010, 4.4010, 4.4010])

setpoint = 4.40  # Target $CO_2$ Flow Rate (kg/s)
t_sim = np.arange(0, 61, 1)  # Simulation timeline (0-60 minutes)

f_open_interp = interp1d(t_open, flow_open, kind='cubic')

f_mpc_interp = interp1d(t_mpc, flow_mpc, kind='linear')

np.random.seed(42)
y_open = f_open_interp(t_sim) + np.random.normal(0, 0.015, len(t_sim))
y_mpc = f_mpc_interp(t_sim) + np.random.normal(0, 0.005, len(t_sim))

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

# Plotting Setpoint
ax.axhline(y=setpoint, color='navy', linestyle='--', linewidth=1, alpha=0.7, label=f'Setpoint Target: {setpoint} kg/s')

# Plotting Open-Loop Response
ax.plot(t_sim, y_open, color='#d62728', linewidth=1.8, label='Baseline System (Without MPC)')

# Plotting MPC Response
ax.plot(t_sim, y_mpc, color='#2ca02c', linewidth=2.5, label='Optimized System (AI-MPC Enabled)')

# Highlighting the Overshoot in Open-Loop
ax.annotate('Critical Overshoot', xy=(10, 4.95), xytext=(15, 5.4),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             fontsize=10, fontweight='bold', color='#d62728')

# Highlighting Steady-State Achievement
ax.annotate('Steady-State Achieved (T=30)', xy=(30, 4.401), xytext=(35, 3.8),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             fontsize=10, fontweight='bold', color='#2ca02c')

# --- PLOT REFINEMENT ---
ax.set_title('Process Control Analysis: $CO_2$ Feed Rate Optimization', fontsize=16, pad=20)
ax.set_xlabel('Operation Time (Minutes)', fontsize=12)
ax.set_ylabel('$CO_2$ Flow Rate (kg/s)', fontsize=12)
ax.set_ylim(-0.2, 6.0)
ax.legend(loc='upper right', frameon=True, shadow=True)

plt.tight_layout()
plt.show()

print("-" * 40)
print("CONTROL SYSTEM PERFORMANCE REPORT")
print("-" * 40)
print(f"{'Metric':<25} | {'Baseline':<10} | {'AI-MPC':<10}")
print("-" * 40)
print(f"{'Max Overshoot (kg/s)':<25} | {np.max(flow_open):<10.4f} | {np.max(flow_mpc):<10.4f}")
print(f"{'Settling Time (min)':<25} | {'~60':<10} | {'30':<10}")
print(f"{'Steady-State Error':<25} | {'Low':<10} | {'Near-Zero':<10}")
print("-" * 40)