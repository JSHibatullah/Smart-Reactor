import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
t_data = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
flow_mpc = np.array([0, 0.185, 0.38, 0.52, 0.595, 0.6048, 0.6046, 
                     0.6048, 0.6048, 0.6047, 0.6048, 0.6048, 0.6048])
flow_open = np.array([0, 0.35, 0.6048, 0.685, 0.6, 0.6048, 0.485, 
                      0.72, 0.604, 0.52, 0.65, 0.62, 0.6048])

setpoint = 0.6048  # Target Steady State
t_sim = np.arange(0, 61, 1)  # Simulasi per menit
f_open_interp = interp1d(t_data, flow_open, kind='linear')
f_mpc_interp = interp1d(t_data, flow_mpc, kind='cubic')
np.random.seed(42)
y_open = f_open_interp(t_sim) + np.random.normal(0, 0.005, len(t_sim))
y_mpc = f_mpc_interp(t_sim) + np.random.normal(0, 0.001, len(t_sim))
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
ax.axhline(y=setpoint, color='navy', linestyle='--', linewidth=1, alpha=0.7, label=f'Setpoint Target: {setpoint}')
ax.plot(t_sim, y_open, color='#d62728', linewidth=1.5, label='Open Loop (Without MPC) - Fluctuating')
ax.plot(t_sim, y_mpc, color='#2ca02c', linewidth=2.5, label='MPC System - Stable & Precise')
ax.annotate('Open Loop Overshoot', xy=(15, 0.685), xytext=(18, 0.75),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=9, fontweight='bold', color='#d62728')
ax.annotate('NCG Spike 1: Temperature Drop', xy=(30, 0.485), xytext=(22, 0.42),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=9, style='italic')
ax.annotate('NCG Spike 2: Fluctuating Pressure', xy=(45, 0.52), xytext=(40, 0.45),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=9, style='italic')
ax.annotate('Stable MPC', xy=(25, 0.6048), xytext=(30, 0.65),
             arrowprops=dict(arrowstyle='->', color='darkgreen'),
             fontsize=9, fontweight='bold', color='#2ca02c')
ax.set_title('Comparative Analysis of SOEC Output $H_2$ Rate Control: MPC vs Open Loop', fontsize=16, pad=20)
ax.set_xlabel('Time (Minutes)', fontsize=12)
ax.set_ylabel('$H_2$ Flow Rate (kg/s)', fontsize=12)
ax.set_ylim(-0.05, 0.85)
ax.legend(loc='lower right', frameon=True, shadow=True)

plt.tight_layout()
plt.show()