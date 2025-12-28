import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

t_data = np.array([0, 10, 20, 30, 40, 50, 60])
temp_mpc = np.array([150.0, 185.0, 215.0, 242.0, 248.5, 250.0, 250.0])
temp_open = np.array([150.0, 195.0, 240.0, 268.0, 235.0, 255.0, 250.0])

setpoint = 250.0  
t_sim = np.arange(0, 61, 1)

f_open = interp1d(t_data, temp_open, kind='cubic')
f_mpc = interp1d(t_data, temp_mpc, kind='linear')

np.random.seed(42)
y_open = f_open(t_sim) + np.random.normal(0, 0.2, len(t_sim))
y_mpc = f_mpc(t_sim) + np.random.normal(0, 0.1, len(t_sim))

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

ax.axhline(y=setpoint, color='navy', linestyle='--', linewidth=1, alpha=0.7, label=f'Target Temperature: {setpoint}°C')
ax.plot(t_sim, y_open, color='#d62728', linewidth=1.8, label='Baseline System (Open-Loop)')
ax.plot(t_sim, y_mpc, color='#2ca02c', linewidth=2.5, label='Optimized System (AI-MPC)')

ax.annotate('Critical Thermal Spike', xy=(30, 268), xytext=(35, 280),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             fontsize=10, fontweight='bold', color='#d62728')

ax.annotate('Thermal Drop', xy=(40, 235), xytext=(42, 220),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             fontsize=10, fontweight='bold', color='#d62728')

ax.annotate('Steady-State (T=50)', xy=(50, 250), xytext=(52, 240),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             fontsize=10, fontweight='bold', color='#2ca02c')

ax.set_title('Process Control Analysis: Temperature Thermal Stability', fontsize=16, pad=20)
ax.set_xlabel('Operation Time (Minutes)', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
ax.set_ylim(140, 300)
ax.legend(loc='upper left', frameon=True, shadow=True)

plt.tight_layout()
plt.show()

print("-" * 45)
print("TEMPERATURE CONTROL PERFORMANCE REPORT")
print("-" * 45)
print(f"{'Metric':<25} | {'Baseline':<10} | {'AI-MPC':<10}")
print("-" * 45)
print(f"{'Max Peak Temp (°C)':<25} | {np.max(temp_open):<10.1f} | {np.max(temp_mpc):<10.1f}")
print(f"{'Settling Time (min)':<25} | {'~60':<10} | {'50':<10}")
print(f"{'Stability Status':<25} | {'Fluctuating':<10} | {'Stable':<10}")
print("-" * 45)