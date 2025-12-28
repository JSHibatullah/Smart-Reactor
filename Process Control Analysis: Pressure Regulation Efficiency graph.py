import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
t_data = np.array([0, 10, 20, 30, 40, 50, 60])
pres_mpc = np.array([1.0, 25.0, 55.0, 75.0, 79.5, 80.0, 80.0])
pres_open = np.array([1.0, 35.0, 70.0, 88.0, 72.0, 83.0, 80.0])
setpoint = 80.0  
t_sim = np.arange(0, 61, 1)
f_open = interp1d(t_data, pres_open, kind='cubic')
f_mpc = interp1d(t_data, pres_mpc, kind='linear')

np.random.seed(42)
y_open = f_open(t_sim) + np.random.normal(0, 0.1, len(t_sim))
y_mpc = f_mpc(t_sim) + np.random.normal(0, 0.05, len(t_sim))
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

ax.axhline(y=setpoint, color='navy', linestyle='--', linewidth=1, alpha=0.7, label=f'Target Pressure: {setpoint} Bar')
ax.plot(t_sim, y_open, color='#d62728', linewidth=1.8, label='Baseline System (Open-Loop)')
ax.plot(t_sim, y_mpc, color='#2ca02c', linewidth=2.5, label='Optimized System (AI-MPC)')

ax.annotate('Pressure Overshoot', xy=(30, 88), xytext=(35, 95),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             fontsize=10, fontweight='bold', color='#d62728')

ax.annotate('System Oscillation', xy=(40, 72), xytext=(42, 65),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             fontsize=10, fontweight='bold', color='#d62728')

ax.annotate('Pressure Locked (T=50)', xy=(50, 80), xytext=(52, 75),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             fontsize=10, fontweight='bold', color='#2ca02c')

ax.set_title('Process Control Analysis: Pressure Regulation Efficiency', fontsize=16, pad=20)
ax.set_xlabel('Operation Time (Minutes)', fontsize=12)
ax.set_ylabel('Pressure (Bar)', fontsize=12)
ax.set_ylim(-5, 110)
ax.legend(loc='upper left', frameon=True, shadow=True)

plt.tight_layout()
plt.show()

print("-" * 45)
print("PRESSURE CONTROL PERFORMANCE REPORT")
print("-" * 45)
print(f"{'Metric':<25} | {'Baseline':<10} | {'AI-MPC':<10}")
print("-" * 45)
print(f"{'Max Overshoot (Bar)':<25} | {np.max(pres_open):<10.1f} | {np.max(pres_mpc):<10.1f}")
print(f"{'Settling Time (min)':<25} | {'~60':<10} | {'50':<10}")
print(f"{'Control Precision':<25} | {'Poor':<10} | {'Excellent':<10}")
print("-" * 45)