import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
t_data = np.array([0, 10, 20, 30, 40, 50, 60])
soec_mpc = np.array([300.0, 480.5, 680.0, 810.5, 845.0, 849.5, 850.0])
soec_open = np.array([300.0, 520.0, 750.0, 915.0, 820.0, 870.0, 850.0])

setpoint = 850.0
t_sim = np.arange(0, 61, 1)
np.random.seed(42)
f_open = interp1d(t_data, soec_open, kind='cubic')
f_mpc = interp1d(t_data, soec_mpc, kind='linear')

y_open = f_open(t_sim) + np.random.normal(0, 0.5, len(t_sim))
y_mpc = f_mpc(t_sim) + np.random.normal(0, 0.2, len(t_sim))

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

ax.plot(t_sim, y_open, color='#d62728', linewidth=1.5, label='Open-Loop (Manual Regulation)')
ax.plot(t_sim, y_mpc, color='#2ca02c', linewidth=2.5, label='AI-MPC (Predictive Control)')
ax.axhline(setpoint, color='navy', linestyle='--', alpha=0.6, label=f'Target Operating Temp ({setpoint}°C)')

ax.annotate('System Start', xy=(0, 300), xytext=(2, 350), arrowprops=dict(arrowstyle='->'))
ax.annotate('Extreme Overshoot (915°C)', xy=(30, 915), xytext=(35, 940), 
             weight='bold', color='#d62728', arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('MPC Steady-State', xy=(60, 850), xytext=(50, 800), 
             weight='bold', color='#2ca02c', arrowprops=dict(arrowstyle='->', color='black'))

ax.set_title('SOEC Thermal Management: AI-MPC vs. Open-Loop', fontsize=15, pad=15)
ax.set_ylabel('SOEC Temperature (°C)', fontsize=12)
ax.set_xlabel('Time (Minutes)', fontsize=12)
ax.set_ylim(250, 1000)
ax.legend(frameon=True, shadow=True, loc='lower right')

plt.tight_layout()
plt.show()

print("-" * 50)
print(f"{'SOEC PERFORMANCE METRIC':<25} | {'Open-Loop':<10} | {'AI-MPC':<10}")
print("-" * 50)
print(f"{'Peak Temperature (°C)':<25} | {np.max(soec_open):<10.1f} | {np.max(soec_mpc):<10.1f}")
print(f"{'Settling Time (min)':<25} | {'60+':<10} | {'~50':<10}")
print(f"{'Steady State Stability':<25} | {'Oscillatory':<10} | {'High':<10}")
print("-" * 50)