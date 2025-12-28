import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(0, 7, 200)  # Panjang reaktor (m)


CO2 = 0.30 * np.exp(-0.35 * z)          # CO2 menurun
H2  = 0.65 * np.exp(-0.50 * z)           # H2 menurun lebih cepat
CH3OH = 0.30 - CO2                       # Metanol meningkat

# Normalisasi agar tetap fraksi mol
total = CO2 + H2 + CH3OH
CO2 /= total
H2 /= total
CH3OH /= total

T_in = 220  # Suhu inlet (°C)
T_max = 275 # Batas aman katalis
Temperature = T_in + (T_max - T_in) * (1 - np.exp(-0.6 * z))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(z, CO2, label='CO$_2$')
plt.plot(z, H2, label='H$_2$')
plt.plot(z, CH3OH, label='CH$_3$OH')

plt.xlabel('Length of the Reactor (m)')
plt.ylabel('Mol Fraction')
plt.title('Composition Profile Along the Length of the Reactor')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(z, Temperature)
plt.axhline(280, linestyle='--', label='Catalyst Damage Limit (280°C)')

plt.xlabel('length of the Reactor (m)')
plt.ylabel('Temperature (°C)')
plt.title('Reaktor Temperature Profile (Isothermal Reaction)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()