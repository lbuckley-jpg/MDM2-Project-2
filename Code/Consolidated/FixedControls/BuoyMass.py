import numpy as np
import matplotlib.pyplot as plt

# Physical constants
rho = 1025.0
g   = 9.81
a   = 0.5

# Damping parameter (radiation + drag only, no PTO)
b = 0.3    # THIS NEEDS REFERENCE!!

# Target wave frequency
f_range = np.linspace(0.1, 0.5)
f_target = 0.25 #REF NEEDED HERE
omega_range = 2 * np.pi * f_range
omega_t  = 2 * np.pi * f_target

# Buoy geometry
R = 2.0     # free var
A = np.pi * R**2

# Resonant m from d/dm[(1-m)^2 + b^2*m^2] = 0  →  m = 1/(1+b^2)
m_peak       = 1.0 / (1.0 + b**2)

# find resonant masses
M_range = (rho * g * A * m_peak) / omega_range**2
M_target = (rho * g * A * m_peak) / omega_t**2

print(f'Optimal Mass for wave freq {f_target} hz : {M_target:.2f} kg')

# Plot optimal mass against wave freq
fig, ax = plt.subplots()
ax.plot(f_range, M_range, color='blue')

ax.set_xlim(f_range.min(), f_range.max())
ax.set_ylim(M_range.min(), M_range.max())
x_left = ax.get_xlim()[0]
y_bottom = ax.get_ylim()[0]

ax.vlines(f_target, ymin=y_bottom, ymax=M_target, colors='crimson', linestyles='--')
ax.hlines(M_target, xmin=x_left, xmax=f_target, colors='crimson', linestyles='--')

ax.scatter([f_target], [M_target], color='crimson', s=60, 
           label=
                f'f = {f_target} hz\n'
                f'M = {M_target:.1f} kg')

ax.set_xlabel('Resonant Buoy Mass M [kg]')
ax.set_ylabel('Wave Frequency [hz]')
ax.set_title(
    f'Resonant mass vs Wave frequency with no PTO\n'
    f'Radius = {R} m,  Damping parameter b = {b},'
)
ax.legend()
#ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()