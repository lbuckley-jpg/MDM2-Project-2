import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# ── Physical constants ─────────────────────────────────────────────────────
rho     = 1025.0
g       = 9.81
R       = 2.0
A       = np.pi * R**2
b       = 0.3          # damping parameter (no PTO)
f_target = 0.12
omega_t  = 2 * np.pi * f_target

# ── Method: hydrostatic spring ─────────────────────────────────────────────
# Restoring spring constant from buoyancy alone
k_spring = rho * g * A

# Damping ratio (from parametric form: c = b*M*omega, xi = c/(2*sqrt(k*M)))
# At resonance xi = b/2 (to leading order)
xi = b / 2.0

# Undamped resonant mass
M_undamped = k_spring / omega_t**2

# Damped resonant mass (peak of displacement response)
M_damped = k_spring / (omega_t**2 * (1 - xi**2))

print("── Hydrostatic spring method ──────────────────────────────")
print(f"  Spring constant k    : {k_spring:.2f} N/m")
print(f"  Damping ratio  xi    : {xi:.4f}")
print(f"  M_res (undamped)     : {M_undamped:.2f} kg")
print(f"  M_res (damped)       : {M_damped:.2f} kg")

# ── Method: free-decay simulation + FFT ───────────────────────────────────
# Equation of motion (no forcing, no PTO):
#   M * z'' + b*M*omega_n * z' + rho*g*A * z = 0
# Normalised: z'' + b*omega_n*z' + omega_n^2 * z = 0

def free_decay(t, y, omega_n):
    z, zdot = y
    zddot = -b * omega_n * zdot - omega_n**2 * z
    return [zdot, zddot]

omega_n = np.sqrt(k_spring / M_damped)

t_span = (0, 300)
t_eval = np.linspace(*t_span, 30000)
z0     = [0.5, 0.0]          # initial displacement, zero velocity

sol = solve_ivp(free_decay, t_span, z0,
                t_eval=t_eval, args=(omega_n,),
                method='RK45', rtol=1e-8)

# FFT of free-decay signal to find dominant frequency
dt   = t_eval[1] - t_eval[0]
z    = sol.y[0]
N    = len(z)
freq = np.fft.rfftfreq(N, d=dt)
amp  = np.abs(np.fft.rfft(z))

# Find peak frequency from FFT
peak_idx   = np.argmax(amp)
f_fft      = freq[peak_idx]
omega_fft  = 2 * np.pi * f_fft
M_fft      = k_spring / omega_fft**2   # back-calculate mass

print("\n── Free-decay FFT method ──────────────────────────────────")
print(f"  Dominant frequency   : {f_fft:.5f} Hz")
print(f"  Target frequency     : {f_target:.5f} Hz")
print(f"  M back-calculated    : {M_fft:.2f} kg")

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: free decay time series
axes[0].plot(sol.t, z, color="steelblue", lw=1.2)
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Displacement [m]")
axes[0].set_title("Free-decay time series")
axes[0].grid(True, alpha=0.4)

# Right: FFT spectrum
axes[1].plot(freq, amp, color="steelblue", lw=1.5)
axes[1].axvline(f_fft, ls="--", color="crimson", lw=1.8,
                label=f"FFT peak = {f_fft:.5f} Hz")
axes[1].axvline(f_target, ls=":", color="forestgreen", lw=1.8,
                label=f"Target    = {f_target:.5f} Hz")
axes[1].set_xlim(0, 3 * f_target)
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_ylabel("Amplitude")
axes[1].set_title("FFT of free-decay — natural frequency")
axes[1].legend()
axes[1].grid(True, alpha=0.4)

plt.suptitle(
    f"Hydrostatic spring method  |  k = {k_spring:.1f} N/m  |"
    f"  M_res = {M_damped:.1f} kg  |  xi = {xi:.3f}",
    fontsize=11
)
plt.tight_layout()
plt.show()