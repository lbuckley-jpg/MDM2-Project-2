import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_COLS = [
    "omega_rad_s", "frequency_Hz", "wave_amplitude_m",
    "c_pto_Ns_m", "k_pto_N_m", "buoy_mass_kg", "buoy_radius_m",
    "water_depth_m", "water_density_kg_m3", "P_absorbed_W", "E_cycle_J",
]


def load(csv_path="results/pto_results.csv"):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["P_absorbed_W", "E_cycle_J"])
    return df


def save_fig(fig, name, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(output_dir, f"{name}_{ts}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ── 1. Power vs frequency, one curve per mass ──────────────────────────────

def plot_power_vs_freq_by_mass(df, output_dir="results"):
    best = df.loc[df.groupby(["buoy_mass_kg", "frequency_Hz"])["P_absorbed_W"].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, grp in best.sort_values("frequency_Hz").groupby("buoy_mass_kg"):
        ax.plot(grp["frequency_Hz"], grp["P_absorbed_W"], marker=".", label=f"{m:.0f} kg")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Best absorbed power [W]")
    ax.set_title("Optimal P_absorbed vs frequency (each mass)")
    ax.legend(fontsize=7, ncol=2, title="Mass")
    ax.grid(True)
    save_fig(fig, "power_vs_freq_by_mass", output_dir)


# ── 2. Power vs frequency, one curve per c_pto ─────────────────────────────

def plot_power_vs_freq_by_cpto(df, output_dir="results"):
    top_cptos = df.groupby("c_pto_Ns_m")["P_absorbed_W"].mean().nlargest(8).index
    sub = df[df["c_pto_Ns_m"].isin(top_cptos)]
    fig, ax = plt.subplots(figsize=(10, 6))
    for c, grp in sub.sort_values("frequency_Hz").groupby("c_pto_Ns_m"):
        avg = grp.groupby("frequency_Hz")["P_absorbed_W"].mean()
        ax.plot(avg.index, avg.values, marker=".", label=f"{c:.2e}")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Mean absorbed power [W]")
    ax.set_title("P_absorbed vs frequency (selected c_pto values, averaged over mass)")
    ax.legend(fontsize=7, ncol=2, title="c_pto [Ns/m]")
    ax.grid(True)
    save_fig(fig, "power_vs_freq_by_cpto", output_dir)


# ── 3. Power vs c_pto, one curve per frequency ─────────────────────────────

def plot_power_vs_cpto_by_freq(df, output_dir="results"):
    fig, ax = plt.subplots(figsize=(10, 6))
    for f, grp in df.sort_values("c_pto_Ns_m").groupby("frequency_Hz"):
        avg = grp.groupby("c_pto_Ns_m")["P_absorbed_W"].mean()
        ax.plot(avg.index, avg.values, marker=".", label=f"{f:.3f} Hz")
    ax.set_xlabel("c_pto [Ns/m]")
    ax.set_ylabel("Mean absorbed power [W]")
    ax.set_title("P_absorbed vs c_pto (each frequency, averaged over mass)")
    ax.legend(fontsize=7, ncol=2, title="Frequency")
    ax.grid(True)
    save_fig(fig, "power_vs_cpto_by_freq", output_dir)


# ── 4. E_cycle vs mass, one curve per frequency ────────────────────────────

def plot_ecycle_vs_mass_by_freq(df, output_dir="results"):
    best = df.loc[df.groupby(["buoy_mass_kg", "frequency_Hz"])["E_cycle_J"].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 6))
    for f, grp in best.sort_values("buoy_mass_kg").groupby("frequency_Hz"):
        ax.plot(grp["buoy_mass_kg"], grp["E_cycle_J"], marker=".", label=f"{f:.3f} Hz")
    ax.set_xlabel("Buoy mass [kg]")
    ax.set_ylabel("Best E_cycle [J]")
    ax.set_title("Optimal E_cycle vs mass (each frequency)")
    ax.legend(fontsize=7, ncol=2, title="Frequency")
    ax.grid(True)
    save_fig(fig, "ecycle_vs_mass_by_freq", output_dir)


# ── 5. Optimal c_pto vs frequency, one curve per mass ──────────────────────

def plot_optimal_cpto_vs_freq(df, output_dir="results"):
    best = df.loc[df.groupby(["buoy_mass_kg", "frequency_Hz"])["P_absorbed_W"].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, grp in best.sort_values("frequency_Hz").groupby("buoy_mass_kg"):
        ax.plot(grp["frequency_Hz"], grp["c_pto_Ns_m"], marker=".", label=f"{m:.0f} kg")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Optimal c_pto [Ns/m]")
    ax.set_title("Optimal PTO damping vs frequency (each mass)")
    ax.legend(fontsize=7, ncol=2, title="Mass")
    ax.grid(True)
    save_fig(fig, "optimal_cpto_vs_freq", output_dir)


# ── 6. Heatmap: best power over (mass, frequency) ──────────────────────────

def plot_power_heatmap(df, output_dir="results"):
    best = df.loc[df.groupby(["buoy_mass_kg", "frequency_Hz"])["P_absorbed_W"].idxmax()]
    pivot = best.pivot_table(index="frequency_Hz", columns="buoy_mass_kg", values="P_absorbed_W")
    M, F = np.meshgrid(pivot.columns.values, pivot.index.values)
    fig, ax = plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(M, F, pivot.values, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="Best P_absorbed [W]")
    ax.set_xlabel("Buoy mass [kg]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Best absorbed power over (mass, frequency)")
    save_fig(fig, "power_heatmap_mass_freq", output_dir)


# ── 7. Heatmap: best power over (c_pto, frequency) ─────────────────────────

def plot_power_heatmap_cpto_freq(df, output_dir="results"):
    avg = df.groupby(["c_pto_Ns_m", "frequency_Hz"])["P_absorbed_W"].mean().reset_index()
    pivot = avg.pivot_table(index="frequency_Hz", columns="c_pto_Ns_m", values="P_absorbed_W")
    if pivot.empty:
        return
    C, F = np.meshgrid(pivot.columns.values, pivot.index.values)
    fig, ax = plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(C, F, pivot.values, shading="auto", cmap="inferno")
    fig.colorbar(pcm, ax=ax, label="Mean P_absorbed [W]")
    ax.set_xlabel("c_pto [Ns/m]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Mean absorbed power over (c_pto, frequency)")
    save_fig(fig, "power_heatmap_cpto_freq", output_dir)


# ── 8. Global optimum summary ──────────────────────────────────────────────

def print_summary(df):
    idx = df["P_absorbed_W"].idxmax()
    row = df.loc[idx]
    print("\n══ Global best combination ══")
    print(f"  Mass     = {row['buoy_mass_kg']:.1f} kg")
    print(f"  Freq     = {row['frequency_Hz']:.4f} Hz  (ω = {row['omega_rad_s']:.4f} rad/s)")
    print(f"  c_pto    = {row['c_pto_Ns_m']:.2e} Ns/m")
    print(f"  k_pto    = {row['k_pto_N_m']:.2e} N/m")
    print(f"  P_abs    = {row['P_absorbed_W']:.2f} W")
    print(f"  E_cycle  = {row['E_cycle_J']:.2f} J")
    print()


# ── Run all ─────────────────────────────────────────────────────────────────

def analyse(csv_path="results/pto_results.csv", output_dir="results"):
    df = load(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}\n")
    print_summary(df)

    plot_power_vs_freq_by_mass(df, output_dir)
    plot_power_vs_freq_by_cpto(df, output_dir)
    plot_power_vs_cpto_by_freq(df, output_dir)
    plot_ecycle_vs_mass_by_freq(df, output_dir)
    plot_optimal_cpto_vs_freq(df, output_dir)
    plot_power_heatmap(df, output_dir)
    plot_power_heatmap_cpto_freq(df, output_dir)

    print("\nDone – all plots saved.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyse PTO batch results.")
    parser.add_argument("--csvpath",   type=str, default="results/pto_results.csv")
    parser.add_argument("--outputdir", type=str, default="results")
    args = parser.parse_args()
    analyse(csv_path=args.csvpath, output_dir=args.outputdir)
