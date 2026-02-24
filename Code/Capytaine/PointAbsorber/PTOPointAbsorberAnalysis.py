import os
import time

import matplotlib.pyplot as plt
import pandas as pd

import argparse


def generate_plots(csv_path="results/pto_results.csv", output_dir="results", c_pto=1.0e5, k_pto=5.0e5, freq_target=0.20):

    # Load data
    df = pd.read_csv(csv_path)

    # Filter by PTO settings
    mask_pto = (df["c_pto_Ns_m"] == c_pto) & (df["k_pto_N_m"] == k_pto)
    df_pto = df[mask_pto]

    if df_pto.empty:
        raise ValueError(
            f"No rows found for c_pto={c_pto}, k_pto={k_pto} in {csv_path}"
        )

    # Further filter by frequency (within rounding tolerance)
    df_mass = df_pto[df_pto["frequency_Hz"].round(3) == round(freq_target, 3)]

    if df_mass.empty:
        raise ValueError(
            f"No rows at frequency ≈ {freq_target} Hz for c_pto={c_pto}, k_pto={k_pto}"
        )

    df_mass = df_mass.sort_values("buoy_mass_kg")

    # Plot: energy per cycle vs mass
    fig, ax = plt.subplots()
    ax.plot(df_mass["buoy_mass_kg"], df_mass["E_cycle_J"], marker="o")
    ax.set_xlabel("Buoy mass [kg]")
    ax.set_ylabel("Energy per cycle [J]")
    ax.set_title(f"E_cycle vs mass (f={freq_target} Hz, c_pto={c_pto:.1e}, k_pto={k_pto:.1e})")
    ax.grid(True)
    fig.tight_layout()

    # Prepare timestamped output path
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"pto_analysis_mass_{timestamp}.png"
    out_path = os.path.join(output_dir, filename)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved plot to {out_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Analyze PTO results and generate plots."
    )

    # Path to the CSV produced by the simulation
    parser.add_argument("--csvpath",type=str,required=False, default="results/pto_results.csv")

    # Directory where plots will be written
    parser.add_argument("--outputdir",type=str,required=False, default="results"
)

    # PTO settings to filter on

    parser.add_argument("--cpto",type=float,required=False)

    parser.add_argument("--kpto",type=float,required=False)

    # Target wave frequency
    parser.add_argument("--freq",type=float,required=False)

    args = parser.parse_args()

    generate_plots(csv_path=args.csvpath, output_dir=args.outputdir, c_pto=args.cpto, k_pto=args.kpto, freq_target=args.freq)

    