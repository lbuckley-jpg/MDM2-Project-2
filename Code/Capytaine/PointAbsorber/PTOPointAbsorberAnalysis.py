import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path="pto_results.csv"):
    df = pd.read_csv(csv_path)

    # Plot 1: Power vs mass for a fixed PTO setting

    mask = (df["c_pto_Ns_m"] == 1.0e5) & (df["k_pto_N_m"] == 5.0e5)
    df_fixed_pto = df[mask].sort_values("frequency_Hz")

    plt.figure()
    plt.plot(df_fixed_pto["frequency_Hz"], df_fixed_pto["P_absorbed_W"], marker="o")
    plt.xlabel("Wave frequency [Hz]")
    plt.ylabel("Absorbed power [W]")
    plt.title("Heave PTO absorbed power vs frequency")
    plt.grid(True)



if __name__ == "__main__":
    main()

