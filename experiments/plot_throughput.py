import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_path = "results/throughput_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run experiments first.")
        return
        
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    
    # Custom colors and styles
    # We want a premium look
    colors = {
        "Threefry": "#1f77b4",
        "Tyche V1": "#7f7f7f",
        "Tyche V2 (T=16)": "#e377c2",
        "Tyche V2 (T=32)": "#bcbd22",
        "Tyche V2 (T=64)": "#17becf"
    }
    
    # Plot Threefry
    tf_df = df[df["generator"] == "Threefry"]
    if not tf_df.empty:
        plt.plot(tf_df["batch_size"], tf_df["throughput_GBs"], marker='o', linewidth=2, label="Threefry", color=colors["Threefry"])
        
    # Plot Tyche V1
    v1_df = df[df["generator"] == "Tyche V1"]
    if not v1_df.empty:
        plt.plot(v1_df["batch_size"], v1_df["throughput_GBs"], marker='s', linestyle='--', linewidth=1.5, label="Tyche V1 (T=4, R=4)", color=colors["Tyche V1"])
        
    # Plot selected Tyche V2 configs (e.g. R=2 and R=4 for each T)
    for tile_size in [16, 32, 64]:
        for num_rounds in [2, 4]:
            v2_subset = df[(df["generator"] == "Tyche V2") & (df["tile_size"].astype(str) == str(tile_size)) & (df["num_rounds"].astype(str) == str(num_rounds))]
            if not v2_subset.empty:
                v2_subset = v2_subset.sort_values(by="batch_size")
                label = f"Tyche V2 (T={tile_size}, R={num_rounds})"
                linestyle = ":"
                marker = 'x'
                plt.plot(v2_subset["batch_size"], v2_subset["throughput_GBs"], marker=marker, linestyle=linestyle, linewidth=1.5, label=label)
                
    # Plot selected Tyche V2.1 configs (e.g. R=2 and R=4 for each T)
    for tile_size in [16, 32, 64]:
        for num_rounds in [2, 4]:
            v2_1_subset = df[(df["generator"] == "Tyche V2.1") & (df["tile_size"].astype(str) == str(tile_size)) & (df["num_rounds"].astype(str) == str(num_rounds))]
            if not v2_1_subset.empty:
                v2_1_subset = v2_1_subset.sort_values(by="batch_size")
                label = f"Tyche V2.1 (T={tile_size}, R={num_rounds})"
                linestyle = "-"
                marker = '^' if num_rounds == 2 else 'v'
                plt.plot(v2_1_subset["batch_size"], v2_1_subset["throughput_GBs"], marker=marker, linestyle=linestyle, linewidth=2, label=label)

    # Plot Philox and Threefry
    for name_prefix in ["Philox-32", "Philox-64", "Threefry-32", "Threefry-64"]:
        for num_rounds in [2, 4]:
            subset = df[(df["generator"] == name_prefix) & (df["num_rounds"].astype(str) == str(num_rounds))]
            if not subset.empty:
                subset = subset.sort_values(by="batch_size")
                label = f"{name_prefix} (R={num_rounds})"
                linestyle = "-." if "Philox" in name_prefix else "-"
                marker = 's' if "32" in name_prefix else 'd'
                plt.plot(subset["batch_size"], subset["throughput_GBs"], marker=marker, linestyle=linestyle, linewidth=2, label=label)
                
    plt.xscale("log")
    plt.xlabel("Batch Size (number of uint32 elements)", fontsize=11)
    plt.ylabel("Throughput (GB/s)", fontsize=11)
    plt.title("RNG Throughput Scaling comparison on H100", fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    plot_path = "results/throughput_scaling.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved successfully to {plot_path}")

if __name__ == "__main__":
    main()
