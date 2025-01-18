import matplotlib.pyplot as plt

def plot_rmse_r2(df, filename="rmse_r2_plot.png"):
    """
    Plots a dual-axis chart for RMSE and R2 from the given DataFrame and saves it as a PNG file.

    Parameters:
        df (pd.DataFrame): DataFrame containing columns ['data', 'RMSE', 'R2'].
        filename (str): Name of the output PNG file.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # X-axis values
    x = df['data']

    # Plot RMSE on the left y-axis
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("RMSE")
    ax1.plot(x, df['RMSE'], marker="o", label="RMSE", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Plot R2 on the right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("R²")
    ax2.plot(x, df['R2'], marker="o", linestyle="--", label="R²", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Title and legends
    fig.suptitle("Comparison of RMSE and R² Across Datasets", fontsize=14)
    fig.tight_layout()

    # Stack legends vertically
    ax1.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), labelcolor="blue")
    ax2.legend(loc="upper left", bbox_to_anchor=(0.1, 0.85), labelcolor="red")

    plt.tight_layout()

    # Save the plot
    if filename:
        plt.savefig(filename)

    
    plt.close()


