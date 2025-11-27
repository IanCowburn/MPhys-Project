import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from matplotlib import gridspec

class TransformerPlotting:
    def __init__(self, difference, embedder_output, y_true, y_pred, rmse):
        self.difference = difference
        self.embedder_output = embedder_output
        self.y_true = y_true
        self.y_pred = y_pred
        self.rmse = rmse

    def plot_all(self):
        # Transformer Regression Error Distribution Plot
        plt.figure(figsize=(10, 6))
        plt.hist(self.difference, bins=100, range=(-1e6, 1e6), alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error [MeV]')
        plt.ylabel('Frequency')
        plt.title(f'Prediction Error Distribution (RMSE: {self.rmse:.2f} MeV)')
        plt.grid(True, alpha=0.3)
        plt.savefig('transformer_regression_error_distribution.png', dpi=300)
        plt.show()
        plt.close()
        # Transformer Embedder Output Distribution Plot
        # plt.figure(figsize=(10, 6))
        # embedder_flat = self.embedder_output.flatten()
        # plt.hist(embedder_flat, bins=100, alpha=0.7, edgecolor='black')
        # plt.xlabel('Embedder Output Value')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of Embedder Outputs (Before Final Linear Layer)')
        # plt.grid(True, alpha=0.3)
        # plt.savefig('embedder_output_distribution.png', dpi=300)
        # plt.show()
        # plt.close()
        # print(f"Embedder output stats: mean={embedder_flat.mean():.4f}, std={embedder_flat.std():.4f}, "
        #     f"min={embedder_flat.min():.4f}, max={embedder_flat.max():.4f}")

        # Nicer distribution + ratio layout
        

        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.05)

        ax_main = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

        # Main histograms (use step hist for cleaner look)
        bins = 100
        x_min, x_max = 0, 6e6
        ax_main.hist(self.y_true, bins=bins, range=(x_min, x_max),
                     histtype='step', linewidth=1.5, color='blue', label='True')
        ax_main.hist(self.y_pred, bins=bins, range=(x_min, x_max),
                     histtype='step', linewidth=1.5, color='red', label='Predicted')
        ax_main.set_ylabel('Frequency')
        ax_main.set_title('Invariant Mass Distribution')
        ax_main.grid(True, alpha=0.2)
        ax_main.legend(loc='upper right')

        # Ratio (Pred/True)
        counts_true, bin_edges = np.histogram(self.y_true, bins=bins, range=(x_min, x_max))
        counts_pred, _ = np.histogram(self.y_pred, bins=bins, range=(x_min, x_max))
        ratio = np.divide(counts_pred, counts_true, where=counts_true != 0)
        ratio[np.isnan(ratio)] = 0.0

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax_ratio.plot(bin_centers, ratio, color='black', linewidth=1.2)
        ax_ratio.axhline(1.0, color='red', linestyle='--', linewidth=1)
        ax_ratio.set_ylabel('Pred/True')
        ax_ratio.set_xlabel('Invariant Mass [MeV]')
        ax_ratio.set_ylim(0, 2)
        ax_ratio.grid(True, alpha=0.2, axis='y')

        # Hide x tick labels on main plot (shared x)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        plt.tight_layout()
        plt.savefig('true_invariant_mass_distribution.png', dpi=300)
        plt.show()
        plt.close()

        # Transformer Regression 2D Histogram Plot
        plt.figure(figsize=(8,6))
        plt.hist2d(self.y_true, self.y_pred, bins=250, range=[[0, 6e6], [0, 6e6]], cmap='viridis', cmin=1)
        plt.plot([0, 6e6], [0, 6e6], color='red', linestyle='--', linewidth=1)
        plt.xlabel('Expected Invariant Mass [MeV]')
        plt.ylabel('Predicted Invariant Mass [MeV]')
        plt.title('2D Histogram: Expected vs Predicted Invariant Mass')
        plt.colorbar(label='Counts')
        plt.savefig('transformer_regression_2d_histogram.png', dpi=300)
        plt.show()
        plt.close()



        # Transformer Regression Low-binned Plot (Percent per bin)
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = 10
        x_range = (0, 6e6)
        y_range = (0, 6e6)
        # 2D histogram -> percentages
        H, xedges, yedges = np.histogram2d(self.y_true, self.y_pred, bins=bins, range=[x_range, y_range])
        total = H.sum()
        H_percent = (100.0 * H / total) if total > 0 else H
        # Plot normalized to percent
        pcm = ax.pcolormesh(
            xedges, yedges, H_percent.T, cmap='viridis',
            norm=mcolors.Normalize(vmin=0.0, vmax=H_percent.max())
        )
        # Annotate each bin with its percentage
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        for i, xc in enumerate(xcenters):
            for j, yc in enumerate(ycenters):
                val = H_percent[i, j]
                # choose text color for readability
                txt_color = 'white' if val >= 40 else 'black'
                ax.text(xc, yc, f"{val:.1f}%", ha='center', va='center',
                        fontsize=12, fontweight='bold', color=txt_color)
        ax.set_xlim(x_range); ax.set_ylim(y_range)
        ax.set_xlabel('Expected Invariant Mass [MeV]')
        ax.set_ylabel('Predicted Invariant Mass [MeV]')
        ax.set_title('Binned plot: Expected vs Predicted Invariant Mass (Percent per bin)')
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label('Percentage of events (%)')
        cbar.set_ticks([0, H_percent.max()/4, H_percent.max()/2, 3*H_percent.max()/4, H_percent.max()])
        plt.tight_layout()
        plt.savefig('transformer_regression_binned_plot_percent.png', dpi=300)
        plt.show()
        plt.close()