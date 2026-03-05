import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from matplotlib import gridspec
from scipy.stats import pearsonr

class TransformerPlotting:
    def __init__(self, y_true, y_pred, invariant_mass_corr, invariant_mass_rmse, invariant_mass_difference, ht_corr, ht_rmse, ht_difference, invariant_mass_biased, ht_biased, embedder_output, train_loss_history, val_loss_history):
        self.invariant_mass_difference = invariant_mass_difference
        self.ht_difference = ht_difference
        self.invariant_mass_biased = invariant_mass_biased
        self.ht_biased = ht_biased
        self.embedder_output = embedder_output
        self.y_true = y_true
        self.y_pred = y_pred
        self.invariant_mass_rmse = invariant_mass_rmse
        self.ht_rmse = ht_rmse
        self.invariant_mass_corr = invariant_mass_corr
        self.ht_corr = ht_corr
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history

    def plot_loss_curves(self, outpath='Transformer_training_validation_loss.png'):
        if self.train_loss_history is None or self.val_loss_history is None:
            return
        if len(self.train_loss_history) == 0 or len(self.val_loss_history) == 0:
            return
        with plt.rc_context({'font.size': 16}):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.train_loss_history, label='Train Loss', color='blue', linewidth=2)
            ax.plot(self.val_loss_history, label='Validation Loss', color='orange', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(outpath, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)

    
    def plot_all(self):
        # In GeV:
        self.invariant_mass_difference /= 1e3
        self.ht_difference /= 1e3
        self.y_true /= 1e3  
        self.y_pred /= 1e3
        self.invariant_mass_rmse /= 1e3
        self.ht_rmse /= 1e3

        # Separate mass and HT from y_true and y_pred
        y_true_mass_gev = self.y_true[:, 0]
        y_true_ht_gev = self.y_true[:, 1]
        y_pred_mass_gev = self.y_pred[:, 0]
        y_pred_ht_gev = self.y_pred[:, 1]
        error_mass_gev = self.invariant_mass_difference
        error_ht_gev = self.ht_difference

        # Compute additional statistics
        mean_error_mass = np.mean(error_mass_gev)
        std_error_mass = np.std(error_mass_gev)
        mean_error_ht = np.mean(error_ht_gev)
        std_error_ht = np.std(error_ht_gev)

        err_counts_mass, err_bins_mass = np.histogram(error_mass_gev, bins=200, range=(-1e3, 1e3))
        mode_error_mass = 0.5 * (err_bins_mass[:-1] + err_bins_mass[1:])[np.argmax(err_counts_mass)]
        err_counts_ht, err_bins_ht = np.histogram(error_ht_gev, bins=200, range=(-1e3, 1e3))
        mode_error_ht = 0.5 * (err_bins_ht[:-1] + err_bins_ht[1:])[np.argmax(err_counts_ht)]

        box_kws = dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9)

        # --- Error Distribution Plots (side-by-side, shared legend) ---
        with plt.rc_context({'font.size': 16}):
            fig, (ax_err_mass, ax_err_ht) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, constrained_layout=True)

            # Calculate histogram with error bars
            err_counts_mass_plot, err_bins_mass_plot, _ = ax_err_mass.hist(error_mass_gev, bins=100, range=(-1e3, 1e3), alpha=0.7, edgecolor='black', label='Mass Error')
            err_counts_ht_plot, err_bins_ht_plot, _ = ax_err_ht.hist(error_ht_gev, bins=100, range=(-1e3, 1e3), alpha=0.7, edgecolor='black', label='HT Error')
            
            # Add statistical errors (Poisson)
            bin_centers_err_mass = 0.5 * (err_bins_mass_plot[:-1] + err_bins_mass_plot[1:])
            err_mass_errors = np.sqrt(err_counts_mass_plot)
            ax_err_mass.errorbar(bin_centers_err_mass, err_counts_mass_plot, yerr=err_mass_errors, fmt='none', ecolor='black', elinewidth=1, capsize=3, alpha=0.6)
            
            bin_centers_err_ht = 0.5 * (err_bins_ht_plot[:-1] + err_bins_ht_plot[1:])
            err_ht_errors = np.sqrt(err_counts_ht_plot)
            ax_err_ht.errorbar(bin_centers_err_ht, err_counts_ht_plot, yerr=err_ht_errors, fmt='none', ecolor='black', elinewidth=1, capsize=3, alpha=0.6)

            ax_err_mass.set_xlabel('Mass Prediction Error [GeV]')
            ax_err_mass.set_ylabel('Frequency')
            ax_err_ht.set_xlabel('$H_T$ Prediction Error [GeV]')

            textstr_mass = '\n'.join([
                f'RMSE: {self.invariant_mass_rmse:.2f} GeV',
                f'Mean Error: {mean_error_mass:.2f} GeV',
                f'Std Dev: {std_error_mass:.2f} GeV',
                f'Mode: {mode_error_mass:.2f} GeV',
                f'Pearson R: {self.invariant_mass_corr:.4f}'
            ])
            ax_err_mass.text(0.02, 0.98, textstr_mass, transform=ax_err_mass.transAxes, fontsize=14,
                             va='top', bbox=box_kws, color='black')

            textstr_ht = '\n'.join([
                f'RMSE: {self.ht_rmse:.2f} GeV',
                f'Mean Error: {mean_error_ht:.2f} GeV',
                f'Std Dev: {std_error_ht:.2f} GeV',
                f'Mode: {mode_error_ht:.2f} GeV',
                f'Pearson R: {self.ht_corr:.4f}'
            ])
            ax_err_ht.text(0.02, 0.98, textstr_ht, transform=ax_err_ht.transAxes, fontsize=14,
                           va='top', bbox=box_kws, color='black')

            ax_err_mass.grid(True, alpha=0.3)
            ax_err_ht.grid(True, alpha=0.3)

            # Get legend handles from axis
            handles_err_mass = ax_err_mass.get_legend_handles_labels()[0]
            handles_err_ht = ax_err_ht.get_legend_handles_labels()[0]
            if handles_err_mass or handles_err_ht:
                fig.legend(['Mass Error', 'HT Error'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05), fontsize=14)

            fig.savefig('Transformer_mass_ht_error_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)

        # --- Distribution Plots with Ratio (side-by-side, shared legend) ---
        bins = 100
        x_min, x_max = 0, 6e3
        x_min_ht, x_max_ht = 0, 4e3

        with plt.rc_context({'font.size': 16}):
            fig = plt.figure(figsize=(16, 8))
            gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[3, 1], hspace=0.05, wspace=0.25)
            ax_main_mass = fig.add_subplot(gs[0, 0])
            ax_main_ht = fig.add_subplot(gs[0, 1])
            ax_ratio_mass = fig.add_subplot(gs[1, 0], sharex=ax_main_mass)
            ax_ratio_ht = fig.add_subplot(gs[1, 1], sharex=ax_main_ht)

            main_kwargs = dict(bins=bins, histtype='step', linewidth=3)
            mass_true = ax_main_mass.hist(y_true_mass_gev, range=(x_min, x_max), color='blue', label='True', **main_kwargs)
            mass_pred = ax_main_mass.hist(y_pred_mass_gev, range=(x_min, x_max), color='red', label='Predicted', **main_kwargs)
            ht_true = ax_main_ht.hist(y_true_ht_gev, range=(x_min_ht, x_max_ht), color='blue', **main_kwargs)
            ht_pred = ax_main_ht.hist(y_pred_ht_gev, range=(x_min_ht, x_max_ht), color='red', **main_kwargs)

            # Add statistical errors (Poisson) for mass
            counts_true_mass, bin_edges_mass = np.histogram(y_true_mass_gev, bins=bins, range=(x_min, x_max))
            counts_pred_mass, _ = np.histogram(y_pred_mass_gev, bins=bins, range=(x_min, x_max))
            bin_centers_mass_plot = 0.5 * (bin_edges_mass[:-1] + bin_edges_mass[1:])
            errors_true_mass = np.sqrt(counts_true_mass)
            errors_pred_mass = np.sqrt(counts_pred_mass)
            ax_main_mass.errorbar(bin_centers_mass_plot, counts_true_mass, yerr=errors_true_mass, fmt='none', ecolor='blue', elinewidth=1, capsize=3, alpha=0.6)
            ax_main_mass.errorbar(bin_centers_mass_plot, counts_pred_mass, yerr=errors_pred_mass, fmt='none', ecolor='red', elinewidth=1, capsize=3, alpha=0.6)
            
            # Add statistical errors (Poisson) for HT
            counts_true_ht, bin_edges_ht = np.histogram(y_true_ht_gev, bins=bins, range=(x_min_ht, x_max_ht))
            counts_pred_ht, _ = np.histogram(y_pred_ht_gev, bins=bins, range=(x_min_ht, x_max_ht))
            bin_centers_ht_plot = 0.5 * (bin_edges_ht[:-1] + bin_edges_ht[1:])
            errors_true_ht = np.sqrt(counts_true_ht)
            errors_pred_ht = np.sqrt(counts_pred_ht)
            ax_main_ht.errorbar(bin_centers_ht_plot, counts_true_ht, yerr=errors_true_ht, fmt='none', ecolor='blue', elinewidth=1, capsize=3, alpha=0.6)
            ax_main_ht.errorbar(bin_centers_ht_plot, counts_pred_ht, yerr=errors_pred_ht, fmt='none', ecolor='red', elinewidth=1, capsize=3, alpha=0.6)

            ax_main_mass.set_ylabel('Frequency')

            textstr_mass = '\n'.join([
                f'Pearson R: {self.invariant_mass_corr:.4f}',
                f'RMSE: {self.invariant_mass_rmse:.2f} GeV'
            ])
            ax_main_mass.text(0.98, 0.98, textstr_mass, transform=ax_main_mass.transAxes,
                             fontsize=16, va='top', ha='right', bbox=box_kws, color='black')

            textstr_ht = '\n'.join([
                f'Pearson R: {self.ht_corr:.4f}',
                f'RMSE: {self.ht_rmse:.2f} GeV'
            ])
            ax_main_ht.text(0.98, 0.98, textstr_ht, transform=ax_main_ht.transAxes,
                            fontsize=16, va='top', ha='right', bbox=box_kws, color='black')

            ax_main_mass.grid(True, alpha=0.2)
            ax_main_ht.grid(True, alpha=0.2)

            # Ratio plot for mass
            ratio_mass = np.divide(counts_pred_mass, counts_true_mass, where=counts_true_mass != 0)
            ratio_mass[np.isnan(ratio_mass)] = 0.0
            ax_ratio_mass.plot(bin_centers_mass_plot, ratio_mass, color='black', linewidth=1.2)
            ax_ratio_mass.axhline(1.0, color='red', linestyle='--', linewidth=1)
            ax_ratio_mass.set_ylabel('Pred/True')
            ax_ratio_mass.set_xlabel('Invariant Mass [GeV]')
            ax_ratio_mass.set_ylim(0, 2)
            ax_ratio_mass.grid(True, alpha=0.2, axis='y')

            # Ratio plot for HT
            ratio_ht = np.divide(counts_pred_ht, counts_true_ht, where=counts_true_ht != 0)
            ratio_ht[np.isnan(ratio_ht)] = 0.0
            ax_ratio_ht.plot(bin_centers_ht_plot, ratio_ht, color='black', linewidth=1.2)
            ax_ratio_ht.axhline(1.0, color='red', linestyle='--', linewidth=1)
            ax_ratio_ht.set_ylabel('Pred/True')
            ax_ratio_ht.set_xlabel('$H_T$ [GeV]')
            ax_ratio_ht.set_ylim(0, 2)
            ax_ratio_ht.grid(True, alpha=0.2, axis='y')

            plt.setp(ax_main_mass.get_xticklabels(), visible=False)
            plt.setp(ax_main_ht.get_xticklabels(), visible=False)

            handles = [mass_true[2][0], mass_pred[2][0]]
            fig.legend(handles, ['True', 'Predicted'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=18)

            fig.savefig('Transformer_mass_ht_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)

        # --- 2D Histograms ---
        bins_2d_mass = 250
        bins_2d_ht = 250
        mass_range = [[0, 6e3], [0, 6e3]]
        ht_range = [[0, 4e3], [0, 4e3]]

        counts_mass_2d, _, _ = np.histogram2d(
            y_true_mass_gev, y_pred_mass_gev, bins=bins_2d_mass, range=mass_range
        )
        counts_ht_2d, _, _ = np.histogram2d(
            y_true_ht_gev, y_pred_ht_gev, bins=bins_2d_ht, range=ht_range
        )
        max_count = max(counts_mass_2d.max(), counts_ht_2d.max())
        norm_shared = mcolors.Normalize(vmin=0, vmax=max_count if max_count > 0 else 1)

        with plt.rc_context({'font.size': 20}):
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

            h_mass = axes[0].hist2d(
                y_true_mass_gev, y_pred_mass_gev,
                bins=bins_2d_mass, range=mass_range, cmap='hot', norm=norm_shared, cmin=1
            )
            axes[0].plot([0, 6e3], [0, 6e3], color='red', linestyle='--', linewidth=1, label='Perfect prediction')
            axes[0].set_xlabel('Expected Invariant Mass [GeV]')
            axes[0].set_ylabel('Predicted Invariant Mass [GeV]')
            textstr_mass = '\n'.join([
                f'Pearson R: {self.invariant_mass_corr:.4f}',
                f'RMSE: {self.invariant_mass_rmse:.2f} GeV',
                f'Bias: {mode_error_mass:.2f} GeV'
            ])
            axes[0].text(0.02, 0.98, textstr_mass, transform=axes[0].transAxes,
                         fontsize=20, va='top', bbox=box_kws, color='black')

            h_ht = axes[1].hist2d(
                y_true_ht_gev, y_pred_ht_gev,
                bins=bins_2d_ht, range=ht_range, cmap='hot', norm=norm_shared, cmin=1
            )
            axes[1].plot([0, 4e3], [0, 4e3], color='red', linestyle='--', linewidth=1)
            axes[1].set_xlabel('Expected $H_T$ [GeV]')
            axes[1].set_ylabel('Predicted $H_T$ [GeV]')
            textstr_ht = '\n'.join([
                f'Pearson R: {self.ht_corr:.4f}',
                f'RMSE: {self.ht_rmse:.2f} GeV',
                f'Bias: {mode_error_ht:.2f} GeV'
            ])
            axes[1].text(0.02, 0.98, textstr_ht, transform=axes[1].transAxes,
                         fontsize=20, va='top', bbox=box_kws, color='black')

            cbar = fig.colorbar(h_mass[3], ax=axes, label='Counts')
            cbar.ax.tick_params(labelsize=18)

            handles_2d = axes[0].get_legend_handles_labels()
            if handles_2d[0]:
                fig.legend(handles_2d[0], handles_2d[1], loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=18)

            fig.savefig('Transformer_mass_ht_2d_histogram.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)

        # --- Binned Plots (side-by-side, shared colorbar) ---
        bins_plot = 5
        x_range_mass = (0.5e3, 3e3)
        y_range_mass = (0.5e3, 3e3)
        x_range_ht = (0, 4e3)
        y_range_ht = (0, 4e3)

        H_mass, xedges_mass, yedges_mass = np.histogram2d(y_true_mass_gev, y_pred_mass_gev, bins=bins_plot, range=[x_range_mass, y_range_mass])
        H_ht, xedges_ht, yedges_ht = np.histogram2d(y_true_ht_gev, y_pred_ht_gev, bins=bins_plot, range=[x_range_ht, y_range_ht])

        total_mass = H_mass.sum(); total_ht = H_ht.sum()
        H_percent_mass = (100.0 * H_mass / total_mass) if total_mass > 0 else H_mass
        H_percent_ht = (100.0 * H_ht / total_ht) if total_ht > 0 else H_ht

        vmax_shared = max(H_percent_mass.max(), H_percent_ht.max())
        vmax_shared = vmax_shared if vmax_shared > 0 else 1
        norm_percent = mcolors.Normalize(vmin=0.0, vmax=vmax_shared)

        with plt.rc_context({'font.size': 16}):
            fig, (ax_binned_mass, ax_binned_ht) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

            pcm1 = ax_binned_mass.pcolormesh(
                xedges_mass, yedges_mass, H_percent_mass.T, cmap='hot', norm=norm_percent
            )
            xcenters_mass = 0.5 * (xedges_mass[:-1] + xedges_mass[1:])
            ycenters_mass = 0.5 * (yedges_mass[:-1] + yedges_mass[1:])
            for i, xc in enumerate(xcenters_mass):
                for j, yc in enumerate(ycenters_mass):
                    val = H_percent_mass[i, j]
                    txt_color = 'white' if val >= 40 else 'black'
                    ax_binned_mass.text(xc, yc, f"{val:.1f}%", ha='center', va='center',
                                         fontsize=12, fontweight='bold', color=txt_color)

            ax_binned_mass.set_xlim(x_range_mass); ax_binned_mass.set_ylim(y_range_mass)
            ax_binned_mass.set_xlabel('Expected Invariant Mass [GeV]')
            ax_binned_mass.set_ylabel('Predicted Invariant Mass [GeV]')
            textstr_mass = f'Pearson R: {self.invariant_mass_corr:.4f}'
            ax_binned_mass.text(0.98, 0.02, textstr_mass, transform=ax_binned_mass.transAxes, fontsize=14,
                                va='bottom', ha='right', bbox=box_kws, color='black')

            pcm2 = ax_binned_ht.pcolormesh(
                xedges_ht, yedges_ht, H_percent_ht.T, cmap='hot', norm=norm_percent
            )
            xcenters_ht = 0.5 * (xedges_ht[:-1] + xedges_ht[1:])
            ycenters_ht = 0.5 * (yedges_ht[:-1] + yedges_ht[1:])
            for i, xc in enumerate(xcenters_ht):
                for j, yc in enumerate(ycenters_ht):
                    val = H_percent_ht[i, j]
                    txt_color = 'white' if val >= 40 else 'black'
                    ax_binned_ht.text(xc, yc, f"{val:.1f}%", ha='center', va='center',
                                       fontsize=12, fontweight='bold', color=txt_color)

            ax_binned_ht.set_xlim(x_range_ht); ax_binned_ht.set_ylim(y_range_ht)
            ax_binned_ht.set_xlabel('Expected $H_T$ [GeV]')
            ax_binned_ht.set_ylabel('Predicted $H_T$ [GeV]')
            textstr_ht = f'Pearson R: {self.ht_corr:.4f}'
            ax_binned_ht.text(0.98, 0.02, textstr_ht, transform=ax_binned_ht.transAxes, fontsize=14,
                              va='bottom', ha='right', bbox=box_kws, color='black')

            cbar = fig.colorbar(pcm1, ax=[ax_binned_mass, ax_binned_ht], label='Percentage of events (%)')
            cbar.ax.tick_params(labelsize=14)

            fig.savefig('Transformer_mass_ht_binned_plot_percent.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)

        # --- Training and Validation Loss Curves ---
        self.plot_loss_curves()