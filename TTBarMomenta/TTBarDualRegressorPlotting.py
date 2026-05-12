import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from matplotlib import gridspec


class TransformerPlotting:
    def __init__(self, y_true, y_pred, metrics, differences, train_loss_history=None, val_loss_history=None, suffix=""):
        self.y_true = y_true
        self.y_pred = y_pred
        self.metrics = metrics
        self.differences = differences
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history
        self.suffix = suffix
        self.component_names = [
            "antinu_px",
            "antinu_py",
            "antinu_pz",
            "nu_px",
            "nu_py",
            "nu_pz",
        ]

    @staticmethod
    def _axis_range(values, low_q=1.0, high_q=99.0, pad_frac=0.05):
        lo = np.percentile(values, low_q)
        hi = np.percentile(values, high_q)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = np.min(values)
            hi = np.max(values)
            if lo == hi:
                lo -= 1.0
                hi += 1.0
        pad = (hi - lo) * pad_frac
        return lo - pad, hi + pad

    def plot_loss_curves(self, outpath=None):
        if outpath is None:
            outpath = f"TTBar_training_validation_loss{self.suffix}.png"
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
            plt.close(fig)

    def plot_component(self, idx, name):
        true_gev = self.y_true[:, idx]
        pred_gev = self.y_pred[:, idx]
        err_gev = self.differences[:, idx]

        corr = self.metrics[name]['corr']
        rmse_gev = self.metrics[name]['rmse']
        mean_err_gev = self.metrics[name]['mean_error']
        std_err_gev = self.metrics[name]['std_error']

        err_counts, err_bins = np.histogram(err_gev, bins=200)
        mode_error = 0.5 * (err_bins[:-1] + err_bins[1:])[np.argmax(err_counts)]

        box_kws = dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9)

        # --- Error Distribution ---
        err_low, err_high = self._axis_range(err_gev, low_q=0.5, high_q=99.5)
        with plt.rc_context({'font.size': 16}):
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            counts, bins, _ = ax.hist(err_gev, bins=100, range=(err_low, err_high), alpha=0.7, edgecolor='black', label='Error')
            centers = 0.5 * (bins[:-1] + bins[1:])
            yerr = np.sqrt(counts)
            ax.errorbar(centers, counts, yerr=yerr, fmt='none', ecolor='black', elinewidth=1, capsize=3, alpha=0.6)
            ax.set_xlabel(f'{name} Prediction Error [GeV]')
            ax.set_ylabel('Frequency')
            textstr = '\n'.join([
                f'RMSE: {rmse_gev:.2f} GeV',
                f'Mean Error: {mean_err_gev:.2f} GeV',
                f'Std Dev: {std_err_gev:.2f} GeV',
                f'Mode: {mode_error:.2f} GeV',
                f'Pearson R: {corr:.4f}'
            ])
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14, va='top', bbox=box_kws, color='black')
            ax.grid(True, alpha=0.3)
            fig.savefig(f'TTBar_{name}_error_distribution{self.suffix}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        # --- Distribution + Ratio ---
        x_low, x_high = self._axis_range(np.concatenate([true_gev, pred_gev]))
        bins = 100
        with plt.rc_context({'font.size': 16}):
            fig = plt.figure(figsize=(10, 8))
            gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.05)
            ax_main = fig.add_subplot(gs[0, 0])
            ax_ratio = fig.add_subplot(gs[1, 0], sharex=ax_main)

            main_kwargs = dict(bins=bins, range=(x_low, x_high), histtype='step', linewidth=3)
            h_true = ax_main.hist(true_gev, color='blue', label='True', **main_kwargs)
            h_pred = ax_main.hist(pred_gev, color='red', label='Predicted', **main_kwargs)

            counts_true, edges = np.histogram(true_gev, bins=bins, range=(x_low, x_high))
            counts_pred, _ = np.histogram(pred_gev, bins=bins, range=(x_low, x_high))
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax_main.errorbar(centers, counts_true, yerr=np.sqrt(counts_true), fmt='none', ecolor='blue', elinewidth=1, capsize=3, alpha=0.6)
            ax_main.errorbar(centers, counts_pred, yerr=np.sqrt(counts_pred), fmt='none', ecolor='red', elinewidth=1, capsize=3, alpha=0.6)

            ax_main.set_ylabel('Frequency')
            textstr = '\n'.join([
                f'Pearson R: {corr:.4f}',
                f'RMSE: {rmse_gev:.2f} GeV'
            ])
            ax_main.text(0.98, 0.98, textstr, transform=ax_main.transAxes,
                         fontsize=16, va='top', ha='right', bbox=box_kws, color='black')
            ax_main.grid(True, alpha=0.2)

            ratio = np.divide(counts_pred, counts_true, where=counts_true != 0)
            ratio[np.isnan(ratio)] = 0.0
            ax_ratio.plot(centers, ratio, color='black', linewidth=1.2)
            ax_ratio.axhline(1.0, color='red', linestyle='--', linewidth=1)
            ax_ratio.set_ylabel('Pred/True')
            ax_ratio.set_xlabel(f'{name} [GeV]')
            ax_ratio.set_ylim(0, 2)
            ax_ratio.grid(True, alpha=0.2, axis='y')
            plt.setp(ax_main.get_xticklabels(), visible=False)

            handles = [h_true[2][0], h_pred[2][0]]
            fig.legend(handles, ['True', 'Predicted'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=14)
            fig.savefig(f'TTBar_{name}_distribution{self.suffix}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        # --- 2D Histogram ---
        bins_2d = 200
        xy_low, xy_high = self._axis_range(np.concatenate([true_gev, pred_gev]))
        with plt.rc_context({'font.size': 18}):
            fig, ax = plt.subplots(1, 1, figsize=(8, 7), constrained_layout=True)
            h = ax.hist2d(
                true_gev,
                pred_gev,
                bins=bins_2d,
                range=[[xy_low, xy_high], [xy_low, xy_high]],
                cmap='hot',
                cmin=1,
            )
            ax.plot([xy_low, xy_high], [xy_low, xy_high], color='red', linestyle='--', linewidth=1)
            ax.set_xlabel(f'Expected {name} [GeV]')
            ax.set_ylabel(f'Predicted {name} [GeV]')
            textstr = '\n'.join([
                f'Pearson R: {corr:.4f}',
                f'RMSE: {rmse_gev:.2f} GeV',
                f'Bias: {mode_error:.2f} GeV'
            ])
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14, va='top', bbox=box_kws, color='black')
            fig.colorbar(h[3], ax=ax, label='Counts')
            # Place 'Perfect prediction' legend above the plot
            from matplotlib.lines import Line2D
            legend_line = Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Perfect prediction')
            ax.legend(handles=[legend_line], loc='lower right', bbox_to_anchor=(1.0, 1.01), fontsize=12, frameon=False)
            fig.savefig(f'TTBar_{name}_2d_histogram{self.suffix}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        # --- Binned percentage plot ---
        bins_plot = 5
        H, xedges, yedges = np.histogram2d(
            true_gev,
            pred_gev,
            bins=bins_plot,
            range=[[xy_low, xy_high], [xy_low, xy_high]],
        )
        total = H.sum()
        H_percent = (100.0 * H / total) if total > 0 else H
        vmax = H_percent.max() if H_percent.max() > 0 else 1
        norm_percent = mcolors.Normalize(vmin=0.0, vmax=vmax)

        with plt.rc_context({'font.size': 16}):
            fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
            pcm = ax.pcolormesh(xedges, yedges, H_percent.T, cmap='hot', norm=norm_percent)
            xcenters = 0.5 * (xedges[:-1] + xedges[1:])
            ycenters = 0.5 * (yedges[:-1] + yedges[1:])
            for i, xc in enumerate(xcenters):
                for j, yc in enumerate(ycenters):
                    val = H_percent[i, j]
                    txt_color = 'white' if val >= 40 else 'black'
                    ax.text(xc, yc, f"{val:.1f}%", ha='center', va='center', fontsize=12, fontweight='bold', color=txt_color)

            ax.set_xlim((xy_low, xy_high))
            ax.set_ylim((xy_low, xy_high))
            ax.set_xlabel(f'Expected {name} [GeV]')
            ax.set_ylabel(f'Predicted {name} [GeV]')
            ax.text(0.98, 0.02, f'Pearson R: {corr:.4f}', transform=ax.transAxes,
                    fontsize=12, va='bottom', ha='right', bbox=box_kws, color='black')
            fig.colorbar(pcm, ax=ax, label='Percentage of events (%)')
            fig.savefig(f'TTBar_{name}_binned_plot_percent{self.suffix}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

    def _plot_combined_2d(self, indices, names, title_prefix, outpath):
        """Plot a 1x3 grid of 2D histograms for px, py, pz."""
        from matplotlib.lines import Line2D
        box_kws = dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9)
        bins_2d = 200

        with plt.rc_context({'font.size': 14}):
            fig, axes = plt.subplots(1, 3, figsize=(24, 7))
            for ax, idx, name in zip(axes, indices, names):
                true_gev = self.y_true[:, idx]
                pred_gev = self.y_pred[:, idx]
                corr = self.metrics[name]['corr']
                rmse_gev = self.metrics[name]['rmse']

                err_gev = self.differences[:, idx]
                err_counts, err_bins = np.histogram(err_gev, bins=200)
                mode_error = 0.5 * (err_bins[:-1] + err_bins[1:])[np.argmax(err_counts)]

                xy_low, xy_high = self._axis_range(np.concatenate([true_gev, pred_gev]))
                h = ax.hist2d(
                    true_gev, pred_gev,
                    bins=bins_2d,
                    range=[[xy_low, xy_high], [xy_low, xy_high]],
                    cmap='hot',
                    cmin=1,
                )
                ax.plot([xy_low, xy_high], [xy_low, xy_high], color='red', linestyle='--', linewidth=1)
                label = name.split('_')[-1]  # px, py, pz
                ax.set_xlabel(f'Expected {label} [GeV]')
                ax.set_ylabel(f'Predicted {label} [GeV]')
                ax.set_title(name)
                textstr = '\n'.join([
                    f'R: {corr:.4f}',
                    f'RMSE: {rmse_gev:.2f}',
                    f'Bias: {mode_error:.2f}'
                ])
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12, va='top', bbox=box_kws, color='black')
                fig.colorbar(h[3], ax=ax, label='Counts')

            legend_line = Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Perfect prediction')
            fig.legend(handles=[legend_line], loc='upper center', ncol=1, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 1.03))
            fig.suptitle(f'{title_prefix} — Predicted vs Expected [GeV]', fontsize=16, y=1.06)
            fig.savefig(outpath, dpi=300, bbox_inches='tight')
            plt.close(fig)

    def plot_combined_nu(self, outpath=None):
        if outpath is None:
            outpath = f'TTBar_nu_combined_2d{self.suffix}.png'
        self._plot_combined_2d([3, 4, 5], ['nu_px', 'nu_py', 'nu_pz'], 'Neutrino (ν)', outpath)

    def plot_combined_antinu(self, outpath=None):
        if outpath is None:
            outpath = f'TTBar_antinu_combined_2d{self.suffix}.png'
        self._plot_combined_2d([0, 1, 2], ['antinu_px', 'antinu_py', 'antinu_pz'], 'Anti-neutrino (ν̄)', outpath)

    def plot_all(self):
        for idx, name in enumerate(self.component_names):
            self.plot_component(idx, name)
        self.plot_combined_nu()
        self.plot_combined_antinu()
        self.plot_loss_curves()
