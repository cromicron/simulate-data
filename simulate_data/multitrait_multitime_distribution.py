import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

prop_cycle = plt.cm.get_cmap("tab20").colors  # 20 unique colors
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=prop_cycle)


def check_correlation_matrix(mat: np.ndarray, tol=1e-8) -> bool:
    """
    Check if a matrix is a valid correlation matrix.

    Conditions:
    - Square
    - Symmetric
    - Diagonal all ones
    - Positive semi-definite (all eigenvalues >= 0)

    Args:
        mat: numpy array
        tol: tolerance for numerical checks

    Returns:
        bool: True if valid, False otherwise
    """
    # Check square
    if mat.shape[0] != mat.shape[1]:
        print("❌ Matrix is not square.")
        return False

    # Check symmetry
    if not np.allclose(mat, mat.T, atol=tol):
        print("❌ Matrix is not symmetric.")
        return False

    # Check diagonal ones
    if not np.allclose(np.diag(mat), 1, atol=tol):
        print("❌ Diagonal entries are not all 1.")
        return False

    # Check positive semidefiniteness
    eigenvalues = np.linalg.eigvalsh(mat)
    if np.any(eigenvalues < -tol):  # allow tiny negatives from numerical error
        print(f"❌ Matrix is not positive semidefinite. Min eigenvalue = {eigenvalues.min():.6f}")
        return False

    # If all checks passed
    print("✅ Matrix is a valid correlation matrix.")
    return True


class MTMTDistribution:
    """
    Multi-Trait × Multi-Time population.
    You define the distribution at init; simulate(n) draws a sample.
    """
    def __init__(
        self,
        mu,                 # (T, P) mean matrix
        Sigma_time,         # (T, T) time covariance
        Sigma_traits,       # (P, P) trait covariance
        trait_names=None,   # list of length P
        seed=123,
    ):
        self.mu = np.asarray(mu)
        self.Sigma_time = np.asarray(Sigma_time)
        self.Sigma_traits = np.asarray(Sigma_traits)

        T, P = self.mu.shape
        if self.Sigma_time.shape != (T, T):
            raise ValueError(f"Sigma_time must be ({T},{T})")
        if self.Sigma_traits.shape != (P, P):
            raise ValueError(f"Sigma_traits must be ({P},{P})")

        if trait_names is None:
            trait_names = [f"Trait{j+1}" for j in range(P)]
        if len(trait_names) != P:
            raise ValueError(f"trait_names must have length {P}")
        self.trait_names = trait_names

        # precompute Cholesky factors once
        self.Lt = np.linalg.cholesky(self.Sigma_time)
        self.Lp = np.linalg.cholesky(self.Sigma_traits)

        self.T, self.P = T, P
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def ar1_cov(T, rho=0.7, sigma=1.0):
        idx = np.arange(T)
        return (sigma**2) * (rho ** np.abs(idx[:, None] - idx[None, :]))

    @staticmethod
    def cov_from_sds_corr(sds, R):
        sds = np.asarray(sds)
        R = np.asarray(R)
        D = np.diag(sds)
        return D @ R @ D

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def simulate(self, n, as_df=False):
        """Draw n samples -> (n, T, P) or long DataFrame if as_df=True."""
        Z = self.rng.standard_normal((n, self.T, self.P))
        X = Z @ self.Lp.T
        X = np.einsum('tu, ntp -> nup', self.Lt, X)
        X = X + self.mu

        if not as_df:
            return X

        # long DataFrame: unit, wave, trait, value
        records = []
        for i in range(n):
            for t in range(self.T):
                for p, trait in enumerate(self.trait_names):
                    records.append((i, t, trait, X[i, t, p]))
        return pd.DataFrame(records, columns=["unit", "wave", "trait", "value"])

    # ------------------------
    # Visualization methods
    # ------------------------

    def plot_trait_corr(self):
        """Heatmap of trait correlation (derived from Sigma_traits)."""
        D = np.sqrt(np.diag(self.Sigma_traits))
        R = self.Sigma_traits / np.outer(D, D)
        fig, ax = plt.subplots()
        im = ax.imshow(R, vmin=-1, vmax=1)
        ax.set_xticks(range(self.P)); ax.set_xticklabels(self.trait_names, rotation=45, ha='right')
        ax.set_yticks(range(self.P)); ax.set_yticklabels(self.trait_names)
        ax.set_title("Trait Correlation")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig, ax

    def plot_time_cov(self, correlation=True):
        """Heatmap of time covariance or correlation."""
        if correlation:
            d = np.sqrt(np.diag(self.Sigma_time))
            M = self.Sigma_time / np.outer(d, d)
            title = "Time Correlation"
            vmin, vmax = -1, 1
        else:
            M = self.Sigma_time
            title = "Time Covariance"
            vmin, vmax = None, None

        fig, ax = plt.subplots()
        im = ax.imshow(M, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(self.T)); ax.set_xticklabels([f"t{t+1}" for t in range(self.T)])
        ax.set_yticks(range(self.T)); ax.set_yticklabels([f"t{t+1}" for t in range(self.T)])
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig, ax

    def plot_population_time_profiles(
            self, ci=None, empirical_data=None, show_theoretical=False
    ):
        """
        Plot trajectories across time.

        Args:
            ci: None | '2sd' (only used if show_theoretical=True)
            empirical_data: optional DataFrame from simulate(as_df=True)
            show_theoretical: whether to overlay theoretical curves
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D

        time = np.arange(1, self.T + 1)
        fig, ax = plt.subplots(figsize=(7, 5))

        colors = plt.cm.get_cmap("tab20", len(self.trait_names)).colors

        # --- Theoretical profiles (optional) ---
        if show_theoretical:
            for p, name in enumerate(self.trait_names):
                mean = self.mu[:, p]
                ax.plot(time, mean, lw=2, color=colors[p], label=name)

                if ci == '2sd':
                    var_t = np.diag(self.Sigma_time) * self.Sigma_traits[p, p]
                    sd_t = np.sqrt(var_t)
                    ax.fill_between(time, mean - 2 * sd_t, mean + 2 * sd_t,
                                    alpha=0.15, color=colors[p])

        # --- Empirical overlay ---
        if empirical_data is not None:
            summary = (
                empirical_data
                .groupby(["wave", "trait"])["value"]
                .mean()
                .reset_index()
            )
            for p, name in enumerate(self.trait_names):
                sub = summary.query("trait == @name")
                ax.plot(
                    sub["wave"] + 1,  # +1 if waves are 0-based
                    sub["value"],
                    linestyle="--" if show_theoretical else "-",
                    lw=1.5,
                    color=colors[p],
                    label=name if not show_theoretical else None
                )

        # --- Force full use of canvas ---
        ax.relim()
        ax.autoscale_view()
        ax.margins(y=0.1)

        ax.set_xlabel("Wave")
        ax.set_ylabel("Value")
        ax.set_title("Population Time Profiles")
        ax.set_xticks(time)
        ax.set_xticklabels([str(t) for t in time])

        # --- Legend ---
        handles, labels = ax.get_legend_handles_labels()

        if show_theoretical and empirical_data is not None:
            style_handles = [
                Line2D([0], [0], color="black", lw=2, linestyle="-", label="Theoretical"),
                Line2D([0], [0], color="black", lw=2, linestyle="--", label="Empirical")
            ]
            ax.legend(handles + style_handles,
                      labels + [h.get_label() for h in style_handles],
                      ncol=2, fontsize="small")
        else:
            ax.legend(ncol=2, fontsize="small")

        plt.tight_layout()
        return fig, ax

    def plot_sample_trajectories(self, n=30, trait=None):
        import matplotlib.pyplot as plt
        if trait is None:
            p = 0;
            trait = self.trait_names[0]
        else:
            p = self.trait_names.index(trait)
        X = self.simulate(n)
        time = np.arange(1, self.T + 1)
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot(time, X[i, :, p], alpha=0.6)
        ax.plot(time, self.mu[:, p], linewidth=3)
        ax.set_xlabel("Wave")
        ax.set_ylabel(trait)
        ax.set_title(f"Sample Trajectories ({trait})")
        ax.set_xticks(time)  # << force integer ticks
        ax.set_xticklabels([str(t) for t in time])  # << integer-looking labels
        plt.tight_layout()
        return fig, ax

    def plot_distributions(self, n=500, kind='violin'):
        """
        Simulate n units and show per-wave distributions for each trait.
        kind: 'violin' or 'box'
        """
        X = self.simulate(n)  # (n, T, P)
        time_labels = [f"t{t+1}" for t in range(self.T)]

        fig, axes = plt.subplots(self.P, 1, figsize=(6, 2.5*self.P), sharex=True)
        if self.P == 1:
            axes = [axes]

        for p, ax in enumerate(axes):
            data_by_wave = [X[:, t, p] for t in range(self.T)]
            if kind == 'violin':
                parts = ax.violinplot(data_by_wave, showmeans=True, showextrema=True, showmedians=False)
            elif kind == 'box':
                ax.boxplot(data_by_wave, showfliers=False)
            else:
                raise ValueError("kind must be 'violin' or 'box'")

            ax.set_ylabel(self.trait_names[p])
            ax.set_xticks(range(1, self.T+1))
            ax.set_xticklabels(time_labels)

        axes[-1].set_xlabel("Wave")
        fig.suptitle(f"Per-wave Distributions (n={n})")
        plt.tight_layout()
        return fig, axes



if __name__ == "__main__":
    # Big Five intercorrelations (domains) as an example
    Sigma_traits = np.array([
        [1.00, 0.14, 0.22, -0.34, 0.20],
        [0.14, 1.00, 0.28, -0.29, 0.15],
        [0.22, 0.28, 1.00, -0.30, -0.02],
        [-0.34, -0.29, -0.30, 1.00, -0.06],
        [0.20, 0.15, -0.02, -0.06, 1.00],
    ])
    mu = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.6, 0.2, 0.1, 0.0],
        [1.1, 0.9, 0.3, 0.2, 0.1],
    ])
    Sigma_time = MTMTDistribution.ar1_cov(3, rho=0.6, sigma=1.0)

    trait_names = ["Extraversion", "Agreeableness", "Conscientiousness",
                   "Neuroticism", "Openness"]

    # Define your population...
    pop = MTMTDistribution(mu=mu, Sigma_time=Sigma_time,
                           Sigma_traits=Sigma_traits,
                           trait_names=["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism",
                                        "Openness"],
                           seed=42)

    # Plots
    pop.plot_trait_corr()
    pop.plot_time_cov(correlation=True)
    pop.plot_population_time_profiles()
    pop.plot_sample_trajectories(n=40, trait="Conscientiousness")
    pop.plot_distributions(n=800, kind='violin')
    plt.show()


