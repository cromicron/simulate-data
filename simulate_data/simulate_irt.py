import numpy as np
import matplotlib.pyplot as plt


class IRTItem:
    """Base class for IRT items."""

    def __init__(self, discrimination=1.0, difficulty=0.0):
        # discrimination can be scalar or vector (for multidimensional models)
        self.a = np.atleast_1d(discrimination).astype(float)
        self.b = difficulty  # may be None for polytomous models (GRM/Hybrid)

    def _linear_predictor(self, theta):
        """
        Compute z = theta @ a - b, robust to 1D or 2D theta and optional b=None.
        theta:
          - shape (n_persons,) for 1D models (plot grids or 1D sims)
          - shape (n_persons, n_dim) for multi-D models
        returns:
          - shape (n_persons,)
        """
        theta = np.asarray(theta)

        # Case: unidimensional theta grid or vector of persons
        if theta.ndim == 1 and self.a.size == 1:
            z = self.a[0] * theta
            if self.b is not None:
                z = z - self.b
            return z

        # Case: multidimensional persons (n_persons, n_dim)
        if theta.ndim == 2:
            z = theta @ self.a
            if self.b is not None:
                z = z - self.b
            return z

        raise ValueError("Unsupported shape for theta or discrimination.")

    def _prepare_theta_for_plot(self, theta_range, n_points):
        """
        Generate theta grid for plotting. If item is multidimensional, vary the first
        factor across the grid and set all remaining factors to zero.
        Returns:
          - 1D array for unidimensional items (n_points,)
          - 2D array for multidimensional items (n_points, n_dim)
        """
        theta = np.linspace(*theta_range, n_points)
        if self.a.size > 1:
            # Multi-D: vary factor 0, hold others at 0
            theta = np.column_stack([theta, np.zeros((n_points, self.a.size - 1))])
        return theta

    def simulate(self, theta):
        raise NotImplementedError("Subclasses must implement simulate()")

    def plot(self, theta_range=(-4, 4), n_points=200, cumulative=False):
        raise NotImplementedError("Subclasses must implement plot()")


# ---------------------------
# Dichotomous items
# ---------------------------

class IRT2PLItem(IRTItem):
    """2PL binary item."""

    def p_correct(self, theta):
        z = self._linear_predictor(theta)
        return 1 / (1 + np.exp(-z))

    def simulate(self, theta):
        p = self.p_correct(theta)
        return np.random.binomial(1, p)

    def plot(self, theta_range=(-4, 4), n_points=200, cumulative=False):
        theta = self._prepare_theta_for_plot(theta_range, n_points)
        p = self.p_correct(theta)
        x = theta if np.asarray(theta).ndim == 1 else theta[:, 0]
        title_suffix = f"(a={self.a})"
        plt.plot(x, p, label="2PL ICC")
        plt.title("2PL Item Characteristic Curve " + title_suffix)
        plt.xlabel(r"$\theta$")
        plt.ylabel("P(correct)")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()


class IRT3PLItem(IRT2PLItem):
    """3PL binary item with guessing parameter c."""

    def __init__(self, discrimination=1.0, difficulty=0.0, guessing=0.2):
        super().__init__(discrimination, difficulty)
        self.c = guessing

    def p_correct(self, theta):
        z = self._linear_predictor(theta)
        return self.c + (1 - self.c) / (1 + np.exp(-z))

    def plot(self, theta_range=(-4, 4), n_points=200, cumulative=False):
        theta = self._prepare_theta_for_plot(theta_range, n_points)
        p = self.p_correct(theta)
        x = theta if np.asarray(theta).ndim == 1 else theta[:, 0]
        title_suffix = f"(a={self.a}, c={self.c:.2f})"
        plt.plot(x, p, label="3PL ICC")
        plt.title("3PL Item Characteristic Curve " + title_suffix)
        plt.xlabel(r"$\theta$")
        plt.ylabel("P(correct)")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()


# ---------------------------
# Polytomous items
# ---------------------------

class GradedResponseItem(IRTItem):
    """Samejima's GRM for ordinal items."""

    def __init__(self, discrimination=1.0, thresholds=None):
        # For GRM, 'difficulty' is represented by thresholds; set b=None
        super().__init__(discrimination, difficulty=None)
        if thresholds is None:
            raise ValueError("GRM requires thresholds.")
        self.thresholds = np.array(thresholds, dtype=float)

    def category_probs(self, theta):
        """
        Returns:
          probs: (n_persons, n_categories)
          Pstar: (n_persons, n_thresholds+2) cumulative boundaries [1, P*_1..P*_{m-1}, 0]
        """
        z_core = self._linear_predictor(theta)  # shape (n_persons,)
        z = z_core[:, None] - self.thresholds[None, :]  # (n_persons, m-1)
        Pstar_inner = 1 / (1 + np.exp(-z))
        Pstar = np.hstack([
            np.ones((Pstar_inner.shape[0], 1)),
            Pstar_inner,
            np.zeros((Pstar_inner.shape[0], 1))
        ])
        probs = Pstar[:, :-1] - Pstar[:, 1:]
        probs = np.clip(probs, 1e-12, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs, Pstar

    def simulate(self, theta):
        probs, _ = self.category_probs(theta)
        return np.array([np.random.choice(len(p), p=p) for p in probs])

    def plot(self, theta_range=(-4, 4), n_points=200, cumulative=False):
        theta = self._prepare_theta_for_plot(theta_range, n_points)
        probs, Pstar = self.category_probs(theta)
        x = theta if np.asarray(theta).ndim == 1 else theta[:, 0]
        title_suffix = f"(a={self.a})"

        if cumulative:
            for k in range(1, Pstar.shape[1] - 1):
                plt.plot(x, Pstar[:, k], label=f"P*(≥{k})")
            plt.title("GRM Cumulative Curves " + title_suffix)
            plt.ylabel("Cumulative probability")
        else:
            for k in range(probs.shape[1]):
                plt.plot(x, probs[:, k], label=f"Category {k}")
            plt.title("GRM Category Response Curves " + title_suffix)
            plt.ylabel("P(category)")

        plt.xlabel(r"$\theta$")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()


class HybridOrdinalItem(IRTItem):
    """
    Hybrid ordinal item with threshold-specific slopes.
    discrimination: scalar or vector (n_dim) loading on latent vector
    difficulties: thresholds (length = n_cat - 1)
    slopes: per-threshold discriminations (length = n_cat - 1)
    """

    def __init__(self, discrimination=1.0, difficulties=None, slopes=None):
        if difficulties is None:
            raise ValueError("HybridOrdinalItem requires difficulties (thresholds).")
        self.difficulties = np.array(difficulties, dtype=float)
        self.slopes = (
            np.array(slopes, dtype=float)
            if slopes is not None
            else np.repeat(np.atleast_1d(discrimination).astype(float)[0], len(self.difficulties))
        )
        # store 'a' as vector (n_dim) for multi-D; no global b here
        self.a = np.atleast_1d(discrimination).astype(float)
        self.b = None

    def category_probs(self, theta):
        theta = np.asarray(theta)
        if theta.ndim == 1:
            theta = theta[:, None]
        z_core = theta @ self.a[:, None]  # (n_persons, 1)
        z = self.slopes[None, :] * (z_core - self.difficulties[None, :])
        Pstar_inner = 1 / (1 + np.exp(-z))
        Pstar = np.hstack([
            np.ones((Pstar_inner.shape[0], 1)),
            Pstar_inner,
            np.zeros((Pstar_inner.shape[0], 1))
        ])
        probs = Pstar[:, :-1] - Pstar[:, 1:]
        probs = np.clip(probs, 1e-12, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs, Pstar

    def simulate(self, theta):
        probs, _ = self.category_probs(theta)
        return np.array([np.random.choice(len(p), p=p) for p in probs])

    def plot(self, theta_range=(-4, 4), n_points=200, cumulative=False):
        theta = self._prepare_theta_for_plot(theta_range, n_points)
        probs, Pstar = self.category_probs(theta)
        x = theta if np.asarray(theta).ndim == 1 else theta[:, 0]
        title_suffix = f"(a={self.a}, slopes={self.slopes})"

        if cumulative:
            for k in range(1, Pstar.shape[1] - 1):
                plt.plot(x, Pstar[:, k], label=f"P*(≥{k})")
            plt.title("Hybrid Ordinal Cumulative Curves " + title_suffix)
            plt.ylabel("Cumulative probability")
        else:
            for k in range(probs.shape[1]):
                plt.plot(x, probs[:, k], label=f"Category {k}")
            plt.title("Hybrid Ordinal Category Curves " + title_suffix)
            plt.ylabel("P(category)")

        plt.xlabel(r"$\theta$")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()


class NominalResponseItem(IRTItem):
    """Bock's NRM for nominal categories."""

    def __init__(self, slopes, intercepts):
        # slopes shape: (n_dim, n_categories) or (1, n_categories)
        self.slopes = np.atleast_2d(slopes).astype(float)
        self.intercepts = np.array(intercepts, dtype=float)
        if self.slopes.shape[1] != len(self.intercepts):
            raise ValueError("slopes must have shape (n_dim, n_categories) to match intercepts length.")
        # no global difficulty b
        self.b = None
        # keep a for consistency in plotting helper (dimension awareness)
        self.a = np.ones(self.slopes.shape[0], dtype=float)

    def category_probs(self, theta):
        theta = np.asarray(theta)
        if theta.ndim == 1:  # plotting grid or 1D sims
            theta = theta[:, None]  # shape (n_points, 1)
        z = theta @ self.slopes + self.intercepts[None, :]  # (n_persons, n_cat)
        z_shift = z - np.max(z, axis=1, keepdims=True)
        expz = np.exp(z_shift)
        probs = expz / expz.sum(axis=1, keepdims=True)
        return probs

    def simulate(self, theta):
        probs = self.category_probs(theta)
        return np.array([np.random.choice(len(p), p=p) for p in probs])

    def plot(self, theta_range=(-4, 4), n_points=200, cumulative=False):
        theta = self._prepare_theta_for_plot(theta_range, n_points)
        probs = self.category_probs(theta)
        x = theta if np.asarray(theta).ndim == 1 else theta[:, 0]
        title_suffix = f"(slopes shape={self.slopes.shape})"

        for k in range(probs.shape[1]):
            plt.plot(x, probs[:, k], label=f"Category {k}")
        plt.title("NRM Category Response Curves " + title_suffix)
        plt.xlabel(r"$\theta$")
        plt.ylabel("P(category)")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    np.random.seed(123)

    theta_trait = np.random.normal(0, 1, size=200)              # main latent trait
    theta_method = np.random.normal(0, 1, size=200)             # method factor
    theta_2d = np.column_stack([theta_trait, theta_method])     # persons × 2 dims

    # --- 2PL ---
    print("\n--- 2PL ---")
    item_trait = IRT2PLItem(discrimination=1.0, difficulty=0.0)  # trait only
    item_trait.plot()
    print("2PL trait-only simulate:", item_trait.simulate(theta_trait)[:10])

    item_trait_method = IRT2PLItem(discrimination=[1.0, 0.4], difficulty=0.0)
    item_trait_method.plot()
    print("2PL trait+method simulate:", item_trait_method.simulate(theta_2d)[:10])

    # --- 3PL ---
    print("\n--- 3PL ---")
    item3pl = IRT3PLItem(discrimination=1.0, difficulty=-0.5, guessing=0.2)
    item3pl.plot()
    print("3PL simulate:", item3pl.simulate(theta_trait)[:10])

    # --- GRM ---
    print("\n--- GRM ---")
    grm_trait = GradedResponseItem(1.0, thresholds=[-1, 0, 1, 2])
    grm_trait.plot(cumulative=False)
    print("GRM trait-only simulate:", grm_trait.simulate(theta_trait)[:10])

    grm_trait_method = GradedResponseItem(discrimination=[1.0, 0.3],
                                          thresholds=[-1, 0, 1, 2])
    grm_trait_method.plot(cumulative=False)
    print("GRM trait+method simulate:", grm_trait_method.simulate(theta_2d)[:10])

    # --- Hybrid Ordinal ---
    print("\n--- Hybrid ---")
    hybrid_trait = HybridOrdinalItem(discrimination=1.0,
                                     difficulties=[-1.5, 0.0, 1.0],
                                     slopes=[0.5, 1.5, 0.8])
    hybrid_trait.plot(cumulative=False)
    print("Hybrid trait-only simulate:", hybrid_trait.simulate(theta_trait)[:10])

    hybrid_trait_method = HybridOrdinalItem(discrimination=[1.0, 0.4],
                                            difficulties=[-1.5, 0.0, 1.0],
                                            slopes=[0.5, 1.5, 0.8])
    hybrid_trait_method.plot(cumulative=False)
    print("Hybrid trait+method simulate:", hybrid_trait_method.simulate(theta_2d)[:10])

    # --- NRM ---
    print("\n--- NRM ---")
    nrm_trait = NominalResponseItem(slopes=[[0.5, -0.2, 0.8]],
                                    intercepts=[0.0, 1.0, -0.5])
    nrm_trait.plot()
    print("NRM trait-only simulate:", nrm_trait.simulate(theta_trait)[:10])

    nrm_trait_method = NominalResponseItem(slopes=[[0.8, -0.5, 0.3],
                                                   [0.3,  0.2, -0.1]],
                                           intercepts=[0.0, 1.0, -0.5])
    nrm_trait_method.plot()
    print("NRM trait+method simulate:", nrm_trait_method.simulate(theta_2d)[:10])
