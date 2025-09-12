import numpy as np
import matplotlib.pyplot as plt


class IRTItem:
    """Base class for IRT items."""

    def __init__(self, discrimination=1.0, difficulty=0.0):
        self.a = np.atleast_1d(discrimination).astype(float)
        self.b = difficulty  # may be None for polytomous models

    def _ensure_matrix(self, theta):
        """Always return theta as 2D array (n_persons, n_dim)."""
        theta = np.asarray(theta)
        if theta.ndim == 1:
            return theta[:, None]
        return theta

    def _linear_predictor(self, theta):
        """Compute z = theta @ a - b. Always returns (n,)."""
        theta = self._ensure_matrix(theta)
        z = theta @ self.a[:, None]  # (n,1)
        z = z.ravel()
        if self.b is not None:
            z = z - self.b
        return z

    def _prepare_theta_for_plot(self, theta_range, n_points):
        """Grid for plotting. If multi-D, vary first factor, fix others at 0."""
        theta = np.linspace(*theta_range, n_points)
        if self.a.size > 1:
            theta = np.column_stack([theta, np.zeros((n_points, self.a.size - 1))])
        return theta

    def simulate(self, theta):
        raise NotImplementedError

    def plot(self, theta_range=(-4, 4), n_points=200, cumulative=False):
        raise NotImplementedError


# ---------------------------
# Dichotomous items
# ---------------------------

class IRT2PLItem(IRTItem):
    """2PL binary item."""
    def __init__(self, discrimination=None, difficulty=None):
        if discrimination is None:
            discrimination = np.random.uniform(0.5, 2.0)   # slopes
        if difficulty is None:
            difficulty = np.random.uniform(-2, 2)          # threshold
        super().__init__(discrimination, difficulty)


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
    def __init__(self, discrimination=None, difficulty=None, guessing=None):
        if discrimination is None:
            discrimination = np.random.uniform(0.5, 2.0)
        if difficulty is None:
            difficulty = np.random.uniform(-2, 2)
        if guessing is None:
            guessing = np.random.uniform(0.05, 0.25)
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
    """Samejima's GRM for ordinal items, with conversion to/from mirt-style d."""

    def __init__(self, discrimination=None, thresholds=None, d=None, n_cats=5):
        if discrimination is None:
            discrimination = np.random.uniform(0.5, 2.0)
        self.a = np.atleast_1d(discrimination).astype(float)

        # Option 1: user provides Samejima thresholds b
        if thresholds is not None:
            self.b = np.array(thresholds, dtype=float)

        # Option 2: user provides mirt d-parameters
        elif d is not None:
            d = np.array(d, dtype=float)
            self.b = -d / self.a[0]

        # Option 3: generate random thresholds
        else:
            self.b = np.sort(np.random.uniform(-2, 2, size=n_cats - 1))

        self.b_global = None  # no global difficulty

    # --- aliases ---
    @property
    def thresholds(self):
        return self.b

    @property
    def kappa(self):
        return self.b

    @property
    def d(self):
        """mirt-style step parameters (vector)"""
        return -self.a[0] * self.b

    def category_probs(self, theta):
        theta = self._ensure_matrix(theta)
        z_core = (theta @ self.a[:, None]).ravel()[:, None]
        z = z_core - self.b[None, :]
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
        title_suffix = f"(a={self.a}, b={self.b}, d={self.d})"

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
    """Hybrid ordinal item with threshold-specific slopes."""
    def __init__(self, discrimination=None, difficulties=None, slopes=None, n_cats=4):
        if discrimination is None:
            discrimination = np.random.uniform(0.5, 2.0)
        if difficulties is None:
            difficulties = np.sort(np.random.uniform(-2, 2, size=n_cats - 1))
        if slopes is None:
            slopes = np.random.uniform(0.5, 2.0, size=len(difficulties))
        self.b = np.array(difficulties, dtype=float)
        self.slopes = np.array(slopes, dtype=float)
        self.a = np.atleast_1d(discrimination).astype(float)
        self.b_global = None

    @property
    def thresholds(self):
        return self.b

    @property
    def kappa(self):
        return self.b

    @property
    def d(self):
        return -self.b

    def category_probs(self, theta):
        theta = self._ensure_matrix(theta)
        z_core = (theta @ self.a[:, None])  # (n,1)
        z = self.slopes[None, :] * (z_core - self.b[None, :])
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
        title_suffix = f"(a={self.a}, b={self.b}, d={self.d}, slopes={self.slopes})"

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
    def __init__(self, slopes=None, intercepts=None, n_cats=3, n_dim=1):
        if slopes is None:
            slopes = np.random.uniform(-1, 1, size=(n_dim, n_cats))
        if intercepts is None:
            intercepts = np.random.uniform(-1, 1, size=n_cats)
        self.slopes = np.atleast_2d(slopes).astype(float)
        self.intercepts = np.array(intercepts, dtype=float)
        self.b = None
        self.a = np.ones(self.slopes.shape[0], dtype=float)


    def category_probs(self, theta):
        theta = self._ensure_matrix(theta)
        z = theta @ self.slopes + self.intercepts[None, :]
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

    theta_trait = np.random.normal(0, 1, size=200)
    theta_method = np.random.normal(0, 1, size=200)
    theta_2d = np.column_stack([theta_trait, theta_method])

    # --- 2PL ---
    print("\n--- 2PL ---")
    item_trait = IRT2PLItem(discrimination=1.0, difficulty=0.0)
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
