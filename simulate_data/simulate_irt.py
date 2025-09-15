import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class IRTItem:
    """Base class for IRT items."""

    def __init__(self, discrimination=1.0, difficulty=0.0, latent_dims=None):
        self.a = np.atleast_1d(discrimination).astype(float)
        self.b = difficulty  # may be None for polytomous models
        self._set_latent_dims(latent_dims)

    def _set_latent_dims(self, latent_dims):
        if latent_dims is None:
            self.latent_dims = [f"Dim{i+1}" for i in range(len(self.a))]
        else:
            if isinstance(latent_dims, str):
                latent_dims = [latent_dims]
            if len(latent_dims) != len(self.a):
                raise ValueError("Length of latent_dims must match number of loadings")
            self.latent_dims = latent_dims

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
    def __init__(self, discrimination=None, difficulty=None, latent_dims=None):
        if discrimination is None:
            discrimination = np.random.uniform(0.5, 2.0)
        if difficulty is None:
            difficulty = np.random.uniform(-2, 2)
        super().__init__(discrimination, difficulty, latent_dims=latent_dims)

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
        title_suffix = f"(a={self.a}, dims={self.latent_dims})"
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
    def __init__(self, discrimination=None, difficulty=None, guessing=None, latent_dims=None):
        if discrimination is None:
            discrimination = np.random.uniform(0.5, 2.0)
        if difficulty is None:
            difficulty = np.random.uniform(-2, 2)
        if guessing is None:
            guessing = np.random.uniform(0.05, 0.25)
        super().__init__(discrimination, difficulty, latent_dims=latent_dims)
        self.c = guessing

    def p_correct(self, theta):
        z = self._linear_predictor(theta)
        return self.c + (1 - self.c) / (1 + np.exp(-z))

    def plot(self, theta_range=(-4, 4), n_points=200, cumulative=False):
        theta = self._prepare_theta_for_plot(theta_range, n_points)
        p = self.p_correct(theta)
        x = theta if np.asarray(theta).ndim == 1 else theta[:, 0]
        title_suffix = f"(a={self.a}, c={self.c:.2f}, dims={self.latent_dims})"
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

    def __init__(self, discrimination=None, thresholds=None, d=None, n_cats=5, latent_dims=None):
        if discrimination is None:
            discrimination = np.random.uniform(0.5, 2.0)
        self.a = np.atleast_1d(discrimination).astype(float)

        if thresholds is not None:
            self.b = np.array(thresholds, dtype=float)
        elif d is not None:
            d = np.array(d, dtype=float)
            self.b = -d / self.a[0]
        else:
            self.b = np.sort(np.random.uniform(-4, 4, size=n_cats - 1))

        self._set_latent_dims(latent_dims)
        self.b_global = None

    @property
    def thresholds(self):
        return self.b

    @property
    def kappa(self):
        return self.b

    @property
    def d(self):
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
        title_suffix = f"(a={self.a}, b={self.b}, d={self.d}, dims={self.latent_dims})"

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
    def __init__(self, discrimination=None, difficulties=None, slopes=None, n_cats=4, latent_dims=None):
        if discrimination is None:
            discrimination = np.random.uniform(0.5, 2.0)
        self.a = np.atleast_1d(discrimination).astype(float)

        if difficulties is None:
            difficulties = np.sort(np.random.uniform(-2, 2, size=n_cats - 1))
        self.b = np.array(difficulties, dtype=float)

        if slopes is None:
            slopes = np.random.uniform(0.5, 2.0, size=len(self.b))
        self.slopes = np.array(slopes, dtype=float)

        self._set_latent_dims(latent_dims)
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
        z_core = (theta @ self.a[:, None])
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
        title_suffix = f"(a={self.a}, b={self.b}, slopes={self.slopes}, dims={self.latent_dims})"

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
    def __init__(self, slopes=None, intercepts=None, n_cats=3, n_dim=1, latent_dims=None):
        if slopes is None:
            slopes = np.random.uniform(-1, 1, size=(n_dim, n_cats))
        if intercepts is None:
            intercepts = np.random.uniform(-1, 1, size=n_cats)
        self.slopes = np.atleast_2d(slopes).astype(float)
        self.intercepts = np.array(intercepts, dtype=float)
        self.a = np.ones(self.slopes.shape[0], dtype=float)
        self.b = None
        self._set_latent_dims(latent_dims)

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
        title_suffix = f"(slopes shape={self.slopes.shape}, dims={self.latent_dims})"
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
# ItemBank
# ---------------------------

class ItemBank:
    def __init__(self, item_type="GRM", n_items=5, latent_dims=None, **kwargs):
        if isinstance(latent_dims, str):
            latent_dims = [latent_dims]
        self.latent_dims = latent_dims if latent_dims is not None else ["Trait"]

        self.item_type = item_type.upper()
        self.n_items = n_items
        self.items = [self._create_item(latent_dims=self.latent_dims, **kwargs)
                      for _ in range(n_items)]

    def _create_item(self, **kwargs):
        if self.item_type == "2PL":
            return IRT2PLItem(**kwargs)
        elif self.item_type == "3PL":
            return IRT3PLItem(**kwargs)
        elif self.item_type == "GRM":
            return GradedResponseItem(**kwargs)
        elif self.item_type == "HYBRID":
            return HybridOrdinalItem(**kwargs)
        elif self.item_type == "NRM":
            return NominalResponseItem(**kwargs)
        else:
            raise ValueError(f"Unknown item_type {self.item_type}")

    def summary(self):
        print(f"ItemBank: {self.n_items} {self.item_type} items")
        print(f"Latent dimensions: {self.latent_dims}")
        for i, item in enumerate(self.items, 1):
            if isinstance(item, GradedResponseItem):
                print(f"  Item {i}: a={item.a}, b={item.b}, d={item.d}")
            elif isinstance(item, HybridOrdinalItem):
                print(f"  Item {i}: a={item.a}, b={item.b}, slopes={item.slopes}")
            elif isinstance(item, IRT2PLItem):
                print(f"  Item {i}: a={item.a}, b={item.b}")
            elif isinstance(item, IRT3PLItem):
                print(f"  Item {i}: a={item.a}, b={item.b}, c={item.c}")
            elif isinstance(item, NominalResponseItem):
                print(f"  Item {i}: slopes shape={item.slopes.shape}")

    def simulate(self, theta_matrix):
        """
        Simulate responses for all items, given wide-format latent scores.

        Args:
            theta_matrix: numpy array or DataFrame of shape (n_persons, n_dims).
                          - If DataFrame, its columns must include all dims
                            referenced by items' .latent_dims

        Returns:
            np.ndarray (n_persons, n_items) of responses
        """

        if isinstance(theta_matrix, pd.DataFrame):
            wide = theta_matrix
        else:
            wide = np.asarray(theta_matrix)

        n_persons = wide.shape[0]
        responses = np.zeros((n_persons, self.n_items), dtype=int)

        for j, item in enumerate(self.items):
            if isinstance(wide, np.ndarray):
                theta_sub = wide
            else:
                dim_idx = [wide.columns.get_loc(dim) for dim in item.latent_dims]
                theta_sub = wide.iloc[:, dim_idx].values
            responses[:, j] = item.simulate(theta_sub)

        return responses



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
    item_trait = IRT2PLItem(discrimination=1.0, difficulty=0.0, latent_dims="Extraversion")
    item_trait.plot()
    print("2PL trait-only simulate:", item_trait.simulate(theta_trait)[:10])

    item_trait_method = IRT2PLItem(discrimination=[1.0, 0.4], difficulty=0.0,
                                   latent_dims=["Extraversion", "Acquiescence"])
    item_trait_method.plot()
    print("2PL trait+method simulate:", item_trait_method.simulate(theta_2d)[:10])

    # --- 3PL ---
    print("\n--- 3PL ---")
    item3pl = IRT3PLItem(discrimination=1.0, difficulty=-0.5, guessing=0.2,
                         latent_dims="Extraversion")
    item3pl.plot()
    print("3PL simulate:", item3pl.simulate(theta_trait)[:10])

    # --- GRM ---
    print("\n--- GRM ---")
    grm_trait = GradedResponseItem(1.0, thresholds=[-1, 0, 1, 2], latent_dims="Extraversion")
    grm_trait.plot(cumulative=False)
    print("GRM trait-only simulate:", grm_trait.simulate(theta_trait)[:10])

    grm_trait_method = GradedResponseItem(discrimination=[1.0, 0.3],
                                          thresholds=[-1, 0, 1, 2],
                                          latent_dims=["Extraversion", "Acquiescence"])
    grm_trait_method.plot(cumulative=False)
    print("GRM trait+method simulate:", grm_trait_method.simulate(theta_2d)[:10])

    # --- Hybrid Ordinal ---
    print("\n--- Hybrid ---")
    hybrid_trait = HybridOrdinalItem(discrimination=1.0,
                                     difficulties=[-1.5, 0.0, 1.0],
                                     slopes=[0.5, 1.5, 0.8],
                                     latent_dims="Extraversion")
    hybrid_trait.plot(cumulative=False)
    print("Hybrid trait-only simulate:", hybrid_trait.simulate(theta_trait)[:10])

    hybrid_trait_method = HybridOrdinalItem(discrimination=[1.0, 0.4],
                                            difficulties=[-1.5, 0.0, 1.0],
                                            slopes=[0.5, 1.5, 0.8],
                                            latent_dims=["Extraversion", "Acquiescence"])
    hybrid_trait_method.plot(cumulative=False)
    print("Hybrid trait+method simulate:", hybrid_trait_method.simulate(theta_2d)[:10])

    # --- NRM ---
    print("\n--- NRM ---")
    nrm_trait = NominalResponseItem(slopes=[[0.5, -0.2, 0.8]],
                                    intercepts=[0.0, 1.0, -0.5],
                                    latent_dims="Extraversion")
    nrm_trait.plot()
    print("NRM trait-only simulate:", nrm_trait.simulate(theta_trait)[:10])

    nrm_trait_method = NominalResponseItem(slopes=[[0.8, -0.5, 0.3],
                                                   [0.3,  0.2, -0.1]],
                                           intercepts=[0.0, 1.0, -0.5],
                                           latent_dims=["Extraversion", "Acquiescence"])
    nrm_trait_method.plot()
    print("NRM trait+method simulate:", nrm_trait_method.simulate(theta_2d)[:10])

    # --- ItemBank ---
    print("\n--- ItemBank ---")
    bank = ItemBank(item_type="GRM", n_items=3, latent_dims="Extraversion", n_cats=5)
    bank.summary()
