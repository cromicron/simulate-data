from simulate_data.survey_simulator import SurveyDataset
import numpy as np
from factor_analyzer import FactorAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo, FactorAnalyzer
import seaborn as sns


if __name__ == "__main__":
    sigma_traits = np.array([
        # ---- Superdim 1 ----
        [1.00, 0.63, 0.58, 0.33, 0.27, 0.36, 0.41, 0.24, 0.29, 0.30, 0.39, 0.26, 0.34, 0.28, 0.32],
        [0.63, 1.00, 0.67, 0.31, 0.29, 0.38, 0.36, 0.25, 0.30, 0.27, 0.35, 0.28, 0.37, 0.26, 0.33],
        [0.58, 0.67, 1.00, 0.32, 0.28, 0.35, 0.37, 0.26, 0.31, 0.29, 0.33, 0.27, 0.36, 0.25, 0.34],

        # ---- Superdim 2 ----
        [0.33, 0.31, 0.32, 1.00, 0.65, 0.59, 0.30, 0.36, 0.27, 0.33, 0.25, 0.37, 0.30, 0.28, 0.32],
        [0.27, 0.29, 0.28, 0.65, 1.00, 0.72, 0.28, 0.35, 0.26, 0.31, 0.27, 0.34, 0.32, 0.29, 0.30],
        [0.36, 0.38, 0.35, 0.59, 0.72, 1.00, 0.31, 0.37, 0.28, 0.34, 0.29, 0.35, 0.31, 0.26, 0.33],

        # ---- Superdim 3 ----
        [0.41, 0.36, 0.37, 0.30, 0.28, 0.31, 1.00, 0.71, 0.60, 0.37, 0.34, 0.33, 0.28, 0.32, 0.27],
        [0.24, 0.25, 0.26, 0.36, 0.35, 0.37, 0.71, 1.00, 0.66, 0.32, 0.28, 0.29, 0.31, 0.30, 0.26],
        [0.29, 0.30, 0.31, 0.27, 0.26, 0.28, 0.60, 0.66, 1.00, 0.33, 0.30, 0.31, 0.29, 0.27, 0.32],

        # ---- Superdim 4 ----
        [0.30, 0.27, 0.29, 0.33, 0.31, 0.34, 0.37, 0.32, 0.33, 1.00, 0.73, 0.62, 0.38, 0.31, 0.29],
        [0.39, 0.35, 0.33, 0.25, 0.27, 0.29, 0.34, 0.28, 0.30, 0.73, 1.00, 0.67, 0.35, 0.33, 0.32],
        [0.26, 0.28, 0.27, 0.37, 0.34, 0.35, 0.33, 0.29, 0.31, 0.62, 0.67, 1.00, 0.36, 0.30, 0.28],

        # ---- Superdim 5 ----
        [0.34, 0.37, 0.36, 0.30, 0.32, 0.31, 0.28, 0.31, 0.29, 0.38, 0.35, 0.36, 1.00, 0.69, 0.61],
        [0.28, 0.26, 0.25, 0.28, 0.29, 0.26, 0.32, 0.30, 0.27, 0.31, 0.33, 0.30, 0.69, 1.00, 0.73],
        [0.32, 0.33, 0.34, 0.32, 0.30, 0.33, 0.27, 0.26, 0.32, 0.29, 0.32, 0.28, 0.61, 0.73, 1.00]
    ])

    mu = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 0.3, 0.1, 0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.4, 0.1, 0.18, 0.3, 0.6, 0.2],
        [0.7, 0.4, 0.15, 0.55, 0.6, 0.3, 0.6, 0.5, 0.65, 0.6, 0.12, 0.25, 0.4, 0.6, 0.5]
    ])
    trait_names=[
        "Zuversicht","Richtung", "Antrieb",
        "Vertrauen","Nähe", "Konfliktfähigkeit",
        "Entwicklung","Inspiration", "Anerkennung",
        "Neugier", "Lernen", "Einsatz",
        "Verantwortung", "Fokus", "Konsequenz"
    ]

    survey_dataset = SurveyDataset(
        dimension_names=trait_names,
        base_trait_correlation_matrix=sigma_traits,
        means_dimensions=mu,
        n_waves=mu.shape[0],
        rho_time = 0.6,
        sigma_time = 1.0,
        var_shares_levels = (0.1, 0.5, 0.4),
        units_levels = (3, 100, 10),
    )

    df_wide, df_long, df_meta = survey_dataset.simulate()

    # Step 1: compute eigenvalues of the correlation matrix

    df_items = df_wide.iloc[:, 4:]




    # =====================================
    # 1. Scree Plot
    # =====================================
    corr = df_items.corr()
    eigvals, _ = np.linalg.eig(corr)
    eigvals = np.sort(eigvals)[::-1]

    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(eigvals)+1), eigvals, marker="o")
    plt.axhline(1, color="grey", linestyle="--", label="Kaiser (λ=1)")
    plt.title("Scree Plot")
    plt.xlabel("Factor")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =====================================
    # 2. Unrotated Factor Analysis
    # =====================================
    n_factors = 5
    fa_unrot = FactorAnalyzer(n_factors=n_factors, rotation=None, method="minres")
    fa_unrot.fit(df_items)

    loadings_unrot = pd.DataFrame(fa_unrot.loadings_,
                                  index=df_items.columns,
                                  columns=[f"Factor{i+1}" for i in range(n_factors)])

    print("\n=== Unrotated Factor Loadings ===")
    print(loadings_unrot.round(3))

    # =====================================
    # 3. Oblique Rotation (Oblimin)
    # =====================================
    fa_oblique = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="minres")
    fa_oblique.fit(df_items)

    loadings_oblique = pd.DataFrame(fa_oblique.loadings_,
                                    index=df_items.columns,
                                    columns=[f"Factor{i+1}" for i in range(n_factors)])

    print("\n=== Rotated (Oblimin) Factor Loadings ===")
    print(loadings_oblique.round(3))

    # Factor correlations (phi)
    phi = pd.DataFrame(fa_oblique.phi_,
                       index=[f"Factor{i+1}" for i in range(n_factors)],
                       columns=[f"Factor{i+1}" for i in range(n_factors)])

    print("\n=== Factor Correlations (Phi Matrix) ===")
    print(phi.round(3))

    # =====================================
    # 4. Structure Table (assign each item to strongest factor)
    # =====================================
    assignments = loadings_oblique.abs().idxmax(axis=1)
    structure = pd.concat([loadings_oblique.round(3),
                           assignments.rename("Highest Loading")],
                          axis=1)

    print("\n=== Rotated Structure Table (items assigned to strongest factor) ===")
    print(structure)

    # =====================================
    # 5. Optional Heatmap of Rotated Loadings
    # =====================================
    plt.figure(figsize=(8, len(df_items.columns)/2))
    sns.heatmap(loadings_oblique, annot=True, cmap="RdBu_r", center=0,
                yticklabels=df_items.columns, xticklabels=loadings_oblique.columns)
    plt.title("Rotated Factor Loadings (Oblimin)")
    plt.tight_layout()
    plt.show()
