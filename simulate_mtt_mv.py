from simulate_data.multilevel_simulator import MTMTDistribution, SymmetricMTMT
from simulate_data.simulate_irt import GradedResponseItem, ItemBank
import numpy as np
from matplotlib import pyplot as plt
import numpy as np

Sigma_traits = np.array([
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
Sigma_time = MTMTDistribution.ar1_cov(3, rho=0.6, sigma=1.0)
pop = MTMTDistribution(
    mu,
    Sigma_time,
    Sigma_traits,
    trait_names=[
        "Zuversicht","Richtung", "Antrieb",
        "Vertrauen","Nähe", "Konfliktfähigkeit",
        "Entwicklung","Inspiration", "Anerkennung",
        "Neugier", "Lernen", "Einsatz",
        "Verantwortung", "Fokus", "Konsequenz"
    ])
#pop.plot_population_time_profiles()
df_sim = pop.simulate(n=500, as_df=True)
pop.plot_population_time_profiles(empirical_data=df_sim)
plt.show()

level_names = ["Org", "Team", "Indiv"]
var_shares = [0.1, 0.2, 0.70]
default_branching = [3, 100, 10]

sim = SymmetricMTMT(
    base=pop,
    level_names=level_names,
    var_shares=var_shares,
    default_branching=default_branching,
)

# Step 1: simulate latent long data
df_indiv, df_meta = sim.simulate(seed=2026)

# Step 2: reshape into wide DataFrame (outside ItemBank)
df_wide = df_indiv.pivot_table(index=["person_id","wave"],
                               columns="trait",
                               values="value").reset_index()

# Step 3: create ItemBank
bank = ItemBank(item_type="GRM", n_items=5, latent_dims="Zuversicht")

# Step 4: pass wide latent scores to simulate
resp_matrix = bank.simulate(df_wide[["Zuversicht"]])

print(resp_matrix.shape)   # (n_persons * waves, n_items)





