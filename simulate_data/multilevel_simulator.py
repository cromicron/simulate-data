from simulate_data.multitrait_multitime_distribution import MTMTDistribution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


# -----------------------------
# Boss-ready plotting helpers
# -----------------------------
def _ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)

def plot_variance_shares(var_shares, level_names, outdir="plots"):
    _ensure_outdir(outdir)
    shares = np.array(var_shares, dtype=float)
    shares = shares / shares.sum()
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar([0], [shares[0]])
    bottom = shares[0]
    for s in shares[1:]:
        ax.bar([0], [s], bottom=bottom)
        bottom += s
    ax.set_xticks([0]); ax.set_xticklabels(["Total variance = 1"])
    ax.set_ylabel("Share")
    ax.set_title("Variance shares by level")
    # annotate
    y0=0
    for name, s in zip(level_names, shares):
        ax.text(0, y0 + s/2, f"{name}: {s:.2f}", ha="center", va="center")
        y0 += s
    fig.tight_layout()
    path = os.path.join(outdir, "01_variance_shares.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_group_center_heatmap(df_meta, level_names, base_traits, which_level=0, outdir="plots"):
    """
    Heatmap of group centers at a level (e.g., level 0 = Orgs, level 1 = Teams).
    Aggregates means across waves (average over waves) for each trait.
    """
    _ensure_outdir(outdir)
    lvl_name = level_names[which_level]
    # pick rows of that level
    meta = df_meta[df_meta["level_index"] == which_level].copy()
    if meta.empty:
        return None
    # find all columns like mean_w{t}_{trait}
    mean_cols = [c for c in meta.columns if c.startswith("mean_w")]
    # average across waves per trait
    trait_avgs = {}
    for trait in base_traits:
        cols = [c for c in mean_cols if c.endswith(f"_{trait}")]
        trait_avgs[trait] = meta[cols].mean(axis=1).values
    mat = np.vstack([trait_avgs[t] for t in base_traits])  # traits x groups
    fig, ax = plt.subplots(figsize=(max(6, mat.shape[1]*0.4), 3+0.3*len(base_traits)))
    im = ax.imshow(mat, aspect="auto")
    ax.set_yticks(range(len(base_traits))); ax.set_yticklabels(base_traits)
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(meta["path"].tolist(), rotation=90)
    ax.set_title(f"{lvl_name} centers (avg over waves)")
    fig.colorbar(im, ax=ax, label="Mean")
    fig.tight_layout()
    path = os.path.join(outdir, f"02_heatmap_{lvl_name}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_sample_trajectories(df_indiv, trait, n_people=20, outdir="plots", seed=0):
    _ensure_outdir(outdir)
    rng = np.random.default_rng(seed)
    candidates = df_indiv["person_id"].drop_duplicates().tolist()
    pick = candidates if len(candidates) <= n_people else list(rng.choice(candidates, size=n_people, replace=False))
    fig, ax = plt.subplots(figsize=(6,4))
    for pid in pick:
        sub = df_indiv[(df_indiv["person_id"]==pid) & (df_indiv["trait"]==trait)].sort_values("wave")
        ax.plot(sub["wave"].values+1, sub["value"].values, alpha=0.6)
    ax.set_xlabel("Wave"); ax.set_ylabel(trait); ax.set_title(f"Sample trajectories ({trait})")
    ax.set_xticks(sorted(df_indiv["wave"].unique()+1))
    fig.tight_layout()
    path = os.path.join(outdir, f"03_trajectories_{trait}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_histograms_by_wave(df_indiv, trait, outdir="plots"):
    _ensure_outdir(outdir)
    waves = sorted(df_indiv["wave"].unique())
    for w in waves:
        fig, ax = plt.subplots(figsize=(5,3))
        vals = df_indiv[(df_indiv["trait"]==trait) & (df_indiv["wave"]==w)]["value"].values
        ax.hist(vals, bins=30, density=True)
        ax.set_title(f"{trait} distribution — Wave {w+1}")
        ax.set_xlabel(trait); ax.set_ylabel("Density")
        fig.tight_layout()
        path = os.path.join(outdir, f"04_hist_{trait}_wave{w+1}.png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    return True

def plot_between_within_variance(df_indiv, level_cols, trait, wave, outdir="plots"):
    """
    Quick ANOVA-style variance split: Between-groups (variance of group means) vs Within (mean of within-group variances).
    level_cols: list of columns defining the grouping (e.g., ["level_0"] for Orgs, ["level_0","level_1"] for Teams)
    """
    _ensure_outdir(outdir)
    sub = df_indiv[(df_indiv["trait"]==trait) & (df_indiv["wave"]==wave)].copy()
    if sub.empty: return None
    # group id
    sub["grp"] = sub[level_cols].astype(str).agg(".".join, axis=1)
    # overall variance
    total_var = sub["value"].var(ddof=1)
    # between: var of group means
    means = sub.groupby("grp")["value"].mean()
    between_var = means.var(ddof=1)
    # within: weighted average of within variances
    within_vars = sub.groupby("grp")["value"].var(ddof=1)
    counts = sub.groupby("grp")["value"].count()
    within_var = np.average(within_vars.fillna(0).values, weights=counts.values)  # NA→0 for singletons
    # sanity: total ~ between + within (not exact if unbalanced, but close)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(["Between","Within","Total"], [between_var, within_var, total_var])
    ax.set_title(f"Variance split @ {'.'.join(level_cols)}\n{trait}, Wave {wave+1}")
    ax.set_ylabel("Variance")
    fig.tight_layout()
    path = os.path.join(outdir, f"05_varsplit_{'-'.join(level_cols)}_{trait}_w{wave+1}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_team_sizes(df_indiv, outdir="plots"):
    """
    Bar charts of team sizes by org (assumes Org in level_0, Team in level_1 if present).
    Falls back gracefully if fewer levels exist.
    """
    _ensure_outdir(outdir)
    levels = [c for c in df_indiv.columns if c.startswith("level_")]
    if len(levels) < 1: return None
    # unique people count per group
    if len(levels) >= 2:
        grp = df_indiv.groupby([levels[0], levels[1]])["person_id"].nunique().reset_index(name="size")
        orgs = grp[levels[0]].unique()
        for org in orgs:
            fig, ax = plt.subplots(figsize=(max(4, 0.5*grp[grp[levels[0]]==org].shape[0]), 3))
            sub = grp[grp[levels[0]]==org]
            ax.bar(sub[levels[1]].astype(str), sub["size"].values)
            ax.set_title(f"Team sizes in {org}")
            ax.set_ylabel("People")
            ax.set_xticklabels(sub[levels[1]].astype(str), rotation=45, ha="right")
            fig.tight_layout()
            path = os.path.join(outdir, f"06_sizes_{org}.png")
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
        return True
    else:
        # only one level → just show sizes
        grp = df_indiv.groupby(levels[0])["person_id"].nunique().reset_index(name="size")
        fig, ax = plt.subplots(figsize=(max(4, 0.5*grp.shape[0]), 3))
        ax.bar(grp[levels[0]].astype(str), grp["size"].values)
        ax.set_title(f"Group sizes by {levels[0]}")
        ax.set_ylabel("People")
        ax.set_xticklabels(grp[levels[0]].astype(str), rotation=45, ha="right")
        fig.tight_layout()
        path = os.path.join(outdir, f"06_sizes_{levels[0]}.png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return True

# -----------------------------
# Example driver (adapt for your data)
# -----------------------------
def make_boss_plots(df_indiv, df_meta, level_names, var_shares, outdir="plots",
                    trait_for_lines=None, trait_for_hists=None, wave_for_varsplit=0):
    """
    Build a concise deck of plots for the boss.
    """
    # pick sensible defaults
    all_traits = list(df_indiv["trait"].unique())
    trait_for_lines = trait_for_lines or all_traits[0]
    trait_for_hists = trait_for_hists or all_traits[0]

    paths = []
    paths.append(plot_variance_shares(var_shares, level_names, outdir))

    # Heatmaps for first two levels if present
    if "level_index" in df_meta.columns:
        if (df_meta["level_index"]==0).any():
            paths.append(plot_group_center_heatmap(df_meta, level_names, all_traits, which_level=0, outdir=outdir))
        if (df_meta["level_index"]==1).any():
            paths.append(plot_group_center_heatmap(df_meta, level_names, all_traits, which_level=1, outdir=outdir))

    paths.append(plot_sample_trajectories(df_indiv, trait_for_lines, n_people=20, outdir=outdir))
    plot_histograms_by_wave(df_indiv, trait_for_hists, outdir=outdir)

    # Variance splits at Org and Team (if available)
    level_cols = [c for c in df_indiv.columns if c.startswith("level_")]
    if len(level_cols) >= 1:
        paths.append(plot_between_within_variance(df_indiv, [level_cols[0]], trait_for_hists, wave_for_varsplit, outdir))
    if len(level_cols) >= 2:
        paths.append(plot_between_within_variance(df_indiv, [level_cols[0], level_cols[1]], trait_for_hists, wave_for_varsplit, outdir))

    plot_team_sizes(df_indiv, outdir)

    # Return list of generated files (None entries may appear if a level missing)
    return [p for p in paths if p]



class SymmetricMTMT:
    """
    Symmetric (balanced) multilevel simulator with easy ragged overrides.

    Uses an existing MTMTDistribution instance (base.mu, base.Lt, base.Lp, base.trait_names).

    Parameters
    ----------
    base : MTMTDistribution
        Your existing distribution object (unchanged).
    level_names : list[str]
        Names for each level from top to bottom (e.g., ["Org", "Team", "Indiv"]).
        The last name is the individual level.
    var_shares : list[float]
        Variance shares per level (same length as level_names). Must sum to 1 (we normalize defensively).
        The SD multiplier at level ℓ is s_ℓ = sqrt(var_shares[ℓ]).
    default_branching : list[int]
        Symmetric children counts per level, length == len(level_names).
        Example: for ["Org","Team","Indiv"], [2, 3, 10] means:
          - 2 orgs (children of μ),
          - each org has 3 teams,
          - each team has 10 individuals.
    overrides : dict[str, int], optional
        Ragged size overrides keyed by the **parent path** (e.g., "Org2", "Org2.Team1").
        Value is the **number of children** that specific parent should have at its next level.
        You only specify the exceptions; everything else stays symmetric.

        Examples:
          overrides = {
            "Org1": 1,            # Org1 has 1 team (others have default 3)
            "Org2.Team2": 20,     # Team2 inside Org2 has 20 individuals (others default 10)
          }

    Notes
    -----
    - Each level ℓ adds an independent MVN component with SD multiplier s_ℓ relative to base Σ.
      Group centers and individuals are generated by summing these components along the path.
    - Output:
        df_indiv: long table rows = person × wave × trait
        df_meta : group centers for all non-individual nodes with flattened means and sd used
    """

    def __init__(self, base, level_names, var_shares, default_branching, overrides=None, seed=123):
        self.base = base
        self.level_names = list(level_names)
        self.var_shares = np.asarray(var_shares, dtype=float)
        # normalize shares just in case
        if not np.isclose(self.var_shares.sum(), 1.0):
            self.var_shares = self.var_shares / self.var_shares.sum()
        self.sd_scales = np.sqrt(self.var_shares)  # SD multipliers per level
        self.default_branching = list(default_branching)
        if len(self.default_branching) != len(self.level_names):
            raise ValueError("default_branching must have same length as level_names "
                             "(children count per level, including last for individuals).")
        self.overrides = overrides or {}
        self.rng = np.random.default_rng(seed)

        if len(self.var_shares) != len(self.level_names):
            raise ValueError("var_shares must have same length as level_names.")

    # ---------- internals ----------
    def _draw_increments(self, sd_scale, n):
        """Draw n samples from MVN(0, sd_scale^2 * Σ); returns (n, T, P)."""
        if sd_scale == 0.0:
            return np.zeros((n, self.base.T, self.base.P), dtype=float)
        Z = self.base.rng.standard_normal((n, self.base.T, self.base.P))
        X = Z @ self.base.Lp.T
        X = np.einsum("tu,ntp->nup", self.base.Lt, X)
        return sd_scale * X

    def _children_count(self, parent_path, level_index):
        """
        How many children should THIS parent have at THIS level_index (i.e., how many nodes at next level)?
        - Check overrides dict first (exact match on parent_path).
        - Otherwise fall back to default_branching[level_index].
        """
        return int(self.overrides.get(parent_path, self.default_branching[level_index]))

    def _flatten_center(self, center):
        """Return dict of flattened mean columns mean_w{t}_{trait} -> value."""
        T, P = self.base.T, self.base.P
        return {
            f"mean_w{t}_{self.base.trait_names[p]}": center[t, p]
            for t in range(T) for p in range(P)
        }

    # ---------- simulate ----------
    def simulate(self, seed=None):
        """
        Returns
        -------
        df_indiv : DataFrame
            Columns: level_0, level_1, ..., <last level name>, person_id, wave, trait, value
        df_meta : DataFrame
            Group centers per non-individual node with: level_index, level_name, path, sd_scale, flattened means
        """
        if seed is not None:
            self.base.reseed(seed)

        T, P = self.base.T, self.base.P
        L = len(self.level_names)

        # Start from the global mean μ (root is not a sampled node)
        parents = [{"center": self.base.mu.copy(), "path": []}]  # virtual root

        meta_rows = []
        indiv_rows = []
        person_id = 0  # global running person id

        # For each level (including last = individuals)
        for lvl in range(L):
            level_name = self.level_names[lvl]
            sd = float(self.sd_scales[lvl])

            children_nodes = []
            # First, compute total number of children so we can draw in one batch (fast)
            total_children = 0
            for parent in parents:
                parent_path = ".".join(parent["path"]) if parent["path"] else ""
                total_children += self._children_count(parent_path, lvl)

            # Draw all increments for this level in one go
            inc_all = self._draw_increments(sd, total_children)
            k = 0  # pointer into inc_all

            for parent in parents:
                parent_path = ".".join(parent["path"]) if parent["path"] else ""
                n_children = self._children_count(parent_path, lvl)

                if lvl < L - 1:
                    # Group level: create child group centers
                    for j in range(n_children):
                        child_center = parent["center"] + inc_all[k]; k += 1
                        child_label = f"{level_name}{j+1}"
                        child_path = parent["path"] + [child_label]
                        children_nodes.append({"center": child_center, "path": child_path})

                        # record group center meta
                        meta_rows.append({
                            "level_index": lvl,
                            "level_name": level_name,
                            "path": ".".join(child_path),
                            "sd_scale": sd,
                            **self._flatten_center(child_center),
                        })
                else:
                    # Individual level: draw individuals around parent center
                    for j in range(n_children):
                        indiv_val = parent["center"] + inc_all[k]; k += 1
                        indiv_label = f"{level_name}{j+1}"
                        indiv_path = parent["path"]  # up to previous level

                        # long rows: one per wave×trait
                        for t in range(T):
                            for p, trait in enumerate(self.base.trait_names):
                                indiv_rows.append({
                                    **{f"level_{h}": name for h, name in enumerate(indiv_path)},
                                    level_name: indiv_label,
                                    "person_id": person_id,
                                    "wave": t,
                                    "trait": trait,
                                    "value": indiv_val[t, p],
                                })
                        person_id += 1

            parents = children_nodes  # descend

        df_indiv = pd.DataFrame(indiv_rows)
        df_meta = pd.DataFrame(meta_rows)
        return df_indiv, df_meta

    @staticmethod
    def plot_individual_profiles(df_indiv, n_indiv=10, traits=None, facet_by_trait=False, col_wrap=3):
        """
        Plot a few individual trajectories across waves.

        Args:
            df_indiv : DataFrame (from simulate)
            n_indiv : number of individuals to sample
            traits : list of traits to include (default = all)
            facet_by_trait : if True, plot separate panels per trait
            col_wrap : number of columns when facetting
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Filter traits if requested
        if traits is not None:
            data = df_indiv[df_indiv["trait"].isin(traits)]
        else:
            data = df_indiv

        # Subsample individuals
        all_ids = data["person_id"].unique()
        if n_indiv < len(all_ids):
            sample_ids = pd.Series(all_ids).sample(n_indiv)
            data = data[data["person_id"].isin(sample_ids)]

        if facet_by_trait:
            g = sns.relplot(
                data=data, kind="line",
                x="wave", y="value",
                hue="person_id", units="person_id",
                col="trait", col_wrap=col_wrap,
                estimator=None, lw=1, alpha=0.7, palette="tab20"
            )
            g.set_titles("Trait: {col_name}")
            g.set_axis_labels("Wave", "Value")
            g.tight_layout()
        else:
            plt.figure(figsize=(8, 5))
            sns.lineplot(
                data=data, x="wave", y="value",
                hue="person_id", style="trait",
                estimator=None, units="person_id", lw=1, alpha=0.7
            )
            plt.title("Sample of Individual Trajectories")
            plt.xlabel("Wave")
            plt.ylabel("Value")
            plt.tight_layout()

    @staticmethod
    def plot_trait_means(
            df_indiv,
            group_level=None,
            ci=95,
            max_groups=10,
            random_state=123):
        """
        Plot average trajectories per trait (optionally grouped by a higher-level node).

        Args:
            df_indiv : DataFrame
            group_level : str or None
                Column to group by (e.g. "level_0", "level_1").
            ci : int or None
                Confidence interval for seaborn lineplot.
            max_groups : int
                Maximum number of groups to plot (if more exist, sample).
            random_state : int
                Seed for reproducible sampling of groups.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 5))

        if group_level is None:
            sns.lineplot(data=df_indiv, x="wave", y="value", hue="trait", ci=ci)
        else:
            groups = df_indiv[group_level].dropna().unique()
            if len(groups) > max_groups:
                groups = pd.Series(groups).sample(max_groups, random_state=random_state)
                df_plot = df_indiv[df_indiv[group_level].isin(groups)]
            else:
                df_plot = df_indiv

            sns.lineplot(
                data=df_plot, x="wave", y="value",
                hue="trait", style=group_level, ci=ci
            )

        plt.title("Mean Trajectories")
        plt.tight_layout()

    @staticmethod
    def plot_group_centers(df_meta, level_name):
        """
        Plot group centers from df_meta for a given level.
        """
        centers = df_meta[df_meta["level_name"] == level_name]
        traits = [c for c in centers.columns if c not in ["level_index", "level_name", "path", "sd_scale"]]
        plt.figure(figsize=(8, 5))
        for _, row in centers.iterrows():
            vals = row[traits].values
            plt.plot(vals, label=row["path"])
        plt.xticks(range(len(traits)), traits, rotation=45)
        plt.title(f"Group Centers at {level_name}")
        plt.legend(fontsize="x-small", ncol=2)
        plt.tight_layout()

    @staticmethod
    def plot_distribution(df_indiv, trait, wave=None):
        """
        Histogram / KDE of a given trait (optionally at one wave).
        """
        data = df_indiv[df_indiv["trait"] == trait]
        if wave is not None:
            data = data[data["wave"] == wave]
        plt.figure(figsize=(6, 4))
        sns.histplot(data["value"], kde=True, color="tab:blue")
        title = f"Distribution of {trait}"
        if wave is not None:
            title += f" at wave {wave}"
        plt.title(title)
        plt.tight_layout()




if __name__ == "__main__":


    # ---- base population ----
    Sigma_traits = np.array([
        [1.00, 0.14, 0.22, -0.34, 0.20],
        [0.14, 1.00, 0.28, -0.29, 0.15],
        [0.22, 0.28, 1.00, -0.30, -0.02],
        [-0.34, -0.29, -0.30, 1.00, -0.06],
        [0.20, 0.15, -0.02, -0.06, 1.00],
    ])
    mu = np.array([[0, 0, 0, 0, 0], [0.8, 0.6, 0.2, 0.1, 0], [1.1, 0.9, 0.3, 0.2, 0.1]])
    Sigma_time = MTMTDistribution.ar1_cov(3, rho=0.6, sigma=1.0)

    pop = MTMTDistribution(mu, Sigma_time, Sigma_traits,
                           trait_names=["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism",
                                        "Openness"])

    # ---- build hierarchy ----
    # 3 levels: Org, Team, Indiv
    level_names = ["Org", "Team", "Indiv"]
    var_shares = [0.05, 0.15, 0.80]  # sums to 1 → Org 5%, Team 15%, Indiv 80%
    default_branching = [3, 100, 10]  # 2 orgs; each org 3 teams; each team 10 people

    # Optional ragged overrides: only specify exceptions
    overrides = {
        "Org1": 1,  # Org1 has 1 team (others have 3)
        "Org2.Team2": 20,  # Team2 inside Org2 has 20 people (others have 10)
    }

    sim = SymmetricMTMT(
        base=pop,
        level_names=level_names,
        var_shares=var_shares,
        default_branching=default_branching,
        overrides=overrides,  # omit or {} for fully symmetric
        seed=123
    )

    df_indiv, df_meta = sim.simulate(seed=2025)

    print(df_indiv.head())
    print(df_meta.head())

    # After you simulate:
    # df_indiv, df_meta = sim.simulate(seed=2025)

    level_names = ["Org", "Team", "Indiv"]  # your fixed levels

    if False:
        files = make_boss_plots(df_indiv, df_meta, level_names, var_shares, outdir="boss_plots",
                                trait_for_lines="Conscientiousness",
                                trait_for_hists="Conscientiousness",
                                wave_for_varsplit=0)

        print("Saved figures:")
        for f in files:
            if f: print(" -", f)




