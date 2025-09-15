from simulate_data.multitrait_multitime_distribution import MTMTDistribution
from simulate_data.multilevel_simulator import SymmetricMTMT
from simulate_data.simulate_irt import ItemBank
import numpy as np
import pandas as pd
from typing import Iterable
from tqdm import tqdm

class SurveyDataset:
    def __init__(
            self,
            dimension_names: Iterable[str],
            base_trait_correlation_matrix: np.array,
            means_dimensions: np.array,
            n_waves: int=1,
            rho_time=None,
            sigma_time=None,
            var_shares_levels=None,
            units_levels=None,
            level_names=("org", "team", "individual"),
            item_types="auto",
            items=None,

    ):

        self.dim_names = dimension_names
        self.base_trait_correlation_matrix = base_trait_correlation_matrix
        self.means_dimensions = means_dimensions

        if n_waves > 1:
            assert (rho_time is not None) and (sigma_time is not None)
            Sigma_time = MTMTDistribution.ar1_cov(3, rho=0.6, sigma=1.0)
        self.population_flat = MTMTDistribution(
            means_dimensions,
            Sigma_time,
            base_trait_correlation_matrix,
            trait_names=dimension_names
        )
        self.hierarchy = SymmetricMTMT(
            base=self.population_flat,
            level_names=level_names,
            var_shares=var_shares_levels,
            default_branching=units_levels,
        )

        if item_types == "auto":
            self.item_dict = {}
            # generate 5 GRM items per dimension
            for latent_dim in dimension_names:
                self.item_dict[latent_dim] = ItemBank(latent_dims=latent_dim)
        else:
            raise NotImplementedError




    def simulate(self, seed=None, data_form="both"):
        # simulate traits
        df_ind, df_meta = self.hierarchy.simulate(seed)
        # simulate items from traits
        df_wide = df_ind.pivot_table(index=["person_id", "level_0", "level_1","wave"],
                                       columns="trait",
                                       values="value").reset_index().rename_axis(None, axis=1)

        df_item_responses = df_wide[df_wide.columns[:df_wide.shape[1] - len(self.dim_names)]].copy()
        for dim in tqdm(self.dim_names):
            item_bank = self.item_dict[dim]
            resp_matrix = item_bank.simulate(df_wide[[dim]])
            item_names_for_dim = [f"{dim}_{i:02}" for i in range(item_bank.n_items)]
            df_item_responses[item_names_for_dim] = resp_matrix

        if data_form == "wide":
            return df_item_responses, df_meta
        elif data_form == "long":
            return df_item_responses.melt(id_vars=["person_id", "level_0", "level_1","wave"]), df_meta
        elif data_form == "both":
            return (
                df_item_responses,
                df_item_responses.melt(id_vars=["person_id", "level_0", "level_1","wave"]),
                df_meta
            )
        else:
            raise ValueError("data_form must be one of 'wide', 'long', or 'both'")

