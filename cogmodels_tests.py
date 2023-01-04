import pandas as pd
import os
from cogmodels_base import *


def test_main():
    import pandas as pd
    import os
    from cogmodels_base import *
    cache_folder = r"D:\U19\data\Probswitch\caching"
    data_file = os.path.join(cache_folder, "bsd_model_data_sample.pq")
    data = pd.read_parquet(data_file)
    pcm = PCModel()
    # TODO: handle miss decisions
    data = pcm.fit_marginal(data, K=pcm.fixed_params['K_marginal'])
    params_df = pcm.create_params(data)
    data_sim = pcm.sim(data, params_df)
    # data_sim.to_csv(os.path.join(cache_folder, "bsd_sim_data_sample.csv"))
    pcm = pcm.fit(data)
    data_sim_opt = pcm.sim(data, pcm.fitted_params)
    data_sim.to_csv(os.path.join(cache_folder, "bsd_simopt_data_sample_v0.csv"))
    pcm.fitted_params.to_csv(os.path.join(cache_folder, "bsd_simopt_params_v0.csv"))

