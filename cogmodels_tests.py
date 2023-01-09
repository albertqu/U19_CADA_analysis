from multiprocessing import Pool, current_process
import pandas as pd
import os
from cogmodels_base import *
from tqdm import tqdm
import psutil

np.random.seed(230)


def test_main():
    # import pandas as pd
    # import os
    # from cogmodels_base import *
    # np.random.seed(230)
    cache_folder = r"D:\U19\data\Probswitch\caching"
    data_arg = "sample" #"full_20230105"
    data_file = os.path.join(cache_folder, f"bsd_model_data_{data_arg}.pq")
    data = pd.read_parquet(data_file)
    data['ID'] = data['Subject']
    pcm = PCModel()
    # TODO: handle miss decisions
    data = pcm.fit_marginal(data)
    # params_df = pcm.create_params(data)
    # data_sim = pcm.sim(data, params_df)
    # data_sim.to_csv(os.path.join(cache_folder, "bsd_sim_data_sample.csv"))
    pcm = pcm.fit(data)
    data_sim_opt = pcm.sim(data, pcm.fitted_params)
    data_sim_opt.to_csv(os.path.join(cache_folder, f"bsd_simopt_data_{data_arg}_v5_test.csv"))
    pcm.fitted_params.to_csv(os.path.join(cache_folder, f"bsd_simopt_params_{data_arg}_v5_test.csv"))
    # v1: optimization without bounds, wrong MLE
    # v2: set bounds, results came biases towards decision 1 (majority decision), wrong MLE
    # v3: fixed p_rew/eps, wrong MLE
    # v4: fix MLE equation fixed p_rew/eps
    # v5: fix MLE equation variable p_rew/eps


def fit_model_per_subject_helper(model):
    # data_subj, data for single subject

    def helper(data_subj):
        pcm = model()
        pcm = pcm.fit(data_subj)
        data_sim_opt = pcm.sim(data_subj, pcm.fitted_params)
        fparams = pcm.fitted_params.reset_index(drop=True)
        fparams['aic'] = pcm.summary['aic']
        fparams['bic'] = pcm.summary['bic']
        return data_sim_opt, fparams
    return helper


def fit_model_per_subject(data_subj, model):
    # data_subj, data for single subject
    pcm = model()
    pcm = pcm.fit(data_subj)
    data_sim_opt = pcm.sim(data_subj, pcm.fitted_params)
    fparams = pcm.fitted_params.reset_index(drop=True)
    fparams['aic'] = pcm.summary['aic']
    fparams['bic'] = pcm.summary['bic']
    return data_sim_opt, fparams


def test_mp_multiple_animals(model, model_arg):
    cache_folder = r"D:\U19\data\Probswitch\caching"
    data_arg = "sample"#"full_20230105"
    data_file = os.path.join(cache_folder, f"bsd_model_data_{data_arg}.pq")
    data = pd.read_parquet(data_file)
    data['ID'] = data['Subject']
    all_data_sim, all_params = [], []

    # model_arg = 'PC_fixpswgam'
    with Pool(processes=psutil.cpu_count(logical=False)) as pool:

        results = [pool.apply_async(fit_model_per_subject,
                                    args=(data[data['Subject'] == subj].reset_index(drop=True), model)) for subj in data['Subject'].unique()]
        for r in tqdm(results):
            id_sim, id_fp = r.get()
            all_data_sim.append(id_sim)
            all_params.append(id_fp)
    version = 'v7_test2'
    data_sim_opt = pd.concat(all_data_sim, axis=0).sort_values(['Subject', 'Session', 'Trial'])
    data_sim_opt.to_csv(os.path.join(cache_folder, f"bsd_simopt_data_{data_arg}_{model_arg}_{version}.csv"), index=False)
    pd.concat(all_params, axis=0).sort_values('ID').to_csv(os.path.join(cache_folder,
                                                                        f"bsd_simopt_params_{data_arg}_{model_arg}_{version}.csv"), index=False)
    # v7: multiprocessing fit multiple animals


def test_multiple_animals():
    cache_folder = r"D:\U19\data\Probswitch\caching"
    data_arg = "sample" #"full_20230105"
    data_file = os.path.join(cache_folder, f"bsd_model_data_{data_arg}.pq")
    data = pd.read_parquet(data_file)
    data['ID'] = data['Subject']
    all_data_sim, all_params = [], []
    model_arg = 'PC_fixpswgam'
    fit_by_subject = fit_model_per_subject_helper(PCModel_fixpswgam)

    for subj in tqdm(data['Subject'].unique()):
        id_sim, id_fp = fit_by_subject(data[data['Subject'] == subj].reset_index(drop=True))
        all_data_sim.append(id_sim)
        all_params.append(id_fp)
    data_sim_opt = pd.concat(all_data_sim, axis=0).sort_values(['Subject', 'Session', 'Trial'])
    data_sim_opt.to_csv(os.path.join(cache_folder, f"bsd_simopt_data_{data_arg}_{model_arg}_v6.csv"), index=False)
    pd.concat(all_params, axis=0).sort_values('ID').to_csv(os.path.join(cache_folder,
                                                                        f"bsd_simopt_params_{data_arg}_{model_arg}_v6.csv"), index=False)
    # v6: fit multiple animals


if __name__ == '__main__':
    test_mp_multiple_animals(PCModel, 'PC')
    test_mp_multiple_animals(PCModel_fixpswgam, 'PC_fixpswgam')





