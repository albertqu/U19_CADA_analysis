from multiprocessing import Pool, current_process

import numpy as np
import pandas as pd
import os
from cogmodels_base_old import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import psutil

np.random.seed(230)


def test_model_recovery():
    cache_folder = r"D:\U19\data\Probswitch\caching"
    data_arg = "full_20230105"
    data_file = os.path.join(cache_folder, f"bsd_model_data_{data_arg}.pq")
    data = pd.read_parquet(data_file)
    data["ID"] = data["Subject"]
    model, param_file = (
        BIModel,
        os.path.join(cache_folder, f"bsd_simopt_params_{data_arg}_BI_v12.csv"),
    )
    params_df = pd.read_csv(param_file)
    md = model()
    em_data = md.emulate(data, params_df)


def test_model_generate():
    cache_folder = r"D:\U19\data\Probswitch\caching"
    data_arg = "full_20230105"
    data_file = os.path.join(cache_folder, f"bsd_model_data_{data_arg}.pq")
    data = pd.read_parquet(data_file)
    data["ID"] = data["Subject"]
    model_arg = "BI"
    model = BIModel
    param_file = os.path.join(
        cache_folder, f"bsd_simopt_params_{data_arg}_{model_arg}_v12.csv"
    )
    params_df = pd.read_csv(param_file)
    params_df["n_trial"] = 1000
    params_df["n_session"] = 5
    md = model()
    md.fit_marginal(data)
    gen_data = md.generate(params_df.loc[[0]])
    gen_data.to_csv(
        os.path.join(cache_folder, f"bsd_generated_{data_arg}_{model_arg}_v12.csv"),
        index=False,
    )


""" ****************************************
******* Replication to Eckstein 2022 *******
**************************************** """


# next steps test RLCF
def test_model_eckstein2022_RLCF():
    cache_folder = r"D:\U19\data\Probswitch\caching"
    model = RLCF()
    model_arg = "RLCF"
    param_ranges = {
        "st": (0.13, 0.2),
        "beta": (2.5, 2.8),
        "a_neg": (0.65, 0.7),
        "a_pos": (0.71, 0.75),
    }
    np.random.seed(230)

    test_modelgen_mp(
        model,
        model_arg,
        "eckstein2022",
        param_ranges,
        niters=1000,
        save_folder=cache_folder,
    )


def test_model_gen_eckstein2022():
    cache_folder = r"D:\U19\data\Probswitch\caching"
    model = BIModel()
    model_arg = "BI"
    param_ranges = {
        "st": (0.07, 0.1),
        "beta": (3.05, 3.3),
        "p_rew": (0.915, 0.937),
        "sw": (0.142, 0.152),
    }
    np.random.seed(230)
    test_modelgen_mp(
        model,
        model_arg,
        "eckstein2022",
        param_ranges,
        niters=1000,
        save_folder=cache_folder,
    )


def test_modelgen_mp(
    model, model_arg, gen_arg, param_ranges, niters=1000, save_folder=None
):
    if save_folder is None:
        cache_folder = r"D:\U19\data\Probswitch\caching"
    else:
        cache_folder = save_folder
    params = {
        k: np.random.random(niters) * (v[1] - v[0]) + v[0]
        for k, v in param_ranges.items()
    }
    params_df = pd.DataFrame(params)
    params_df["ID"] = [f"GEN{i:03d}" for i in range(1, niters + 1)]
    params_df["n_trial"] = 1000
    params_df["n_session"] = 2  # change to 1
    all_gen = []
    with Pool(processes=psutil.cpu_count(logical=False)) as pool:
        result = pool.map_async(
            model.generate,
            [params_df.loc[[i]] for i in range(len(params_df))],
            chunksize=125,
        )

        for gen_data in tqdm(result.get()):
            all_gen.append(gen_data)
    data = pd.concat(all_gen, axis=0).sort_values(["ID", "Session", "Trial"])
    data.to_csv(
        os.path.join(cache_folder, f"gendata_{model_arg}_{gen_arg}.csv"), index=False
    )


def visualize_model_params_single_session(model, params):
    m = model()
    params_df = pd.DataFrame({p: [params[p]] for p in params})
    params_df["ID"] = "GEN001"
    params_df["n_trial"] = 1000
    params_df["n_session"] = 1
    data = m.generate(params_df)
    animal_behavior_plot_eckstein(data, plot_folder=None, save_arg=None)


def generate_and_recover(model, params_x, method):
    gendata = model.generate(params_x)
    model = model.fit(gendata, method=method)
    fparams = model.fitted_params.reset_index(drop=True)
    fparams["aic"] = model.summary["aic"]
    fparams["bic"] = model.summary["bic"]
    print("done with", params_x["ID"].unique()[0])
    return gendata, fparams


def test_modelgen_recover_mp(
    model,
    model_arg,
    gen_arg,
    param_ranges,
    niters=1000,
    save_folder=None,
    method="trust-constr",
    ntrial=1000,
    nsess=2,
):
    if save_folder is None:
        cache_folder = r"D:\U19\data\Probswitch\caching"
    else:
        cache_folder = save_folder
    params = {
        k: np.random.random(niters) * (v[1] - v[0]) + v[0]
        for k, v in param_ranges.items()
    }
    params_df = pd.DataFrame(params)
    params_df["ID"] = [f"GEN{i:03d}" for i in range(1, niters + 1)]
    params_df["n_trial"] = ntrial
    params_df["n_session"] = nsess  # change to 1
    all_gen = []
    all_params = []
    with Pool(processes=psutil.cpu_count(logical=False)) as pool:
        result = pool.starmap_async(
            generate_and_recover,
            [(model(), params_df.loc[[i]], method) for i in range(len(params_df))],
            chunksize=125,
        )

        for gen_data, fparams in tqdm(result.get()):
            all_gen.append(gen_data)
            all_params.append(fparams)
    data = pd.concat(all_gen, axis=0).sort_values(["ID", "Session", "Trial"])
    params_y = pd.concat(all_params, axis=0).sort_values(["ID"])
    params_df.to_csv(
        os.path.join(cache_folder, f"genrec_{gen_arg}_{model_arg}_{method}_paramsx.csv")
    )
    params_y.to_csv(
        os.path.join(cache_folder, f"genrec_{gen_arg}_{model_arg}_{method}_paramsy.csv")
    )
    data.to_csv(
        os.path.join(cache_folder, f"genrec_{gen_arg}_{model_arg}_{method}_gendata.csv")
    )
    return data, params_df, params_y


def test_model_genrec_eckstein2022():
    model = BIModel
    model_arg = "BI"
    cache_folder = r"D:\U19\data\Probswitch\caching"
    param_ranges = {
        "st": (0.07, 0.1),
        "beta": (3.05, 3.3),
        "p_rew": (0.915, 0.937),
        "sw": (0.142, 0.152),
    }
    np.random.seed(230)
    data, params_x, params_y = test_modelgen_recover_mp(
        model,
        model_arg,
        "eckstein2022",
        param_ranges,
        niters=1000,
        save_folder=cache_folder,
        method="L-BFGS-B",
    )


def test_model_genrec_eckstein2022_BIfp():
    model = BIModel_fixp
    model_arg = "BIfp"
    cache_folder = r"D:\U19\data\Probswitch\caching"
    param_ranges = {"st": (0, 1), "beta": (1, 4), "sw": (0.01, 0.2)}
    np.random.seed(230)
    data, params_x, params_y = test_modelgen_recover_mp(
        model,
        model_arg,
        "eckstein2022_1s5ht",
        param_ranges,
        niters=1000,
        save_folder=cache_folder,
        method="L-BFGS-B",
    )


def test_model_genrec_eckstein2022_PCf():
    model = PCModel_fixpswgam
    model_arg = "PCf"
    cache_folder = r"D:\U19\data\Probswitch\caching"
    param_ranges = {"st": (0, 1), "beta": (1, 4)}
    np.random.seed(230)
    data, params_x, params_y = test_modelgen_recover_mp(
        model,
        model_arg,
        "eckstein2022_1s5ht",
        param_ranges,
        niters=1000,
        save_folder=cache_folder,
        method="L-BFGS-B",
    )


def test_model_genrec_eckstein2022_RLCF():
    model = RLCF
    model_arg = "RLCF"
    cache_folder = r"D:\U19\data\Probswitch\caching"
    param_ranges = {
        "st": (0.13, 0.2),
        "beta": (2.5, 2.8),
        "a_neg": (0.65, 0.7),
        "a_pos": (0.71, 0.75),
    }
    np.random.seed(230)
    data, params_x, params_y = test_modelgen_recover_mp(
        model,
        model_arg,
        "eckstein2022",
        param_ranges,
        niters=1000,
        save_folder=cache_folder,
        method="L-BFGS-B",
    )


def test_model_genrec_eckstein2022_RLst():
    model = RL_st
    model_arg = "RLst"
    cache_folder = r"D:\U19\data\Probswitch\caching"
    param_ranges = {"beta": (2, 4), "alpha": (0.5, 1), "st": (0, 1)}
    np.random.seed(230)
    data, params_x, params_y = test_modelgen_recover_mp(
        model,
        model_arg,
        "eckstein2022_1s5ht",
        param_ranges,
        niters=1000,
        save_folder=cache_folder,
        method="L-BFGS-B",
        ntrial=500,
        nsess=1,
    )


def test_model_genrec_eckstein2022_RL4p():
    model = RL_4p
    model_arg = "RL4p"
    cache_folder = r"D:\U19\data\Probswitch\caching"
    param_ranges = {
        "beta": (2, 4),
        "a_neg": (0.5, 0.9),
        "a_pos": (0.5, 0.9),
        "st": (0, 1),
    }
    np.random.seed(230)
    data, params_x, params_y = test_modelgen_recover_mp(
        model,
        model_arg,
        "eckstein2022_1s5ht",
        param_ranges,
        niters=1000,
        save_folder=cache_folder,
        method="L-BFGS-B",
        ntrial=500,
        nsess=1,
    )


def test_model_genrec_eckstein2022_RL():
    model = RL
    model_arg = "RL"
    cache_folder = r"D:\U19\data\Probswitch\caching"
    param_ranges = {"beta": (2, 4), "alpha": (0.5, 1)}
    np.random.seed(230)
    data, params_x, params_y = test_modelgen_recover_mp(
        model,
        model_arg,
        "eckstein2022_1s5ht",
        param_ranges,
        niters=1000,
        save_folder=cache_folder,
        method="L-BFGS-B",
        ntrial=500,
        nsess=1,
    )


def test_model_genrec_BSD(model):
    mdl = model()
    model_arg = str(mdl)
    cache_folder = r"D:\U19\data\Probswitch\caching"
    data_arg = "full_20230118"
    version = "v13"
    pfile = os.path.join(
        cache_folder, f"bsd_simopt_params_{data_arg}_{model_arg}_{version}.csv"
    )
    params = pd.read_csv(pfile)
    plist = list(mdl.param_dict.keys())
    param_ranges = {p: (params[p].min(), params[p].max()) for p in plist}
    np.random.seed(230)
    # data, params_x, params_y = test_modelgen_recover_mp(model, model_arg, f'{data_arg}_3s5ht', param_ranges,
    #                                                     niters=1000, save_folder=cache_folder, method='L-BFGS-B',
    #                                                     ntrial=500, nsess=3)
    data, params_x, params_y = test_modelgen_recover_mp(
        model,
        model_arg,
        f"{data_arg}_3s5ht_restruct",
        param_ranges,
        niters=500,
        save_folder=cache_folder,
        method="L-BFGS-B",
        ntrial=500,
        nsess=1,
    )


def viz_genrec_recovery(model, gen_arg):
    # viz_genrec_recovery(BIModel, 'full_20230118_3s5ht')
    mdl = model()
    model_arg = str(mdl)
    method = "L-BFGS-B"
    cache_folder = r"D:\U19\data\Probswitch\caching"
    params_x = pd.read_csv(
        os.path.join(cache_folder, f"genrec_{gen_arg}_{model_arg}_{method}_paramsx.csv")
    )
    params_y = pd.read_csv(
        os.path.join(cache_folder, f"genrec_{gen_arg}_{model_arg}_{method}_paramsy.csv")
    )

    plist = list(mdl.param_dict.keys())
    px = params_x.melt(
        id_vars="ID", value_vars=plist, var_name="param", value_name="truth"
    )
    py = params_y.melt(
        id_vars="ID", value_vars=plist, var_name="param", value_name="fitted"
    )
    df = px.merge(py, on=["ID", "param"])

    def drop_outlier(idf):
        xstd, xmean = np.std(idf["truth"]), np.mean(idf["truth"])
        ol_thres = 6
        ub, lb = xmean + ol_thres * xstd, xmean - ol_thres * xstd
        return idf[(idf["fitted"] <= ub) & (idf["fitted"] >= lb)]

    def print_ndropped(data, **kwargs):
        param = data["param"].unique()[0]
        oratio = 1 - len(data) / np.sum(df["param"] == param)
        ax = plt.gca()
        ax.set_title(f"{param}({oratio*100:.02f}% outlier)")

    sns.set_context("talk")
    # TODO: drop outliers by type
    g = sns.lmplot(
        data=df.groupby("param").apply(drop_outlier).reset_index(drop=True),
        x="truth",
        y="fitted",
        col="param",
        col_wrap=2,
        facet_kws={"sharey": False, "sharex": False},
        scatter_kws={"alpha": 0.3, "color": "r"},
    )
    g.map_dataframe(print_ndropped)
    g.fig.suptitle(f"Model Recovery {model_arg} {len(df.ID.unique())} Runs")
    sns.despine()
    plt.tight_layout()
    plt.show()
    return g


def viz_genrec_eckstein2022(model):
    return viz_genrec_recovery(model, "eckstein2022_1s5ht")


def gendata_behavior_plot(model_arg, gen_arg):
    cache_folder = r"D:\U19\data\Probswitch\caching"
    plot_folder = r"D:\U19\plots\Probswitch"
    data = pd.read_csv(os.path.join(cache_folder, f"gendata_{model_arg}_{gen_arg}.csv"))
    # problem at edge of block switch, check edge behavior
    return animal_behavior_plot_eckstein(
        data, plot_folder, f"{model_arg}_gendata_{gen_arg}"
    )


def bsd_animal_behavior_plot():
    cache_folder = r"D:\U19\data\Probswitch\caching"
    plot_folder = r"D:\U19\plots\Probswitch"
    data_arg = "full_20230118"
    data_file = os.path.join(cache_folder, f"bsd_model_data_{data_arg}.pq")
    data = pd.read_parquet(data_file)
    animal_behavior_plot_eckstein(data, plot_folder, save_arg="BSD_all_animal")


def animal_behavior_plot_eckstein(data, plot_folder=None, save_arg=None, title=None):
    # replicates plots in eckstein 2022
    data["correct"] = data["Decision"] == data["Target"]
    consec_stay = data["Decision"].shift(1) == data["Decision"].shift(2)
    after_start = data["Trial"] > 2
    data["stay%"] = 1 - data["Switch"]
    data.loc[data["Trial"] == 1, "stay%"] = 0
    r1back = data["Reward"].shift(1)
    r2back = data["Reward"].shift(2)
    uusel = (r1back == 0) & (r2back == 0)
    ursel = (r1back == 1) & (r2back == 0)
    rusel = (r1back == 0) & (r2back == 1)
    rrsel = (r1back == 1) & (r2back == 1)
    data.loc[uusel & consec_stay & after_start, "trialHistory"] = "--"
    data.loc[ursel & consec_stay & after_start, "trialHistory"] = "-+"
    data.loc[rusel & consec_stay & after_start, "trialHistory"] = "+-"
    data.loc[rrsel & consec_stay & after_start, "trialHistory"] = "++"
    data["trialHistory"] = pd.Categorical(
        data["trialHistory"], categories=["++", "-+", "+-", "--"], ordered=True
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    sns.lineplot(
        data=data[data["blockTrial"] <= 10].reset_index(drop=True),
        x="blockTrial",
        y="correct",
        errorbar="se",
        ax=axes[0],
    )
    sns.barplot(
        data=data.dropna(subset=["trialHistory"]).reset_index(drop=True),
        x="trialHistory",
        y="stay%",
        errorbar="se",
        ax=axes[1],
        hue_order=["++", "-+", "+-", "--"],
    )
    sns.despine()
    if title is not None:
        fig.suptitle(title)
    if plot_folder is not None:
        plt.savefig(os.path.join(plot_folder, f"eckstein2022_behavior_{save_arg}.png"))
    else:
        plt.show()


def replicate_eckstein2022_fit_params():
    cache_folder = r"D:\U19\data\Probswitch\caching"
    plot_folder = r"D:\U19\plots\Probswitch"
    data_arg = "eckstein2022_full"
    for model_arg in ["RLCF", "BI", "PC"]:
        version = "v12"
        params_df = pd.read_csv(
            os.path.join(
                cache_folder, f"bsd_simopt_params_{data_arg}_{model_arg}_{version}.csv"
            )
        )
        print(model_arg, params_df[["bic", "aic"]].describe())


def test_model_differentiability():
    pass


def test_main():
    # import pandas as pd
    # import os
    # from cogmodels_base import *
    # np.random.seed(230)
    cache_folder = r"D:\U19\data\Probswitch\caching"
    data_arg = "sample"  # "full_20230105"
    data_file = os.path.join(cache_folder, f"bsd_model_data_{data_arg}.pq")
    data = pd.read_parquet(data_file)
    data["ID"] = data["Subject"]
    # pcm = PCModel()
    pcm = BIModel()  # RLCF()
    # pcm = PCModel_fixpswgam() #PCModel()
    model_arg = "BI"  # 'RLCF'
    # TODO: handle miss decisions
    data = pcm.fit_marginal(data)
    # data.to_csv(os.path.join(cache_folder, 'bsd_sim_sample_RLCF.csv'))
    params_df = pcm.create_params(data)
    data_sim = pcm.sim(data, params_df)
    # data_sim.to_csv(os.path.join(cache_folder, "bsd_sim_data_sample.csv"))
    pcm = pcm.fit(data)
    data_sim_opt = pcm.sim(data, pcm.fitted_params)
    data_sim_opt.to_csv(
        os.path.join(cache_folder, f"bsd_simopt_data_{data_arg}_{model_arg}_v5.csv")
    )
    pcm.fitted_params.to_csv(
        os.path.join(cache_folder, f"bsd_simopt_params_{model_arg}_v5.csv")
    )
    # versions for PCModel
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
        fparams["aic"] = pcm.summary["aic"]
        fparams["bic"] = pcm.summary["bic"]
        return data_sim_opt, fparams

    return helper


def fit_model_per_subject(data_subj, model):
    # data_subj, data for single subject
    pcm = model()
    pcm = pcm.fit(data_subj)
    data_sim_opt = pcm.sim(data_subj, pcm.fitted_params)
    fparams = pcm.fitted_params.reset_index(drop=True)
    fparams["aic"] = pcm.summary["aic"]
    fparams["bic"] = pcm.summary["bic"]
    return data_sim_opt, fparams


def fit_LR_BSD():
    model_arg = "LR"
    cache_folder = r"D:\U19\data\Probswitch\caching"
    data_arg = "full_20230118"
    version = "v13"
    data_file = os.path.join(cache_folder, f"bsd_model_data_{data_arg}.pq")
    data = pd.read_parquet(data_file)
    data["ID"] = data["Subject"]
    all_data_sim = []
    all_params = []
    for id_i in data["ID"].unique():
        lrm = LR()
        id_sim = lrm.fitsim(data[data["Subject"] == id_i].reset_index(drop=True))
        id_fp = lrm.fitted_params.reset_index(drop=True)
        id_fp["aic"] = lrm.summary["aic"]
        id_fp["bic"] = lrm.summary["bic"]
        all_data_sim.append(id_sim)
        all_params.append(id_fp)
    data_sim_opt = pd.concat(all_data_sim, axis=0).sort_values(
        ["Subject", "Session", "Trial"]
    )
    data_sim_opt.to_csv(
        os.path.join(
            cache_folder, f"bsd_simopt_data_{data_arg}_{model_arg}_{version}.csv"
        ),
        index=False,
    )
    pd.concat(all_params, axis=0).sort_values("ID").to_csv(
        os.path.join(
            cache_folder, f"bsd_simopt_params_{data_arg}_{model_arg}_{version}.csv"
        ),
        index=False,
    )


def test_mp_multiple_animals(model):
    model_arg = str(model())
    cache_folder = r"D:\U19\data\Probswitch\caching"
    data_arg = "full_20230118"
    # data_arg = 'eckstein2022_full'
    data_file = os.path.join(cache_folder, f"bsd_model_data_{data_arg}.pq")
    data = pd.read_parquet(data_file)
    data["ID"] = data["Subject"]
    all_data_sim, all_params = [], []

    # model_arg = 'PC_fixpswgam'
    with Pool(processes=psutil.cpu_count(logical=False)) as pool:
        results = [
            pool.apply_async(
                fit_model_per_subject,
                args=(data[data["Subject"] == subj].reset_index(drop=True), model),
            )
            for subj in data["Subject"].unique()
        ]
        for r in tqdm(results):
            id_sim, id_fp = r.get()
            all_data_sim.append(id_sim)
            all_params.append(id_fp)
    version = "v14"  # 'v11_swp05' # TODO: dont forget to test restructure in v14!
    # v12 for model recovery test
    data_sim_opt = pd.concat(all_data_sim, axis=0).sort_values(
        ["Subject", "Session", "Trial"]
    )
    data_sim_opt.to_csv(
        os.path.join(
            cache_folder, f"bsd_simopt_data_{data_arg}_{model_arg}_{version}.csv"
        ),
        index=False,
    )
    pd.concat(all_params, axis=0).sort_values("ID").to_csv(
        os.path.join(
            cache_folder, f"bsd_simopt_params_{data_arg}_{model_arg}_{version}.csv"
        ),
        index=False,
    )
    # v7: multiprocessing fit multiple animals


# model testing
def test_model_recovery_mp(model, model_arg):
    cache_folder = r"D:\U19\data\Probswitch\caching"
    # data_arg = "full_20230105"
    data_arg = "eckstein2022_full"
    version = "v12"
    data_file = os.path.join(cache_folder, f"bsd_model_data_{data_arg}.pq")
    data = pd.read_parquet(data_file)
    data["ID"] = data["Subject"]
    param_file = os.path.join(
        cache_folder, f"bsd_simopt_params_{data_arg}_{model_arg}_{version}.csv"
    )
    params_df = pd.read_csv(param_file)
    md = model()
    data = md.fit_marginal(data)
    em_data = md.emulate(data, params_df)
    all_data_sim, all_params = [], []
    with Pool(processes=psutil.cpu_count(logical=False)) as pool:
        results = [
            pool.apply_async(
                fit_model_per_subject,
                args=(
                    em_data[em_data["Subject"] == subj].reset_index(drop=True),
                    model,
                ),
            )
            for subj in data["Subject"].unique()
        ]
        for r in tqdm(results):
            id_sim, id_fp = r.get()
            all_data_sim.append(id_sim)
            all_params.append(id_fp)

    data_sim_opt = pd.concat(all_data_sim, axis=0).sort_values(
        ["Subject", "Session", "Trial"]
    )
    data_sim_opt.to_csv(
        os.path.join(
            cache_folder,
            f"bsd_simopt_data_{data_arg}_{model_arg}_recover_{version}.csv",
        ),
        index=False,
    )
    pd.concat(all_params, axis=0).sort_values("ID").to_csv(
        os.path.join(
            cache_folder,
            f"bsd_simopt_params_{data_arg}_{model_arg}_recover_{version}.csv",
        ),
        index=False,
    )
    # md.fit


def test_multiple_animals():
    cache_folder = r"D:\U19\data\Probswitch\caching"
    data_arg = "sample"  # "full_20230105"
    data_file = os.path.join(cache_folder, f"bsd_model_data_{data_arg}.pq")
    data = pd.read_parquet(data_file)
    data["ID"] = data["Subject"]
    all_data_sim, all_params = [], []
    model_arg = "PC_fixpswgam"
    fit_by_subject = fit_model_per_subject_helper(PCModel_fixpswgam)

    for subj in tqdm(data["Subject"].unique()):
        id_sim, id_fp = fit_by_subject(
            data[data["Subject"] == subj].reset_index(drop=True)
        )
        all_data_sim.append(id_sim)
        all_params.append(id_fp)
    data_sim_opt = pd.concat(all_data_sim, axis=0).sort_values(
        ["Subject", "Session", "Trial"]
    )
    data_sim_opt.to_csv(
        os.path.join(cache_folder, f"bsd_simopt_data_{data_arg}_{model_arg}_v6.csv"),
        index=False,
    )
    pd.concat(all_params, axis=0).sort_values("ID").to_csv(
        os.path.join(cache_folder, f"bsd_simopt_params_{data_arg}_{model_arg}_v6.csv"),
        index=False,
    )
    # v6: fit multiple animals


def compare_models():
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

    cache_folder = r"D:\U19\data\Probswitch\caching"
    data_arg = "full_20230118"
    thres = 0.5
    model_names = ["PC", "PCf", "BI", "BIfp", "RLCF", "RL4p"]
    version = "v13"
    results = []

    def calculate_scores_perID(df):
        # assume no decision == -1
        choice_p = df["choice_p"].values
        decision = df["Decision"].astype(int).values
        pred_choice = (choice_p >= thres).astype(int)
        bac = balanced_accuracy_score(decision, pred_choice)
        ac = accuracy_score(decision, pred_choice)
        rauc = roc_auc_score(decision, choice_p)
        return pd.DataFrame(
            {
                "ID": [df["ID"].unique()[0]],
                "BAC": [bac],
                "accuracy": [ac],
                "auc_roc": [rauc],
            }
        )

    for model_arg in model_names:
        simfile = os.path.join(
            cache_folder, f"bsd_simopt_data_{data_arg}_{model_arg}_{version}.csv"
        )
        paramfile = os.path.join(
            cache_folder, f"bsd_simopt_params_{data_arg}_{model_arg}_{version}.csv"
        )
        simdata = pd.read_csv(simfile)
        param_df = pd.read_csv(paramfile)
        simdata = simdata[simdata["Decision"] != -1].reset_index(drop=True)
        rdf = (
            simdata.groupby("ID")
            .apply(calculate_scores_perID)
            .reset_index(drop=True)
            .merge(param_df[["ID", "aic", "bic"]], on="ID")
        )
        rdf["model"] = model_arg
        results.append(rdf)
    result_df = pd.concat(results, axis=0).reset_index(drop=True)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 5))
    for i, metric in enumerate(["BAC", "accuracy", "auc_roc", "aic", "bic"]):
        sns.boxplot(data=result_df, x="model", y=metric, ax=axes.ravel()[i])

    axes.ravel()[-1].axis("off")
    plt.tight_layout()
    plt.show()
    return result_df


## Tests for softmax: conclusion, qdiff works


def softmax(V, beta):
    """
    Anne's implementation
    p = softmax(V) returns the softmax policy
    (probability distribution of picking an action)

    V = [v1,v2, ...,vn] is an n dimensional real vector representating
    the value of n different options

    beta is the inverse temperature parameter, a non-negative scalar

    p =[p1,p2, ..., pn] is the probability vector of selecting actions
    1-n under softmax policy with values V

    """

    # exercise: write your function here
    """
    # method 1
    # initialize a vector with the same size as V but all zeros
    p = np.zeros_like(V)

    # get the denominator, i.e. the sum over i of exp(beta*v_i)
    den = sum(math.exp(beta*v) for v in V)

    # calculate the probability of each action given the values
    for v_idx, v in enumerate(V):
      p[v_idx] = math.exp(beta*v)/den

    # normalize to make sure it sums to 1 (in case there are multiple cases where V is highest)
    p=p/np.sum(p)
    """
    # method 2a (solution)
    # p=np.exp(beta*V)/np.sum(np.exp(beta*V))

    # method 2b (solution)
    # rescuing high betas at the cost of efficiency
    p = [1 / np.sum(np.exp(beta * (V - V[a]))) for a in range(len(V))]

    # error checking
    if (
        (np.sum(p) < 0)
        or (np.sum(p) > 1.000001)
        or (np.any(np.isnan(p)) or (not np.allclose(np.sum(p), 1)))
    ):
        print(p)
        print(beta)
        print(V)
        raise ValueError("p is not a probability")

    return p


def sam_softmax(data, beta):
    return expit(data["qdiff"] * beta)


def qdiff(V, beta):
    return expit(beta * (V[1] - V[0]))


def viz_test_softmax():
    model_arg = "RLCF"
    cache_folder = r"D:\U19\data\Probswitch\caching"
    gen_arg = "eckstein2022"

    test_data = pd.read_csv(
        os.path.join(cache_folder, f"genrec_{model_arg}_{gen_arg}_gendata.csv")
    )

    rlm = RL()
    # TODO: handle miss decisions
    data = rlm.fit_marginal(test_data)
    # data.to_csv(os.path.join(cache_folder, 'bsd_sim_sample_RLCF.csv'))
    params_df = pd.read_csv(
        os.path.join(cache_folder, r"genrec_RLCF_eckstein2022_paramsx.csv")
    )
    # data_sim.to_csv(os.path.join(cache_folder, "bsd_sim_data_sample.csv"))
    c_valid_sel = data["Decision"] != -1
    beta = 3.0
    p_s = expit(data.loc[c_valid_sel, "qdiff"].values * beta)

    v = data.loc[c_valid_sel, ["q0", "q1", "Decision", "ID"]]
    p_annes = np.zeros_like(p_s)
    p_v = np.zeros_like(p_s)
    for i in range(len(v)):
        if i <= 2000:
            q0, q1, d, i_id = v.iloc[i]
            p_annes[i] = softmax(
                np.array([q0, q1]),
                params_df.loc[params_df["ID"] == i_id, "beta"].values[0],
            )[1]
            p_v[i] = qdiff(
                np.array([q0, q1]),
                params_df.loc[params_df["ID"] == i_id, "beta"].values[0],
            )
            print(p_annes[i], p_v[i])


def viz_model_comp():
    cache_folder = r"D:\U19\data\Probswitch\caching"
    data_arg = "full_20230118"
    version = "v13"
    all_mdf = []
    for model in [PCModel_fixpswgam, BIModel_fixp, RLCF, RL_4p, PCModel, BIModel]:
        mdl = model()
        model_arg = str(mdl)
        pfile = os.path.join(
            cache_folder, f"bsd_simopt_params_{data_arg}_{model_arg}_{version}.csv"
        )
        mdf = pd.read_csv(pfile)[["ID", "aic", "bic", "beta", "st"]]
        mdf["model"] = model_arg
        all_mdf.append(mdf)
    all_mdf = pd.concat(all_mdf, axis=0)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    sns.set_context("talk")
    sns.boxplot(data=all_mdf, x="model", y="aic", ax=axes[0][0])
    sns.boxplot(data=all_mdf, x="model", y="bic", ax=axes[0][1])
    for j, prm in enumerate(["beta", "st"]):
        coef_df = pd.pivot_table(
            all_mdf, values=prm, index="ID", columns="model"
        ).reset_index(drop=True)
        sns.heatmap(coef_df.corr(), cmap="coolwarm", ax=axes[1][j])
        mnames = list(coef_df.columns)
        axes[1][j].set_xticks(np.arange(len(mnames)) + 0.5)
        axes[1][j].set_xticklabels(mnames)
        axes[1][j].set_yticks(np.arange(len(mnames)) + 0.5)
        axes[1][j].set_yticklabels(mnames)
    sns.despine()
    plt.show()


if __name__ == "__main__":
    # test_mp_multiple_animals(PCModel, 'PC')
    # test_mp_multiple_animals(PCModel_fixpswgam, 'PC_fixpswgam')
    # test_mp_multiple_animals(BIModel, 'BI')
    import time

    # test_model_genrec_eckstein2022()
    # test_model_genrec_eckstein2022_RL()
    # test_model_genrec_eckstein2022()
    # test_model_eckstein2022_RLCF()
    # test_model_eckstein2022_RLCF()
    print("test RL restruct model recover")
    # test_model_genrec_eckstein2022_RLCF()
    # test_model_genrec_eckstein2022()
    # test_model_genrec_eckstein2022_BIfp()
    # test_model_genrec_eckstein2022_PCf()
    # for model in [PCModel_fixpswgam, BIModel_fixp, RLCF, RL_4p, PCModel, BIModel]:
    for model in [RLCF, RL_4p]:
        test_mp_multiple_animals(model)
        test_model_genrec_BSD(model)
    # test_model_recovery_mp(PCModel, 'PC')
    # test_model_recovery_mp(BIModel, 'BI')
    # test_model_recovery_mp(RLCF, 'RLCF')
    # print(2023.2)
    # test_model_eckstein2022_RLCF()
    # test_mp_multiple_animals(BIModel_fixp, 'BI_fixp')
