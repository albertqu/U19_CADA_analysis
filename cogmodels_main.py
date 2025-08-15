from cogmodels.utils_test import *
from cogmodels import *

""" 
Here we can find example runs of the data 
"""


def test_PCBRL():
    """First test to run to guarantee that there is no runtime error
    in the implementation -- test that code runs
    """
    # pcm = PCModel()
    model = PCBRL()
    model_arg = str(model)  # 'RLCF'
    v_old = "v13"

    # take in data
    cache_folder = CACHE_FOLDER
    data_arg = DATA_ARG
    data_file = os.path.join(cache_folder, f"bsd_model_data_{data_arg}.pq")
    data = pd.read_parquet(data_file)
    data["ID"] = data["Subject"]

    # select 2 animals 2 session each
    animal_list = ["BSD011", "BSD015"]
    session_list = sum([[a + "_01", a + "_02"] for a in animal_list], [])
    data = data[data["Session"].isin(session_list)].reset_index(drop=True)

    # test with parameters
    param_file = os.path.join(
        cache_folder, f"bsd_simopt_params_{data_arg}_PCf_{v_old}.csv"
    )
    params_df = pd.read_csv(param_file)
    params_df = params_df[params_df["ID"].isin(animal_list)].reset_index(drop=True)
    params_df["a_marg"] = [0.19, 0.2]

    data = model.fit_marginal(data)
    # data.to_csv(os.path.join(cache_folder, 'bsd_sim_sample_RLCF.csv'))
    # params_df = model.create_params(data)
    data_sim = model.sim(data, params_df)

    # data_sim.to_csv(os.path.join(cache_folder, "bsd_sim_data_sample.csv"))
    model = model.fit(data)
    data_sim_opt = model.sim(data, model.fitted_params)
    data_sim_opt.to_csv(
        os.path.join(
            cache_folder, f"bsd_simopt_data_{data_arg}_{model_arg}_{v_old}simTest.csv"
        )
    )
    model.fitted_params.to_csv(
        os.path.join(cache_folder, f"bsd_simopt_params_{model_arg}_{v_old}simTest.csv")
    )

    data_plot = data_sim[data_sim["Session"] == "BSD015_01"]
    plt.plot(data_plot["Trial"], np.around(data_plot["m"], 3))
    plt.scatter(data_plot["Trial"], data_plot["Decision"], color="r")
    plt.ylim([-0.1, 1.1])
    plt.show()
    # versions for PCModel
    # v1: optimization without bounds, wrong MLE
    # v2: set bounds, results came biases towards decision 1 (majority decision), wrong MLE
    # v3: fixed p_rew/eps, wrong MLE
    # v4: fix MLE equation fixed p_rew/eps
    # v5: fix MLE equation variable p_rew/eps


def test_BRLwr():
    param_file = os.path.join(CACHE_FOLDER, "bsd_simopt_params_BRLwr_test.csv")
    test_model_functional(BRL_wr(), param_file)


def BSD_model_genrec_viz():
    model = BRL_fwr  # [BRL_fp, BRL_wr, RL_4p, RLCF, PCModel_fixpswgam, PCBRL]
    viz_genrec_recovery(model, f"{DATA_ARG}_3s5ht_restruct", 3)


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
    print("rlmeta")
    # models = [RL_Forgetting3p, RL_Grossman, Pearce_Hall, RL_4p, RLCF, RFLR, BRL_fwr, BIModel_fixp]
    # test_model_identifiability_mp(models, f"{DATA_ARG}_3s5ht")
    # gen_arg = f"{DATA_ARG}_3s5ht"
    # model = RL_Grossman
    # method = "L-BFGS-B"

    # gendata = pd.read_csv(
    #     os.path.join(
    #         CACHE_FOLDER, f"genrec_{gen_arg}_{str(model())}_{method}_gendata.csv"
    #     )
    # )
    # fit_model_all_subjects(
    #     gendata.iloc[: len(gendata) // 20].reset_index(drop=True), RL_Grossman
    # )

    # test_model_genrec_eckstein2022_RLCF()
    # test_model_genrec_eckstein2022()
    # test_model_genrec_eckstein2022_BIfp()
    # test_model_genrec_eckstein2022_PCf()
    for model in [
        # RL_Forgetting,
        # RL_Forgetting3p,
        # # RL_FQST,
        # WSLS,
        # Pearce_Hall,
        RL_Grossman,
        # # RL_Grossman_prime,
        # RL_Grossman_nof,
        # RL_Grossman_nost,
        # BRL_wrp,
        # BRL_fwr,
        # # BRL_fw,
        # # BRL_fp,
        # BRL_wr,
        # RL_4p,
        # BIModel_fixp,
        # BIModel,
        # RFLR,
        # RLCF,
        # # PCModel_fixpswgam,
        # PCBRL,
    ]:  # BRL_fp, BRL_wr, RL_4p, RLCF, PCModel_fixpswgam, PCBRL
        # for model in [BIModel_fixp, PCModel_fixpswgam, BI_log, PCBRL, RL_4p]:
        print(str(model()))
        # test_mp_multiple_sessions(model)
        test_mp_multiple_animals(model)
        test_model_genrec_BSD(model)
    # test_model_recovery_mp(PCModel, 'PC')
    # test_model_recovery_mp(BIModel, 'BI')
    # test_model_recovery_mp(RLCF, 'RLCF')
    # print(2023.2)
    # test_model_eckstein2022_RLCF()
    # test_mp_multiple_animal-s(BIModel_fixp, 'BI_fixp')
