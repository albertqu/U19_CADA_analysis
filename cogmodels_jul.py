import os
from tqdm import tqdm
import pandas as pd
from cogmodels.utils_test import *


CACHE_FOLDER = r"D:\U19\data\DRD_PS\caching"
VERSION = "v1"


def test_mp_multiple_animals_jul(model):
    # Function for fitting data for multiple animals
    model_arg = str(model())
    cache_folder = CACHE_FOLDER
    # data_arg = 'eckstein2022_full'
    data_file = os.path.join(cache_folder, f"jul_model_data.pq")
    data = pd.read_parquet(data_file)
    data = data[
        ((data["session_num"] >= 6) | (data["session_num"] == 4))
        & (data["outcome"] <= 90 * 60)
        & (data["Subject"] != "DRD004")
    ].reset_index(drop=True)
    data["ID"] = data["Subject"] + "_" + data["group"]
    all_data_sim, all_params = [], []

    # model_arg = 'PC_fixpswgam'
    with Pool(processes=psutil.cpu_count(logical=False)) as pool:
        results = [
            pool.apply_async(
                fit_model_per_subject,
                args=(data[data["ID"] == sid].reset_index(drop=True), model),
            )
            for sid in data["ID"].unique()
        ]
        for r in tqdm(results):
            id_sim, id_fp = r.get()
            all_data_sim.append(id_sim)
            all_params.append(id_fp)
    version = VERSION
    data_sim_opt = pd.concat(all_data_sim, axis=0).sort_values(
        ["Subject", "Session", "Trial"]
    )
    data_sim_opt["group"] = data_sim_opt["ID"].apply(lambda s: s.split("_")[1])
    data_sim_opt.to_csv(
        os.path.join(cache_folder, f"simopt_data_jul_{model_arg}_{version}.csv"),
        index=False,
    )
    ap_df = pd.concat(all_params, axis=0).sort_values("ID")
    ap_df["group"] = ap_df["ID"].apply(lambda s: s.split("_")[1])
    ap_df.to_csv(
        os.path.join(cache_folder, f"simopt_params_jul_{model_arg}_{version}.csv"),
        index=False,
    )


if __name__ == "__main__":
    import time

    print("test V1 2 models")
    for model in [
        # RL_Forgetting,
        # RL_Forgetting3p,
        BRL_fwr,
        RL_4p,
        BIModel_fixp,
        RFLR,
    ]:  # BRL_fp, BRL_wr, RL_4p, RLCF, PCModel_fixpswgam, PCBRL
        # for model in [BIModel_fixp, PCModel_fixpswgam, BI_log, PCBRL, RL_4p]:
        print(str(model()))
        test_mp_multiple_animals_jul(model)
