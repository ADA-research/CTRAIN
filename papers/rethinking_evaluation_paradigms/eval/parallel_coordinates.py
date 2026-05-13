import os
from deepcave.plugins.hyperparameter.parallel_coordinates import ParallelCoordinates
from deepcave.runs.converters.optuna import OptunaRun
from ConfigSpace.hyperparameters import NumericalHyperparameter
import plotly.express as px
import plotly.graph_objects as go
from deepcave.utils.styled_plotty import get_hyperparameter_ticks
from deepcave.evaluators.fanova import fANOVA
from collections import defaultdict
import numpy as np
import pickle
import optuna

from deepcave.constants import VALUE_RANGE

renames = {
    "Objective0": "Clean Accuracy",
    "Objective1": "Cert. Accuracy",
    "train_eps_factor": "Train Eps. Factor",
    "warm_up_epochs": "Warmup Epochs",
    "l1_reg_weight": "l1 Reg. Weight",
    "lr_decay_factor": "LR Decay Factor",
    "mtl_ibp:mtl_ibp_alpha": "MTL IBP α",
    "learning_rate": "Learning Rate",
    "mtl_ibp:pgd_alpha": "PGD α",
    "mtl_ibp:pgd_steps": "PGD Steps",
    "sabr:subselection_ratio": "Subsel. Ratio",
    "sabr:pgd_steps": "PGD Steps",
    "sabr:pgd_eps_factor": "PGD Eps. Factor",
    "sabr:pgd_alpha": "PGD α",
    "shi:end_kappa": "End κ",
    "shi:start_kappa": "Start κ",
    "ramp_up_epochs": "Ramp-up Epochs",
    "lr_decay_epoch_1": "LR Decay Epoch 1",
    "lr_decay_epoch_2": "LR Decay Epoch 2",
    "crown_ibp:end_kappa": "End κ",
    "crown_ibp:start_kappa": "Start κ",
    "crown_ibp:end_beta": "End β",
    "crown_ibp:start_beta": "Start β",
    "shi_reg_weight": "Shi Reg. Weight",
    "mtl_ibp:mtl_ibp_eps_factor": "PGD Eps. Factor",
    "optimizer_func": "Optimiser",
}

if __name__ == "__main__":
    os.makedirs("importance_analysis/", exist_ok=True)
    for method in ["mtl_ibp", "sabr", "shi", "crown_ibp_nofusion", "crown_ibp"]:
        for dataset in ["cifar10", "tinyimagenet"]:
            if dataset == "tinyimagenet" and method in ["crown_ibp_nofusion"]:
                continue    
            if dataset == "cifar10" and method in ["crown_ibp"]:
                continue
            for network in [
                "cnn7"
            ]:
                for eps in [2 / 255, 8 / 255] if dataset == "cifar10" else [1 / 255]:
                    for oid, obj in enumerate(["nat", "cert"]):
                        mega_study = optuna.create_study(
                        directions=["maximize", "maximize"], study_name="moctrain"
                        )
                        for seed in range(3):
                            print(
                                f"sqlite:///../results/hpo/optuna_results/{dataset}_{network}_{method}_{eps}_{seed}_optuna_study.db"
                            )
                            study = optuna.load_study(
                                study_name="moctrain",
                                storage=f"sqlite:///../results/hpo/optuna_results/{dataset}_{network}_{method}_{eps}_{seed}_optuna_study.db",
                            )
                            mega_study.add_trials(study.trials[:100])
                        os.makedirs("/tmp/mega_study", exist_ok=True)
                        with open("/tmp/mega_study/mega_study.pkl", "wb") as f:
                            pickle.dump(mega_study, f)
                            
                        run = OptunaRun.from_path(
                            path=f"/tmp/mega_study/",
                        )
                        assert (
                            run.meta["objectives"][0]["upper"]
                            > run.meta["objectives"][1]["upper"]
                        )

                        df = run.get_encoded_data()

                        

                        evaluator = fANOVA(run)
                        evaluator.calculate(
                            run.get_objective(f"Objective{oid}"),
                            run.get_budget_ids()[0],
                            n_trees=10,
                            seed=0,
                        )
                        importances_dict = evaluator.get_importances()
                        importances = {u: v[0] for u, v in importances_dict.items()}
                        important_hp_names = sorted(
                            importances, key=lambda key: importances[key], reverse=True
                        )

                        if eps == 2 / 255:
                            df = df[df["Objective0"] > 0.6]
                            df = df[df["Objective1"] > 0.4]
                        elif eps == 8 / 255:
                            df = df[df["Objective0"] > 0.4]
                            df = df[df["Objective1"] > 0.25]
                        elif eps == 1 / 255:
                            df = df[df["Objective0"] > 0.2]
                            df = df[df["Objective1"] > 0.15]
                        for id, row in df.iterrows():
                            for id2, row2 in df.iterrows():
                                if (
                                    row2["Objective0"] > row["Objective0"]
                                    and row2["Objective1"] >= row["Objective1"]
                                ):
                                    df.drop(id, inplace=True)
                                    break
                                
                        objective_values = []
                        for value in df[f"Objective{oid}"].values:
                            b = np.isnan(value)

                            objective_values += [value]

                        data: defaultdict = defaultdict(dict)

                        
                        for hp_name in important_hp_names[:5]:
                            values = []
                            for hp_v, objective_v in zip(
                                df[hp_name].values, df[f"Objective{oid}"].values
                            ):
                                b = np.isnan(objective_v)
                                values += [hp_v]

                            data[hp_name]["values"] = values
                            data[hp_name]["label"] = f"{renames[hp_name]}<br><span style='font-size:10px'>Imp: {importances[hp_name]:.3f}</span>"
                            data[hp_name]["range"] = [0, 1]
                            print(values)
                            hp = run.configspace[hp_name]
                            tickvals, ticktext = get_hyperparameter_ticks(
                                hp, ticks=4, include_nan=False
                            )

                            data[hp_name]["tickvals"] = tickvals
                            data[hp_name]["ticktext"] = ticktext

                        if oid == 0:


                            if eps == 2 / 255:
                                prange = [0.65, 0.84]
                            elif eps == 8 / 255:
                                prange = [0.2, 0.5]
                            elif eps == 1 / 255:
                                prange = [0.15, 0.45]

                            data[f"Objective{oid}"]["range"] = prange
                        else:
                            if eps == 2 / 255:
                                prange = [0.4, 0.6]
                            elif eps == 8 / 255:
                                prange = [0.2, 0.4]
                            elif eps == 1 / 255:
                                prange = [0.1, 0.3]

                            data[f"Objective{oid}"]["range"] = prange

                        data[f"Objective{oid}"]["values"] = objective_values
                        data[f"Objective{oid}"]["label"] = renames[f"Objective{oid}"]
                        line = dict(
                            color=data[f"Objective{oid}"]["values"],
                            showscale=True,
                            colorscale="aggrnyl",
                        )

                        figure = go.Figure(
                            data=go.Parcoords(
                                line=line,
                                dimensions=list([d for d in data.values()]),
                                labelangle=45,
                                labelfont=dict(size=12, family="Times New Roman", color="black"),
                                legendwidth=2,
                                tickfont=dict(size=12, family="Times New Roman", color="black")
                            ),
                            layout=dict(
                                margin=dict(t=100, b=25, l=60, r=0),
                                template="seaborn",
                                width=450,
                                height=300,
                        ),
                        )

                        figure.update_layout(
                            template="seaborn",
                            font_family="Times New Roman",
                            font_size=12,
                        )

                        figure.write_image(
                            f"importance_analysis/{dataset}_{network}_{method}_{eps}_{obj}.pdf"
                        )
