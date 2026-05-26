import optuna
import pprint
from CTRAIN.model_wrappers.configs import (
    build_crown_ibp_config_space,
    build_sabr_config_space,
    build_mtl_ibp_config_space,
    build_shi_config_space,
)
from itertools import combinations
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
)
from smac.utils.configspace import get_config_hash
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, fclusterdata
from scipy.spatial.distance import pdist
import os

PAPER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_ROOT = os.environ.get("CTRAIN_PAPER_RESULTS_ROOT", os.path.join(PAPER_ROOT, "results"))
RESULTS_PATH = os.path.join(RESULTS_ROOT, "hpo", "pareto_fronts")
DISTANCE_THRESHOLD = 0.05

if __name__ == "__main__":
    os.makedirs(RESULTS_PATH, exist_ok=True)
    for method in ["mtl_ibp", "sabr", "shi", "crown_ibp", "crown_ibp_nofusion"]:
        dataset = "cifar10"
        for network in [
            "cnn7",
            "wide_cnn7",
            "cnn5",
            "cnn9",
            "narrow_cnn7",
        ]:
            
            for eps in [2 / 255, 8 / 255, .3, 1 / 255]:
                mega_study = optuna.create_study(directions=["maximize", "maximize"])
                for seed in range(3):
                    try:
                        print(
                            f"sqlite:///{RESULTS_ROOT}/hpo/optuna_results/{dataset}_{network}_{method}_{eps}_{seed}_optuna_study.db"
                        )
                        study = optuna.load_study(
                            study_name="moctrain",
                            storage=f"sqlite:///{RESULTS_ROOT}/hpo/optuna_results/{dataset}_{network}_{method}_{eps}_{seed}_optuna_study.db",
                        )
                        mega_study.add_trials(study.trials[:100])
                    except Exception as e:
                        print(f"Failed to load study for seed {seed}: {e}")
                        continue
                
                if len(mega_study.trials) != 300:
                    print(f"Warning: Only {len(mega_study.trials)} trials found, expected 300.")
                    continue

                fig = optuna.visualization.plot_pareto_front(
                    mega_study,
                    target_names=["nat acc", "cert acc"],
                    targets=lambda t: (t.values[0], t.values[1]),
                    include_dominated_trials=False,
                )

                fig.write_image(f"{RESULTS_PATH}/pareto_front_{method}_{network}_{dataset}_{eps}.pdf")

                pareto_front_values = {}
                pareto_front_configs = {}
                pareto_front_trials = {}
                for trial in mega_study.best_trials:
                    pareto_front_values[trial.number] = trial.values
                    pareto_front_configs[trial.number] = trial.params
                    pareto_front_trials[trial.number] = trial

                with open(
                    f"{RESULTS_PATH}/pareto_front_{method}_{network}_{dataset}_{eps}.txt", "w"
                ) as f:
                    for trial_number, values in pareto_front_values.items():
                        f.write(f"Trial {trial_number}: {values}\n")
                        f.write(f"Config: {pareto_front_configs[trial_number]}\n")
                        config = {}
                        if method == "crown_ibp" or method == "crown_ibp_nofusion":
                            config_space = build_crown_ibp_config_space(
                                epochs=160 if eps == 2 / 255 else 260,
                                eps=eps,
                            )
                        elif method == "sabr":
                            config_space = build_sabr_config_space(
                                epochs=160 if eps == 2 / 255 else 260,
                                eps=eps,
                            )
                        elif method == "mtl_ibp":
                            config_space = build_mtl_ibp_config_space(
                                epochs=160 if eps == 2 / 255 else 260,
                                eps=eps,
                            )
                        elif method == "shi":
                            config_space = build_shi_config_space(
                                epochs=160 if eps == 2 / 255 else 260,
                                eps=eps,
                            )
                        trial = pareto_front_trials[trial_number]
                        for hp in config_space:
                            hp_obj = config_space[hp]
                            if hasattr(hp_obj, "choices"):
                                config[hp] = trial.suggest_categorical(
                                    hp, hp_obj.choices
                                )
                            elif hasattr(hp_obj, "lower") and hasattr(hp_obj, "upper"):
                                if isinstance(hp_obj, UniformFloatHyperparameter):
                                    config[hp] = trial.suggest_float(
                                        hp, hp_obj.lower, hp_obj.upper, log=hp_obj.log
                                    )
                                elif isinstance(hp_obj, UniformIntegerHyperparameter):
                                    config[hp] = trial.suggest_int(
                                        hp, hp_obj.lower, hp_obj.upper, log=hp_obj.log
                                    )
                                else:
                                    raise ValueError(
                                        f"Unsupported hyperparameter type: {type(hp_obj)}"
                                    )
                            elif isinstance(hp_obj, Constant):
                                config[hp] = hp_obj.default_value
                            else:
                                raise ValueError(f"Unsupported hyperparameter: {hp}")
                        f.write(f"Config hash: {get_config_hash(config, chars=32)}\n")

                # exit(0)
                if len(pareto_front_values) > 5:
                    xs = np.array([v[0] for v in pareto_front_values.values()])
                    ys = np.array([v[1] for v in pareto_front_values.values()])
                    keys = list(pareto_front_values.keys())

                    points = np.array(list(zip(xs, ys)))

                    points_normalized = points.copy()
                    points_normalized[:, 0] = (
                        points_normalized[:, 0] - points_normalized[:, 0].min()
                    ) / (points_normalized[:, 0].max() - points_normalized[:, 0].min())
                    points_normalized[:, 1] = (
                        points_normalized[:, 1] - points_normalized[:, 1].min()
                    ) / (points_normalized[:, 1].max() - points_normalized[:, 1].min())

                    cluster_labels = fclusterdata(points_normalized, DISTANCE_THRESHOLD, criterion='distance')

                    print(f"Found {len(np.unique(cluster_labels))} clusters:")
                    new_points = []
                    new_pareto_front_values = {}
                    for cluster_label in enumerate(np.unique(cluster_labels)):
                        cluster = np.where(cluster_labels == cluster_label[1])[0]
                        cluster_points = points[cluster]
                        print(f"  Cluster {cluster_label}: {len(cluster)} points")
                        print(f"    Points: {cluster_points}")
                        print(f"    Center: {np.mean(cluster_points, axis=0)}")
                        max_nat = 0
                        max_cert = 0
                        for i, (x, y) in enumerate(cluster_points):
                            if x > max_nat:
                                max_nat = x
                                max_cert = y
                        new_points.append((max_nat, max_cert))
                        
                    
                    
                    for k, (x1, y1) in pareto_front_values.items():
                        for i, (x2, y2) in enumerate(new_points):
                            if np.isclose(x1, x2) and np.isclose(y1, y2):
                                new_pareto_front_values[k] = (x2, y2)
                                break
                    pareto_front_values = new_pareto_front_values

                mega_study_subselected = optuna.create_study(
                    directions=["maximize", "maximize"]
                )
                mega_study_subselected.add_trials(
                    [
                        mega_study.trials[trial_number]
                        for trial_number in pareto_front_values.keys()
                    ]
                )
                fig = optuna.visualization.plot_pareto_front(
                    mega_study_subselected,
                    target_names=["nat acc", "cert acc"],
                    targets=lambda t: (t.values[0], t.values[1]),
                )

                fig.write_image(
                    f"{RESULTS_PATH}/pareto_front_{method}_{network}_{dataset}_{eps}_subselected{DISTANCE_THRESHOLD}.pdf"
                )

                with open(
                    f"{RESULTS_PATH}/pareto_front_{method}_{network}_{dataset}_{eps}_subselected{DISTANCE_THRESHOLD}.txt",
                    "w",
                ) as f:
                    for trial_number, values in pareto_front_values.items():
                        f.write(f"Trial {trial_number}: {values}\n")
                        f.write(f"Config: {pareto_front_configs[trial_number]}\n")
                        config = {}
                        if method == "crown_ibp" or method == "crown_ibp_nofusion":
                            config_space = build_crown_ibp_config_space(
                                epochs=160 if eps == 2 / 255 else 260,
                                eps=eps,
                            )
                        elif method == "sabr":
                            config_space = build_sabr_config_space(
                                epochs=160 if eps == 2 / 255 else 260,
                                eps=eps,
                            )
                        elif method == "mtl_ibp":
                            config_space = build_mtl_ibp_config_space(
                                epochs=160 if eps == 2 / 255 else 260,
                                eps=eps,
                            )
                        elif method == "shi":
                            config_space = build_shi_config_space(
                                epochs=160 if eps == 2 / 255 else 260,
                                eps=eps,
                            )
                        trial = pareto_front_trials[trial_number]
                        for hp in config_space:
                            hp_obj = config_space[hp]
                            if hasattr(hp_obj, "choices"):
                                config[hp] = trial.suggest_categorical(
                                    hp, hp_obj.choices
                                )
                            elif hasattr(hp_obj, "lower") and hasattr(hp_obj, "upper"):
                                if isinstance(hp_obj, UniformFloatHyperparameter):
                                    config[hp] = trial.suggest_float(
                                        hp, hp_obj.lower, hp_obj.upper, log=hp_obj.log
                                    )
                                elif isinstance(hp_obj, UniformIntegerHyperparameter):
                                    config[hp] = trial.suggest_int(
                                        hp, hp_obj.lower, hp_obj.upper, log=hp_obj.log
                                    )
                                else:
                                    raise ValueError(
                                        f"Unsupported hyperparameter type: {type(hp_obj)}"
                                    )
                            elif isinstance(hp_obj, Constant):
                                config[hp] = hp_obj.default_value
                            else:
                                raise ValueError(f"Unsupported hyperparameter: {hp}")
                        f.write(f"Config hash: {get_config_hash(config, chars=32)}\n")
