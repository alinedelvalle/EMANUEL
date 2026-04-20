import os
import numpy as np
from multiprocessing import Pool
from pymoo.core.problem import Problem

from multiobjective.individual_utils import IndividualUtils
from multiobjective.ml_evaluation import evaluate_individual

from utils.algorithms_hyperparameters import AlgorithmsHyperparameters



class MLProblem(Problem):

    def __init__(
        self,
        config,
        java_command,
        meka_classpath,
        kfold,
        n_processes,
        limit_time,
        name_dataset,
        path_dataset,
        n_labels,
        log_file,
        metrics_file,
        cache_root="cache"
    ):
        super().__init__(n_var=28, n_obj=2)

        self.config = config
        self.java_command = java_command
        self.meka_classpath = meka_classpath
        self.kfold = kfold
        self.n_processes = n_processes
        self.limit_time = limit_time

        self.name_dataset = name_dataset
        self.path_dataset = path_dataset
        self.n_labels = n_labels

        self.log_file = log_file
        self.metrics_file = metrics_file

        # cache/dataset/fold_k
        self.cache_dir = os.path.join(
            cache_root,
            name_dataset,
            f"fold_{kfold}"
        )

        self.n_ger = 0


    def _evaluate(self, X, out, *args, **kwargs):
        params = []

        for i in range(len(X)):
            is_norm, meka_cmd, weka_cmd = IndividualUtils.get_commands(
                self.config, X[i]
            )

            params.append((
                is_norm,
                meka_cmd,
                weka_cmd,
                self.java_command,
                self.meka_classpath,
                self.limit_time,
                self.kfold,
                self.name_dataset,
                self.path_dataset,
                self.n_labels,
                self.cache_dir
            ))


        with Pool(processes=self.n_processes) as pool:
            res = pool.map(evaluate_individual, params)

        # objectives
        out["F"] = np.array([r["objectives"] for r in res])


        for r in res:
            AlgorithmsHyperparameters.add_metrics(
                r["fold"], 
                r["normalize"], 
                r["meka"], 
                r["weka"], 
                r["metrics"], 
                r["model_size"], 
                r["status"]
            )
        
        AlgorithmsHyperparameters.to_file(
            self.name_dataset,
            self.metrics_file
        )

        # logging
        self.n_ger += 1
        with open(self.log_file, "a") as f:
            f.write(f"Geração: {self.n_ger}\n")

