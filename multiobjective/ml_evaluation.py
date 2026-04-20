import os
import json
import hashlib
from subprocess import TimeoutExpired
from skmultilearn.dataset import load_from_arff
from meka.meka_adapted4 import MekaAdapted
from datetime import datetime
from utils.algorithms_hyperparameters import AlgorithmsHyperparameters

from filelock import FileLock


# =========================
# Cache utilities
# =========================

def command_hash(command: str) -> str:
    return hashlib.md5(command.encode("utf-8")).hexdigest()


def cache_path(base_cache_dir, command):
    h = command_hash(command)
    return os.path.join(base_cache_dir, f"{h}.json")


'''def load_cache(base_cache_dir, command):
    path = cache_path(base_cache_dir, command)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None'''

def load_cache(base_cache_dir, command):
    #cache_file = os.path.join(base_cache_dir, f"{command}.json")
    cache_file = cache_path(base_cache_dir, command)
    lock_file = cache_file + ".lock"

    if not os.path.exists(cache_file):
        return None

    with FileLock(lock_file):
        try:
            with open(cache_file, "r") as f:
                content = f.read().strip()
                if content == "":
                    return None
                return json.loads(content)
        except Exception:
            return None


'''def save_cache(base_cache_dir, command, data):
    os.makedirs(base_cache_dir, exist_ok=True)
    path = cache_path(base_cache_dir, command)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)'''

def save_cache(base_cache_dir, command, data):
    #cache_file = os.path.join(base_cache_dir, f"{command}.json")
    cache_file = cache_path(base_cache_dir, command)
    lock_file = cache_file + ".lock"

    with FileLock(lock_file):
        with open(cache_file, "w") as f:
            json.dump(data, f)



# =========================
# Pure evaluation function
# =========================

def evaluate_individual(params):
    is_normalize, meka_command, weka_command, java_command, meka_classpath, limit_time, kfold, name_dataset, path_dataset, n_labels, base_cache_dir = params

    stats = {}
    macro_f1 = 0.0
    model_size = 1e9
    status = "success"

    # -------------------------
    # Build command signature
    # -------------------------
    command = f"{is_normalize} {meka_command}"
    if weka_command:
        command += f" -W {weka_command}"

    # -------------------------
    # Cache lookup
    # -------------------------
    cached = load_cache(base_cache_dir, command)
    if cached is not None:
        macro_f1 = cached["objectives"]["macro_fscore"]
        model_size = cached["objectives"]["model_size"]
        #print('Aproveitou')

    else:
        # -------------------------
        # Dataset paths
        # -------------------------
        prefix = "norm-" if is_normalize else ""

        train_data = (
            f"{path_dataset}/train_val/"
            f"{name_dataset}-{prefix}train-v2-{kfold}.arff"
        )
        val_data = (
            f"{path_dataset}/train_val/"
            f"{name_dataset}-{prefix}val-v2-{kfold}.arff"
        )

        # -------------------------
        # Number of test examples
        # -------------------------
        x_val, _ = load_from_arff(
            val_data,
            label_count=n_labels,
            label_location="end",
            load_sparse=False
        )
        n_test_example = x_val.shape[0]

        # -------------------------
        # Evaluation
        # -------------------------
        try:
            meka = MekaAdapted(
                meka_classifier=meka_command,
                weka_classifier=weka_command,
                meka_classpath=meka_classpath,
                java_command=java_command,
                timeout=limit_time
            )

            meka.fit_predict(
                n_test_example,
                n_labels,
                train_data,
                val_data
            )

            stats = meka.statistics
            macro_f1 = stats.get("F1 (macro averaged by label)", 0.0)
            model_size = meka.len_model_file

        except TimeoutExpired:
            status = "timeout"

        except Exception as e:
            #status = f"error: {str(e)}"
            status = f"error"

        # -------------------------
        # Objectives
        # -------------------------
        #print(macro_f1, model_size)

        # -------------------------
        # Save cache
        # -------------------------
        save_cache(
            base_cache_dir,
            command,
            {
                "dataset": name_dataset,
                "fold": kfold,
                "normalize": is_normalize,
                "command": command,
                "objectives": {
                    "macro_fscore": macro_f1,
                    "model_size": model_size
                },
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        )

    return {
        "dataset": name_dataset,
        "fold": kfold,
        "normalize": is_normalize,
        "meka": meka_command,
        "weka": weka_command,
        "objectives": [-macro_f1, model_size],
        "metrics": stats,
        "model_size": model_size,
        "status": status
    }

