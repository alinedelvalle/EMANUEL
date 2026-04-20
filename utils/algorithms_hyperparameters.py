import json
import numpy as np
import pandas as pd
from itertools import count


class AlgorithmsHyperparameters:

    _rows = []
    _id_counter = count(start=1)
    _header_written = False

    METRIC_KEYS = [
        'Accuracy',
        'Jaccard index',
        'Hamming score',
        'Exact match',
        'Jaccard distance',
        'Hamming loss',
        'ZeroOne loss',
        'Harmonic score',
        'One error',
        'Rank loss',
        'Avg precision',
        'Log Loss (lim. L)',
        'Log Loss (lim. D)',
        'Micro Precision',
        'Micro Recall',
        'Macro Precision',
        'Macro Recall',
        'F1 (micro averaged)',
        'F1 (macro averaged by label)',
        'AUPRC (macro averaged)',
        'AUROC (macro averaged)',
        'Build Time',
        'Test Time',
        'Total Time',
        'Accuracy (per label)',
        'Harmonic (per label)',
        'Precision (per label)',
        'Recall (per label)',
        'avg. relevance (test set)',
        'avg. relevance (predicted)',
        'avg. relevance (difference)'
    ]

    @classmethod
    def add_metrics(cls, k, normalize, meka, weka, statistics, model_size, status):
        row = {
            "id": next(cls._id_counter),
            "k": k,
            "normalize": normalize,
            "meka": meka,
            "weka": weka,
            "model size": model_size, 
            "status": status
        }

        for key in cls.METRIC_KEYS:
            value = statistics.get(key)
            row[key] = cls._normalize_value(value)

        cls._rows.append(row)


    @classmethod
    def to_file(cls, dataset_name, file_name):
        if not cls._rows:
            return

        df = pd.DataFrame(cls._rows)
        df.insert(1, "Dataset", dataset_name)

        df.to_csv(
            file_name,
            sep=';',
            index=False,
            mode='a',
            header=not cls._header_written
        )

        cls._header_written = True
        cls._rows.clear()

    
    @staticmethod
    def _normalize_value(value):
        if value is None:
            return pd.NA

        # numpy scalar
        if isinstance(value, (np.floating, np.integer)):
            return float(value)

        # python scalar
        if isinstance(value, (int, float)):
            return float(value)

        # array / list → JSON
        if isinstance(value, (list, tuple, np.ndarray)):
            return json.dumps(list(value))

        # fallback (string, etc.)
        return str(value)

