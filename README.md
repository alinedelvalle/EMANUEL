**EMANUEL** is the implementation of the AutoML multi-objective strategy for multi-label classification. To execute the algorithm, we run the main.py file. The inputs to this algorithm are:
* n_run: numbere of runs.
* n_labels: number of labels.
* n_features: number of features.
* is_dataset_sparse: if the dataset is sparse.
* name_dataset: dataset name (ARFF). The k-folds of this dataset must be in folder datasets.
* meka_classpath: path to the lib folder of the MEKA library.
* java_command: path to the environment java.
* project: project path.
* k_folds: number of folds.
* n_process: process number.
* len_population: population size.
* number_generation: number of generations.
* limit_mlc_time: limit time in seconds to train the models.
* len_population: population size.
* number_generation: generation number.
* max_time: maximum runtime of EMANUEL (seconds).

The results of this algorithm are located in the folder results/name_dataset/ and are organized by n_runs. If n_run = 3, there will be the folders: name_dataset1, name_dataset2, and name_dataset3.
* Hypervolume graph by fold.
* Objective space graph by fold.
* Results File by fold.
* Metrics file (CSV) with the history of the evaluated algorithms.
* resuls.csv: algorithms and merics resulting from the k-folds.
* ObjectiveSpace.png: objective space resulting from the k-folds.
