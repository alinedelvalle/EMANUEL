from multiprocessing import Pool

# meka_adapted4: com timeout e tamanho do classificador
from meka.meka_adapted4 import MekaAdapted

from skmultilearn.dataset import load_from_arff

from utils.pareto_froint import Point


class EvaluateMLC:

    def __init__(self, java_command, meka_classpath, fold, n_processes, name_dataset, path_dataset, is_dataset_sparse, n_labels):
        self.java_command = java_command
        self.meka_classpath = meka_classpath    
        self.fold = fold    
        self.n_processes = n_processes
        
        self.name_dataset = name_dataset
        self.path_dataset = path_dataset
        self.is_dataset_sparse = is_dataset_sparse
        self.n_labels = n_labels
    

    def my_eval(self, pto):
        status = "sucess"
        macro_f1 = 0 # f1
        model_size = 1e9 # size - 1GB

        # -------------------------
        # Dataset paths
        # -------------------------
        prefix = "norm-" if pto.norm else ""
        train_data = (f"{self.path_dataset}/train_test/{self.name_dataset}-{prefix}train-{self.fold}.arff")
        test_data = (f"{self.path_dataset}/train_test/{self.name_dataset}-{prefix}test-{self.fold}.arff")      

        # -------------------------
        # Number of test examples
        # -------------------------
        X_test, y_test = load_from_arff(
            test_data,
            label_count=self.n_labels,
            label_location="end",
            load_sparse=False
        )
        n_test_example = X_test.shape[0]
        #print(pto)

        # -------------------------
        # Evaluation
        # -------------------------
        try:
            meka = MekaAdapted(
                meka_classifier = pto.meka,
                weka_classifier = pto.weka,
                meka_classpath = self.meka_classpath, 
                java_command = self.java_command,
                timeout = None
            )

            # predictions
            predictions = meka.fit_predict(n_test_example, self.n_labels, train_data, test_data)
                        
            stats = meka.statistics
            model_size = meka.len_model_file 
            macro_f1 = stats.get('F1 (macro averaged by label)')
            
        except Exception as e:
            status = f"error: {str(e)}"
            #print(status)
    
        return Point(-macro_f1, model_size, pto.norm, pto.meka, pto.weka, pto.k, predictions.toarray())
            


    def calc_points_test(self, list_points):
        # prepare the parameters for the pool
        params = [pto for pto in list_points]

        with Pool(processes=self.n_processes) as pool:
            results = pool.map(self.my_eval, params)

        list_points_test = [r for r in results]
        
        return list_points_test