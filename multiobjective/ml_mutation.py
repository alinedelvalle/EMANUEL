import numpy as np

from pymoo.core.mutation import Mutation

from multiobjective.individual_utils import IndividualUtils


class MLMutation(Mutation):
    
    def __init__(self, prob, config, is_dataset_sparse):
        super().__init__()
        self.prob = prob
        self.config = config
        self.is_dataset_sparse = is_dataset_sparse


    def _do(self, problem, X, **kwargs):
        
        for individual in X:
            
            n_rand = np.random.rand()  
            
            # há mutação
            if (n_rand < self.prob):
                
                # index da mutação
                len_individual = IndividualUtils.get_lenght_individual(self.config, individual)
                if self.is_dataset_sparse:
                    index = np.random.randint(1, len_individual)
                else:
                    index = np.random.randint(0, len_individual)
                
                # novo valor do gene
                value = np.random.randint(0, self.config.get_seed())
                individual[index] = value
                
        return X