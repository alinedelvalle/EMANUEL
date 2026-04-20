import numpy as np

import configuration.SLC as config_SLC
import configuration.SLCEnsemble as config_SLC_ensemble

import configuration.MLC as config_MLC
import configuration.MLCEnsemble as config_MLC_ensemble


class Configuration:
    
    def __init__(self, n_f, n_l):
        self.n_features = n_f # feature
        self.n_labels = n_l # label
    

    # MLC e ensemble MLC
    def get_ml_algorithms(self):
        return np.concatenate([config_MLC_ensemble.get_algorithms(self.n_features, self.n_labels),
                               config_MLC.get_MLC_algorithms(self.n_features, self.n_labels)])
    
    
    # dicionários de configuração
    def get_sl_config(self):
        return config_SLC.get_config(self.n_features, self.n_labels)
    
    
    def get_sl_kernel_config(self):
        return config_SLC.get_kernels_smo()
    
    
    def get_ml_config(self):
        return config_MLC.get_config(self.n_features, self.n_labels) 
    
    
    def get_sl_ensemble_config(self):
        return config_SLC_ensemble.get_config(self.n_features, self.n_labels)
    

    def get_ml_ensemble_config(self):
        return config_MLC_ensemble.get_config(self.n_features, self.n_labels) 
    
    
    def get_all_config(self):
        dict_config = {}
        dict_config.update(self.get_sl_config())
        dict_config.update(self.get_sl_ensemble_config())
        dict_config.update(self.get_ml_config())
        dict_config.update(self.get_ml_ensemble_config())
        return dict_config
    
    # obtém o tamanho da maior lista de hiperparâemtros dos espaço de busca
    def get_seed(self):
        dict_config = self.get_all_config()
                                          
        maxi = 0
        for alg in dict_config.keys():
            dict_config_alg = dict_config.get(alg)
            for list_hips in dict_config_alg.values():
                if isinstance(list_hips, np.ndarray):
                    if len(list_hips) > maxi:
                        maxi = len(list_hips)
        
        return maxi