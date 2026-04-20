import numpy as np

from pymoo.indicators.hv import Hypervolume


class ManipulateHistory:
    
    
    def get_hist_F(res):
        hist = res.history
    
        n_evals = []  # corresponding number of function evaluations\
        hist_F = []   # the objective space values in each generation

        for algorithm in hist:
            # store the number of function evaluations
            n_evals.append(algorithm.evaluator.n_eval)
        
            # retrieve the optimum from the algorithm
            opt = algorithm.opt
        
            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])
            
        return (n_evals, hist_F)
    
    
    def get_hypervolume(res, point):
        # Histórico de execução (número de avaliações e funções objetivos)
        n_evals, hist_F = ManipulateHistory.get_hist_F(res)
            
        metric = Hypervolume (
            ref_point = point,
        )

        hv = [metric.do(_F) for _F in hist_F]
            
        return (n_evals, hist_F, hv)
    
    
