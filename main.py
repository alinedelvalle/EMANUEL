import os
import time
import pathlib
import argparse
from pytictoc import TicToc

import numpy as np
import pandas as pd

from configuration.Configuration import Configuration
from multiobjective.ml_problem import MLProblem
from multiobjective.ml_sampling import MLSampling
from multiobjective.ml_mutation import MLMutation
from multiobjective.individual_utils import IndividualUtils

from utils.graphic import Graphic
from utils.pareto_froint import Point, FNDS
from utils.manipulate_history import ManipulateHistory
from utils.evaluate_froin_points import EvaluateMLC

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.termination import TerminateIfAny
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.max_time import TimeBasedTermination


# ----------------
# File and dir
# ----------------
def create_dir(destination, name_dir):    
    cmd = 'if [ ! -d '+destination+'/'+name_dir+' ]\nthen\n\tmkdir '+destination+'/'+name_dir+'\nfi'
    os.system(cmd)

# -----------------
# Results by fold
# -----------------
def get_results_by_fold(problem, res, k, path_res):
    # Gráfico - objectives space
    Graphic.plot_scatter(res.F[:, 0], res.F[:, 1], 'Objective Space', '- Macro F-score', 'Model Size', path_res+'/ObjectiveSpace'+str(k)+'.png')
    
    # Gráfico - hypervolume
    ref_point = np.array([0, 1e9])
    n_evals, hist_F, hv = ManipulateHistory.get_hypervolume(res, ref_point)
    Graphic.plot_graphic(n_evals, hv, 'Convergence-Hypervolume', 'Evaluations', 'Hypervolume', path_res+'/Hypervolume'+str(k)+'.png')    
    
    # Prepare results for saving to a file
    output_data = f'Execution time:{res.exec_time}\n'
      
    output_data += f'Best solution found:\n'
    for individual in res.X:
        output_data += f'{individual}\n'
     
    output_data += 'Classifiers:\n'
    for individual in res.X:
        is_normalize, meka_command, weka_command = IndividualUtils.get_commands(config, individual)
        output_data += f"Normalize:{is_normalize}\n{meka_command}\n{weka_command}\n"
        
    output_data += f"Function value:\n{res.F}\n"
    
    list_f1 = [] # -f1
    list_f2 = [] # model size
    for l in res.F:
        list_f1 = np.append(list_f1, l[0])
        list_f2= np.append(list_f2, l[1])
    
    # f1
    output_data += 'F1 (macro averaged by label)\n['
    for i in range(len(list_f1)-1):
        output_data += str(-list_f1[i])+','
    output_data += str(-list_f1[len(list_f1)-1])+']\n'
     
    # size
    output_data += 'Model size\n['
    for i in range(len(list_f2)-1):
        output_data += str(list_f2[i])+','
    output_data += str(list_f2[len(list_f2)-1])+']\n'
    
    # Evaluation
    output_data += 'Evaluation:\n['
    for i in range(len(n_evals)-1):
        output_data += str(n_evals[i])+','
    output_data += str(n_evals[len(n_evals)-1])+']\n'
    
    # Hypervolume
    output_data += 'Hypervolume:\n['
    for i in range(len(hv)-1):
        output_data += str(hv[i])+','
    output_data += str(hv[len(hv)-1])+']\n'
    
    # History
    output_data += 'History:\n'
    for array in hist_F:
        if len(array) == 1:
            output_data += str(array[0][0])+', '+str(array[0][len(array[0])-1])+'\n'
        else:
            for i in range(len(array)-1):
                output_data += str(array[i][0])+', '+str(array[i][len(array[i])-1])+', '
            output_data += str(array[i+1][0])+', '+str(array[i+1][len(array[i+1])-1])+'\n'
    
    # Save results
    output_path = pathlib.Path(f'{path_res}/results'+str(k)+'.txt')
    output_path.write_text(output_data)
# -----


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_run', type=int, default=3, help='Execution ID')
    parser.add_argument('--n_labels', type=int, default=7, help='Labels number')
    parser.add_argument('--n_features', type=int, default=19, help='Features number')
    parser.add_argument('--is_dataset_sparse', action='store_true', default=False, help='Is the dataset sparse?') # true: --is_dataset_sparse
    parser.add_argument('--name_dataset', type=str, default='flags', help='Dataset name')
    parser.add_argument('--meka_classpath', type=str, default='./meu-meka-1.9.8/lib/', help='Meka path')
    parser.add_argument('--java_command', type=str, default='YOUR_PATH/bin/java', help='Java path')
    parser.add_argument('--project', type=str, default='YOUR_PATH/emanuel_git', help='Project  path')
    parser.add_argument('--k_folds', type=int, default=3, help='k-fold')
    parser.add_argument('--n_process', type=int, default=3, help='Process number')
    parser.add_argument('--limit_mlc_time', type=int, default=15, help='Runtime limit for the MLC algorithm')
    parser.add_argument('--len_population', type=int, default=6, help='Population size')
    parser.add_argument('--number_generation', type=int, default=50, help='Generation number')
    parser.add_argument('--max_time', type=int, default=300, help='Maximum runtime of EMANUEL') # seconds
    args = parser.parse_args() 


    # default
    n_gene = 28 # len individual
    termination_period = 10
    termination_tol = 0.001 # threshold
    skip = 0
    
    # datasets
    path_dataset = f"{args.project}/datasets/{args.name_dataset}"

    # cache - Stores the result of each mlc algorithm.
    path_cache = f'{args.project}/cache'
    
    # configuration
    config = Configuration(args.n_features, args.n_labels)

    for run in range(1, args.n_run+1):
        #start = time.time()
        timer = TicToc()
        timer.tic()

        print(f'\n\nRun: {str(run)}\n')

        # log
        log_file = f"{args.project}/log/log_{args.name_dataset}_{str(run)}.txt"

        # results
        create_dir(f"{args.project}/results/{args.name_dataset}", f"{args.name_dataset}{str(run)}")
        path_res = f"{args.project}/results/{args.name_dataset}/{args.name_dataset}{str(run)}"
        create_dir(path_res, "predictions")
        path_res_pred = f"{path_res}/predictions"

        # metrics
        metrics_file = f"{path_res}/metrics.csv"
    
        # run for k-folds
        all_points = []
        for k in range(args.k_folds):
            problem = MLProblem(
                config, 
                args.java_command, 
                args.meka_classpath, 
                k, 
                args.n_process, 
                args.limit_mlc_time, 
                args.name_dataset, 
                path_dataset, 
                args.n_labels, 
                log_file, 
                metrics_file,
                path_cache
            )

            algorithm = NSGA2(
                pop_size=args.len_population,
                sampling=MLSampling(config, n_gene, args.is_dataset_sparse),
                crossover=UniformCrossover(prob=0.5),
                mutation=MLMutation(0.05, config, args.is_dataset_sparse),
                eliminate_duplicates=NoDuplicateElimination()
            )
            
            # Termination: maximum number of generations, 'tol' tolerance per 'period' generation, time
            termination = TerminateIfAny(
                MaximumGenerationTermination(args.number_generation), 
                RobustTermination(MultiObjectiveSpaceTermination(tol=termination_tol, n_skip=skip), period=termination_period),
                TimeBasedTermination(max_time=args.max_time//args.k_folds) 
            )
            
            res = minimize(
                problem,
                algorithm,
                termination,
                save_history=True,
                verbose=True
            )    

            print(res.F)

            # saves fold results
            get_results_by_fold(problem, res, k, path_res)

            # find unique solutions - normalize, MEKA, WEKA
            unique_algs = {}
            for i in range (len(res.F)):
                macro_f1, model_size = res.F[i]
                individual = res.X[i]
                is_normalize, meka_command, weka_command = IndividualUtils.get_commands(config, individual)
                pto = Point(macro_f1, model_size, is_normalize, meka_command, weka_command, k)
                # unique algorithms
                cmd = f'{is_normalize}, {meka_command}, {weka_command}'
                unique_algs[cmd] = pto

            # get a unique points list
            list_points = list(unique_algs.values())

            # run the MLC algorithms in the fold k (train/test) 
            emlc = EvaluateMLC(args.java_command, args.meka_classpath, k, args.n_process, args.name_dataset, path_dataset, args.is_dataset_sparse, args.n_labels)
            list_points_test = emlc.calc_points_test(list_points)
            all_points.extend(list_points_test)

        # pareto froint
        fnds = FNDS()
        froint_test = fnds.execute(all_points)
        fnds.plot_froint('Objective Space', '- Macro F-score', 'Model Size', f'{path_res}/ObjectiveSpace.png')  

        # runtime
        runtime = timer.tocvalue()

        # hypervolume
        ref_point= np.array([0, 1e9])
        hv = fnds.get_hypervolume(ref_point)

        # save test froint
        id=0
        df_froint_test = pd.DataFrame(columns=['k', 'normalize', 'meka', 'weka', 'macro_fscore', 'model_size', 'hypervolume', 'runtime'])
        for pto in froint_test:
            df_froint_test.loc[df_froint_test.shape[0]] = [pto.k, pto.norm, pto.meka, pto.weka, -pto.obj1, pto.obj2, hv, runtime]
            df_predictions = pd.DataFrame(pto.predictions)
            df_predictions.to_csv(f'{path_res_pred}/pred{id}.csv', index=False)
            id+=1
        df_froint_test.to_csv(f'{path_res}/result.csv')

        # ---------------------------------------------------------------------
        # NOTE
        # The final boundary is in results.csv
        # For each row in this csv, there is a csv in predictions,
        # pred0.csv are the predictions for row 0 (first row of results.csv) 
        # -----------------------------------------------------------------------
