import sys
import time
from datetime import datetime
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import torch

import threading

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge

from config import *
from sklearn_DKL_GP import Sklearn_DKL_GP
# from skopt.plots import plot_gaussian_process, plot_evaluations, plot_convergence, plot_objective, plot_regret
from skopt import Optimizer, OptimizerMultiobj, Space
from skopt.space import Integer

from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern


from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from simulation_metrics import CPI_metric, power_metric, evaluate_ADRS

n_experiment = 1
print_info = False
mape_line_analysis = True

if smoke_test:
    surrogate_model_tag_list = ["smoke_test"]
    n_experiment = 1
elif "SC-202005121725" == hostname:
    # bookpad
    surrogate_model_tag_list = ["BOOM-Explorer"] #["ActBoost", "SemiBoost", "GBRT"]
    mape_line_analysis = True
    n_experiment = 1
    print_info = False
elif 2 < len(sys.argv):
    # desktop
    surrogate_model_tag_list = [
        #"ASPLOS06",
        #"BOOM-Explorer",
        #"AdaBoost_MLP",
        #"ActBoost",
        # "CatBoostRegressor",
        # "LGBMRegressor",
        # "XGBoost",
        #"GBRT",
        #"M5P",
        #"GBRT-orh",
        #"AdaGBRT-no-iter",
        #"AdaGBRT-base",
        #"AdaGBRT",
        #"AdaGBRT-cv",
        #"SVR_Matern",
        #"SemiBoost",
        "BagGBRT",
    ]
    n_experiment = 10
    print_info = True
else:
    surrogate_model_tag_list = ["BOOM-Explorer"]
    mape_line_analysis = True
    n_experiment = 1
    print_info = True


def to_named_params(results, search_space):
    params = results.x
    param_dict = {}
    params_list = [(dimension.name, param) for dimension, param in zip(search_space, params)]
    for item in params_list:
        param_dict[item[0]] = item[1]
    return param_dict

#transform = 'normalize'
transform = 'identity'
if 2304 == N_SPACE_SIZE:
    problem_space = Space([Integer(low=0, high=3, prior='uniform', transform=transform, name="dispatch_width"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_int"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_fp"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="load_queue"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Dcache"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Icache"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="BP"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="L2cache")])
elif 9216==N_SPACE_SIZE:
    problem_space = Space([Integer(low=0, high=3, prior='uniform', transform=transform, name="dispatch_width"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_int"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_fp"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="load_queue"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Dcache"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Icache"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="BP"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="L2cache"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="decode_width")])
elif 36864==N_SPACE_SIZE:
    problem_space = Space([Integer(low=0, high=3, prior='uniform', transform=transform, name="dispatch_width"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_int"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="exe_fp"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="load_queue"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Dcache"),
                           Integer(low=0, high=2, prior='uniform', transform=transform, name="L1Icache"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="BP"),
                           Integer(low=0, high=1, prior='uniform', transform=transform, name="L2cache"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="decode_width"),
                           Integer(low=0, high=3, prior='uniform', transform=transform, name="rob")])
else:
    print(f"no def for N_SPACE_SIZE={N_SPACE_SIZE}")


def objective(x):
    return CPI_metric(x)


def objective2(x):
    return power_metric(x)


# class Problem_Model(threading.Thread):
class Problem_Model():

    def __init__(self, surrogate_model_config, n=1):
        # threading.Thread.__init__(self)
        self.num = n
        self.surrogate_model_config = surrogate_model_config

    def run(self):
        # from simulation_metrics import case_name

        from get_real_pareto_frontier import get_pareto_optimality_from_file_ax_interface
        real_pareto_data = get_pareto_optimality_from_file_ax_interface(case_name)
        # print(f"real_pareto={real_pareto}")
        base_estimator, base_estimator2, surrogate_model_tag, surrogate_model_dict = self.surrogate_model_config
        surrogate_model_dict["tag"] += "-exp" + str(self.num)

        if surrogate_model_dict['model_is_iterative']:
            N_INIT = 2 if smoke_test else N_SAMPLES_INIT
            N_ITER = N_SAMPLES_ALL - N_INIT
        else:
            N_INIT = 1 if smoke_test else N_SAMPLES_ALL
            N_ITER = 0

        result_filename_prefix = "log"
        if 2304 != N_SPACE_SIZE:
            result_filename_prefix += "_" + str(N_SPACE_SIZE)
        result_filename_prefix +=  '/' + case_name + "-" + surrogate_model_tag + "-exp-" + str(self.num)

        #if 0 == self.num:
        np.random.seed(1234+self.num)

        opt_gp = OptimizerMultiobj(
            dimensions=problem_space,
            base_estimator=base_estimator,
            base_estimator2=base_estimator if base_estimator2 is None else base_estimator2,
            n_random_starts=None,
            n_initial_points=N_INIT,
            initial_point_generator=surrogate_model_dict['initial_point_generator'],
            n_jobs=-1,
            acq_func=surrogate_model_dict['acq_func'],
            acq_optimizer="full",  # "auto",
            random_state=self.num,
            model_queue_size=1,
            acq_func_kwargs=None,  # {"xi": 0.000001, "kappa": 0.001} #favor exploitaton
            acq_optimizer_kwargs={"n_points": 10},
            real_pareto_data=real_pareto_data,
            surrogate_model_dict=surrogate_model_dict,
            n_generation_points=N_ITER,
            mape_line_analysis=mape_line_analysis,
        )

        startTime = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")
        last_time = 0
        #time_used_dataset_acc = 0
        log_file = open(result_filename_prefix + ".log", "w")
        for iter in range(N_SAMPLES_ALL):
            startTime2 = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")
            next_x = opt_gp.ask()
            #start_dataset_time = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")
            #print(f"main next_x={next_x}")
            f_val = objective(next_x)
            f_val2 = objective2(next_x)
            #time_used_dataset = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S") - start_dataset_time
            #time_used_dataset_acc += time_used_dataset.total_seconds()
            #print(f"time_used_dataset_acc={time_used_dataset_acc}")
            #print(f"before tell {time.localtime()}")
            res = opt_gp.tell(next_x, f_val, f_val2)
            #print(f"after tell {time.localtime()}")
            if mape_line_analysis and (surrogate_model_dict['model_is_iterative'] is False):
                # model is not iterative, here should only count in the time of this iteration
                time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                          "%Y-%m-%d %H:%M:%S") - startTime2
            else:
                time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                          "%Y-%m-%d %H:%M:%S") - startTime
            #start_time_io = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")                
            if surrogate_model_dict['model_is_iterative'] and print_info and (N_SAMPLES_INIT < (iter + 2)):
                print(f"sample={iter + 1} ADRS= {res.fun} [{time_used}]")
            log_file.write(
                f"sample= {iter + 1} ADRS= {res.fun} "
                f"predict_error= {res.predict_error} predict_error_unlabelled_mape= {res.predict_error_unlabelled_mape} "
                f"non_uniformity= {res.non_uniformity} "
                f"coverage = {res.coverage}"
                f"time_used= {time_used}\n")
            #time_io_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),"%Y-%m-%d %H:%M:%S") - start_time_io
            #time_used_io_acc += time_io_used.total_seconds()

            if init_effect_analysis and (N_SAMPLES_INIT == (iter + 1)):
                global result_init_alg
                result_init_alg[self.num] = res.fun
                if 0 == self.num and "random" != surrogate_model_dict['initial_point_generator'] and "sobol" != surrogate_model_dict['initial_point_generator']:
                    save_result_init_effect(surrogate_model_tag_real=surrogate_model_tag, surrogate_model_dict=surrogate_model_dict)

            if mape_line_analysis and (N_SAMPLES_INIT < (iter + 2)):
                result_array_samples[iter + 1][self.num] = res.fun
                result_time_array_samples[iter + 1][self.num] = time_used.total_seconds()
                # print(f"predict_error={res.predict_error}")
                result_mape_array1_samples[iter + 1][self.num] = res.predict_error[0]
                result_mape_array2_samples[iter + 1][self.num] = res.predict_error[1]
                result_unlabelled_mape_array1_samples[iter + 1][self.num] = res.predict_error_unlabelled_mape[0]
                result_unlabelled_mape_array2_samples[iter + 1][self.num] = res.predict_error_unlabelled_mape[1]
                result_non_uniformity_samples[iter + 1][self.num] = res.non_uniformity
                result_coverage[iter + 1][self.num] = res.coverage
                result_hv[iter + 1][self.num] = opt_gp.statistics['hv'][-1]
                result_hv_acq_stuck[iter + 1][self.num] = opt_gp.statistics['hv_acq_stuck']
                result_hv_last_iter[iter + 1][self.num] = opt_gp.statistics['hv_last_iter']
            if (N_SAMPLES_INIT < (iter + 2)):
                result_array_sample_map['time_train'][iter + 1][self.num] = res.result_array_sample_map['time_train']
                result_array_sample_map['time_infer'][iter + 1][self.num] = res.result_array_sample_map['time_infer']
                result_array_sample_map['time_explore'][iter + 1][self.num] = res.result_array_sample_map['time_explore']
                result_array_sample_map['time_pareto'][iter + 1][self.num] = res.result_array_sample_map['time_pareto']
                result_array_sample_map['time_adrs'][iter + 1][self.num] = res.result_array_sample_map['time_adrs']
                result_array_sample_map['time_hv'][iter + 1][self.num] = res.result_array_sample_map['time_hv']
                result_array_sample_map['time_filter'][iter + 1][self.num] = res.result_array_sample_map['time_filter']                
                result_array_sample_map['filter_ratio'][iter + 1][self.num] = res.result_array_sample_map['filter_ratio']
                result_array_sample_map['time_pred_pareto'][iter + 1][self.num] = res.result_array_sample_map['time_pred_pareto']
                result_array_sample_map['time_register'][iter + 1][self.num] = res.result_array_sample_map['time_register']
                result_array_sample_map['time_valid'][iter + 1][self.num] = result_time_array_samples[iter + 1][self.num] - np.sum(result_array_sample_map['time_register'][:iter + 2][self.num])
                #result_array_sample_map['time_io_acc'][iter + 1][self.num] = time_used_io
                #result_array_sample_map['time_dataset_acc'][iter + 1][self.num] = time_used_dataset_acc


            # if iter >= 5:
            #    plot_optimizer(res, n_iter=iter - 5, max_iters=5)

        time_used = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                      "%Y-%m-%d %H:%M:%S") - startTime
        print(result_filename_prefix)
        final_result = opt_gp.get_result()
        global result_array
        global result_time_array
        global result_mape_array1, result_mape_array2
        global result_unlabelled_mape_array1, result_unlabelled_mape_array2
        result_array[self.num] = final_result.fun
        if mape_line_analysis:
            result_time_array[self.num] = result_time_array_samples[N_SAMPLES_ALL][self.num]
        else:
            result_time_array[self.num] = time_used.total_seconds()
        result_mape_array1[self.num] = final_result.predict_error[0]
        result_mape_array2[self.num] = final_result.predict_error[1]
        result_unlabelled_mape_array1[self.num] = final_result.predict_error_unlabelled_mape[0]
        result_unlabelled_mape_array2[self.num] = final_result.predict_error_unlabelled_mape[1]        
        log_file.write(f"time_used={time_used}\n")
        log_file.write(f"result={final_result}")
        print(f"adrs= {result_array[self.num]}")
        print(f"time_used={time_used}")
        print(f"[OUT] semi_train = {opt_gp.statistics['semi_train_accumulation']} "
              f"ucb = {opt_gp.statistics['ucb']} "
              f"cv_ranking_beta= {opt_gp.statistics['cv_ranking_beta']} "
              f"non_uniformity_explore = {opt_gp.statistics['non_uniformity_explore']} ")
        log_file.write(f"[OUT] semi_train = {opt_gp.statistics['semi_train_accumulation']} "
              f"ucb = {opt_gp.statistics['ucb']} "
              f"cv_ranking_beta= {opt_gp.statistics['cv_ranking_beta']} "
              f"non_uniformity_explore = {opt_gp.statistics['non_uniformity_explore']} \n")
        #print(f"[OUT] non_uniformity real / last = {opt_gp.real_pareto_data_non_uniformity} / {opt_gp.non_uniformitys[-1]}")
        #log_file.write(f"[OUT] non_uniformity real / last = {opt_gp.real_pareto_data_non_uniformity} / {opt_gp.non_uniformitys[-1]} \n")
        log_file.close()

        save_feature_importances(result_filename_prefix, opt_gp.feature_importances)
        if False:
            fig = plt.figure()
            fig.suptitle(surrogate_model)
            plot_convergence(opt_gp.get_result())
            plt.plot()
            plt.show()
        if False:
            plot_objective(opt_gp.get_result(), n_points=10)
            # plot_regret(opt_gp.get_result())
            plt.show()


def save_feature_importances(result_filename_prefix, feature_importances):
    if feature_importances is not None:
        feature_importances_file = open("log_summary/feature_importances.txt", "a")
        feature_importances_file.write(result_filename_prefix + " ")
        for each in feature_importances[0]:
            feature_importances_file.write(str(each) + " ")
        for each in feature_importances[1]:
            feature_importances_file.write(str(each) + " ")
        feature_importances_file.write("\n")
        feature_importances_file.close()


def get_surrogate_model(surrogate_model_tag):
    base_estimator = base_estimator2 = None
    surrogate_model_tag_real = surrogate_model_tag
    surrogate_model_dict = {}

    surrogate_model_dict['model_is_iterative'] = True
    surrogate_model_dict['semi_train'] = False
    surrogate_model_dict['semi_train_adapt'] = False
    surrogate_model_dict['kernel_train'] = False
    surrogate_model_dict['different_model'] = False
    surrogate_model_dict['acq_func'] = 'HVI'
    surrogate_model_dict['cv_ranking'] = 'no'
    surrogate_model_dict['cv_pool_size'] = 100
    surrogate_model_dict['ucb'] = 0.0
    surrogate_model_dict['ucb_v'] = 0
    surrogate_model_dict['warm_start'] = False
    surrogate_model_dict['labeled_predict_mode'] = False
    surrogate_model_dict['semi_train_iter_max'] = 10
    surrogate_model_dict['predict_last'] = True
    surrogate_model_dict['cv_ranking_beta'] = 0
    surrogate_model_dict['cv_ranking_beta_v'] = 0
    surrogate_model_dict['non_uniformity_explore'] = 0
    surrogate_model_dict['filter'] = 0

    surrogate_model_dict['hv_scale_v'] = 0

    # 'sobol', 'lhs', 'halton', 'hammersly','random', or 'grid'
    surrogate_model_dict['initial_point_generator'] = "random"
    # surrogate_model_dict['initial_point_generator'] = "orthogonal"

    if "smoke_test" == surrogate_model_tag:
        from sklearn.ensemble import AdaBoostRegressor
        hidden_layer_1 = 2
        dt_stump = MLPRegressor(hidden_layer_sizes=(hidden_layer_1),
                                max_iter=2,
                                solver='sgd',  # ['adam', 'sgd', 'lbfgs'],
                                activation='relu',
                                )
        # dt_stump.fit(train_X, train_Y)
        base_estimator = AdaBoostRegressor(
            base_estimator=dt_stump,
            learning_rate=0.001,
            n_estimators=4,
        )
        #surrogate_model_dict['semi_train'] = 1
    elif "smoke_test2" == surrogate_model_tag:
        kernel = Matern()
        if False:
            for hyperparameter in kernel.hyperparameters:
                print(hyperparameter)
            params = kernel.get_params()
            for key in sorted(params): print(f"{key} : {params[key]}")
        # noise_level = 0.0  0.00958
        base_estimator = GaussianProcessRegressor(kernel=kernel,
                                                  # alpha=noise_level **2 ,
                                                  alpha=0.00958 ** 2,
                                                  normalize_y=True,
                                                  # noise="gaussian",
                                                  n_restarts_optimizer=100
                                                  )
    elif "PolyLinear" == surrogate_model_tag:
        base_estimator = Pipeline([
            ("poly", PolynomialFeatures(degree=3)),
            ("std_scaler", StandardScaler()),
            ("lin_reg", LinearRegression())
        ])
        surrogate_model_dict['model_is_iterative'] = False
    elif "Ridge" == surrogate_model_tag:
        base_estimator = KernelRidge(kernel="rbf")
        surrogate_model_dict['model_is_iterative'] = False
    elif "LGBMQuantileRegressor" == surrogate_model_tag:
        from LGBMQuantileRegressor import LGBMQuantileRegressor

        base_estimator = LGBMQuantileRegressor()
        # base_estimator.fit(train_X, train_Y)
        # predict_value_R2 = base_estimator.score(train_X, train_Y)
        # print(f"R2={predict_value_R2}")
        # predict_value = base_estimator.predict(train_X)
        # print(f"MSE={mean_squared_error(predict_value, train_Y)}")
    elif "SVR_Matern" == surrogate_model_tag:
        kernel = 1.0 * Matern(length_scale=3.0, length_scale_bounds=(1e-1, 20.0), nu=2.5)
        kernel2 = 1.0 * Matern(length_scale=3.0, length_scale_bounds=(1e-1, 20.0), nu=2.5)
        base_estimator = SVR(kernel=kernel, degree=8, )
        base_estimator2 = SVR(kernel=kernel2, degree=1, )
        surrogate_model_dict['model_is_iterative'] = False
        surrogate_model_dict['different_model'] = True
    elif "MLP" == surrogate_model_tag:
        HBO_params_cpi = {'learning_rate_init': 0.01, 'activation': 'logistic', 'solver': 'lbfgs'} 
        HBO_params_power = {'learning_rate_init': 0.02, 'activation': 'logistic', 'solver': 'lbfgs'} 
        base_estimator = MLPRegressor(hidden_layer_sizes=(16, 32, 32),
                                      solver='lbfgs',  # ['adam', 'sgd', 'lbfgs'],
                                      activation='relu',
                                      max_iter=10000,
                                      )
        base_estimator = MLPRegressor(hidden_layer_sizes=(16, 32, 32),
                                      **HBO_params_cpi,
                                      max_iter=10000,
                                      )
        base_estimator2 = MLPRegressor(hidden_layer_sizes=(16, 32, 32),
                                       solver='lbfgs',  # ['adam', 'sgd', 'lbfgs'],
                                       activation='relu',
                                       max_iter=10000,
                                       # verbose=True,
                                       )
        base_estimator2 = MLPRegressor(hidden_layer_sizes=(16, 32, 32),
                                      **HBO_params_power,
                                      max_iter=10000,
                                      )            
        # print(f"base_estimator={base_estimator}")
        surrogate_model_dict['model_is_iterative'] = False
    elif "ASPLOS06" == surrogate_model_tag:
        base_estimator = MLPRegressor(hidden_layer_sizes=(16),
                                      solver='sgd',  # ['adam', 'sgd', 'lbfgs'],
                                      activation='relu',
                                      max_iter=10000,
                                      )
        surrogate_model_dict['model_is_iterative'] = False
        #surrogate_model_tag_real += "_pred"
        # surrogate_model_dict['initial_point_generator'] = "hammersly"
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
    elif "GP" == surrogate_model_tag:
        '''
        1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
        1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
        1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                            length_scale_bounds=(0.1, 10.0),
                            periodicity_bounds=(1.0, 10.0)),
        ConstantKernel(0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
        '''            
        #kernel = 1.0 * Matern()
        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 20.0), nu=2.5)
        kernel2 = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 20.0), nu=1.5)
        #noise_level = 0.00958
        HBO_params_cpi = {'n_restarts_optimizer': 2, 'kernel__k2__nu': 2.5, 'normalize_y': False} 
        HBO_params_power = {'n_restarts_optimizer': 9, 'kernel__k2__nu': 1.5, 'normalize_y': False} 
        base_estimator = GaussianProcessRegressor(kernel=kernel,
                                                  #alpha=noise_level ** 2,
                                                  normalize_y=False,
                                                  # noise="gaussian",
                                                  n_restarts_optimizer=2
                                                  )
        base_estimator2 = GaussianProcessRegressor(kernel=kernel2,
                                                  #alpha=noise_level ** 2,
                                                  normalize_y=False,
                                                  # noise="gaussian",
                                                  n_restarts_optimizer=9
                                                  )
        surrogate_model_tag_real += "_Matern"
        # surrogate_model_dict['acq_func'] = 'EHVI'
        surrogate_model_dict['model_is_iterative'] = False
    elif "BOOM-Explorer" == surrogate_model_tag:
        #kernel = Sklearn_DKL_GP(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=2.5)
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=2.5)
        if False:
            for hyperparameter in kernel.hyperparameters:
                print(hyperparameter)
            params = kernel.get_params()
            for key in sorted(params): print(f"{key} : {params[key]}")
        base_estimator = GaussianProcessRegressor(kernel=kernel,
                                                  # alpha=noise_level ** 2,
                                                  #alpha=0.00958,
                                                  normalize_y=True,
                                                  # noise="gaussian",
                                                  n_restarts_optimizer=8
                                                  )
        #kernel2 = Sklearn_DKL_GP(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=2.5)
        kernel2 = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=2.5)
        base_estimator2 = GaussianProcessRegressor(kernel=kernel2,
                                                   # alpha=noise_level ** 2,
                                                   #alpha=0.00958,
                                                   normalize_y=False,
                                                   # noise="gaussian",
                                                   n_restarts_optimizer=6
                                                   )
        #surrogate_model_tag_real += "_DKL_GP"
        #surrogate_model_dict['kernel_train'] = True
        # surrogate_model_dict['acq_func'] = 'EHVI'
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
        #surrogate_model_dict['initial_point_generator'] = "kmeans"
        #surrogate_model_dict['initial_point_generator'] = "distance"
        #surrogate_model_dict['model_is_iterative'] = True
        if 1:
            surrogate_model_dict['ucb'] = 0.05
            surrogate_model_dict['ucb_v'] = 12
        if 1:
            surrogate_model_dict['filter'] = 2
            #surrogate_model_tag_real += '_fifix-5' should modify code
        surrogate_model_tag_real += '_v6'
    elif "RF_custom" == surrogate_model_tag:
        HBO_params_cpi = {'n_estimators': 51, 'max_depth': 11, 'min_samples_leaf': 1} 
        HBO_params_power = {'n_estimators': 52, 'max_depth': 30, 'min_samples_leaf': 1} 
        base_estimator = RandomForestRegressor(**HBO_params_cpi)
        base_estimator = RandomForestRegressor(**HBO_params_power)
        surrogate_model_dict['model_is_iterative'] = False
    elif "RF" == surrogate_model_tag:
        base_estimator = "RF"
    elif "ET_custom" == surrogate_model_tag:
        HBO_params_cpi = {'n_estimators': 199, 'max_depth': 24, 'min_samples_leaf': 1, 'max_features': 'auto'}
        HBO_params_power = {'n_estimators': 197, 'max_depth': 13, 'min_samples_leaf': 1, 'max_features': 'auto'}
        base_estimator = ExtraTreesRegressor(**HBO_params_cpi,n_jobs=2)
        base_estimator2 = ExtraTreesRegressor(**HBO_params_power, n_jobs=2)
        surrogate_model_dict['model_is_iterative'] = False
    elif "ET" == surrogate_model_tag:
        base_estimator = "ET"
    elif "AdaBoost_DTR" == surrogate_model_tag:
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import AdaBoostRegressor

        #dt_stump = DecisionTreeRegressor(max_depth=8)
        # dt_stump.fit(train_X, train_Y)
        HBO_params_cpi = {'n_estimators': 168, 'learning_rate': 0.001, 'base_estimator': DecisionTreeRegressor(max_depth=8)} 
        HBO_params_power = {'n_estimators': 85, 'learning_rate': 0.001, 'base_estimator': DecisionTreeRegressor(max_depth=8)} 
        base_estimator = AdaBoostRegressor(
            #base_estimator=dt_stump,
            **HBO_params_cpi,
        )
        base_estimator2 = AdaBoostRegressor(
            #base_estimator=dt_stump,
            **HBO_params_power,
        )            
        surrogate_model_dict['model_is_iterative'] = False
    elif "AdaBoost_MLP" == surrogate_model_tag:
        from sklearn.ensemble import AdaBoostRegressor

        hidden_layer_1 = 16
        hidden_size = (hidden_layer_1, hidden_layer_1 * 2, hidden_layer_1 * 2)
        dt_stump = MLPRegressor(hidden_layer_sizes=hidden_size, max_iter=10000, )
        # dt_stump.fit(train_X, train_Y)
        HBO_params_cpi = {'n_estimators': 100, 'learning_rate': 0.005} 
        HBO_params_power = {'n_estimators': 100, 'learning_rate': 0.01}
        base_estimator = AdaBoostRegressor(
            base_estimator=dt_stump,
            **HBO_params_cpi,
        )
        base_estimator2 = AdaBoostRegressor(
            base_estimator=dt_stump,
            **HBO_params_power,
        )
        #surrogate_model_dict['initial_point_generator'] = "orthogonal"
        surrogate_model_dict['model_is_iterative'] = False
    elif "ActBoost" == surrogate_model_tag:
        from sklearn.ensemble import AdaBoostRegressor
        hidden_size = (8, 6)
        dt_stump = MLPRegressor(hidden_layer_sizes=hidden_size, max_iter=10000, )
        # dt_stump.fit(train_X, train_Y)
        base_estimator = AdaBoostRegressor(
            base_estimator=dt_stump,
            learning_rate=0.001,
            n_estimators=2 * 10,
        )
        surrogate_model_dict['cv_ranking'] = 'maxsort'
        surrogate_model_dict['acq_func'] = 'cv_ranking'
        surrogate_model_tag_real += '_v4'
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
        surrogate_model_dict['model_is_iterative'] = True
    elif "SemiBoost" == surrogate_model_tag:
        from sklearn.ensemble import AdaBoostRegressor
        hidden_layer_1 = 8
        surrogate_model_dict['warm_start'] = False
        dt_stump = MLPRegressor(hidden_layer_sizes=(hidden_layer_1, hidden_layer_1),
                                max_iter=10000,
                                solver='sgd',  # ['adam', 'sgd', 'lbfgs'],
                                activation='relu',
                                warm_start=surrogate_model_dict['warm_start'],
                                )
        # dt_stump.fit(train_X, train_Y)
        base_estimator = AdaBoostRegressor(
            base_estimator=dt_stump,
            learning_rate=0.001,
            n_estimators=20,
        )
        surrogate_model_dict['cv_ranking'] = 'minsort'
        surrogate_model_dict['model_is_iterative'] = False
        surrogate_model_dict['semi_train'] = 1
        surrogate_model_dict['semi_train_iter_max'] = 50
        surrogate_model_dict['initial_point_generator'] = "lhs"
    elif "M5P" == surrogate_model_tag:
        #surrogate_model_tag_real += '_v2'
        from model_tree.src.ModelTree import ModelTree
        from model_tree.models.linear_regr import linear_regr
        model = linear_regr()
        base_estimator = ModelTree(model, max_depth=4, min_samples_leaf=10, search_type="greedy", n_search_grid=100)
        base_estimator2 = base_estimator
        surrogate_model_dict['different_model'] = True
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
        surrogate_model_dict['model_is_iterative'] = False            
    elif "GBRT-base" == surrogate_model_tag:
        # base_estimator = "GBRT"
        if 36864 == N_SPACE_SIZE:
            HBO_params_cpi = {'n_estimators': 139, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.6}
            HBO_params_power = {'n_estimators': 199, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.6}            
        else:
            HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
            HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        # base_estimator = GradientBoostingRegressor(loss="squared_error",
        #                                                       n_estimators=73,
        #                                                       learning_rate=0.1,
        #                                                       max_depth=19,
        #                                                       subsample=0.5,
        #                                                       )
        base_estimator = GradientBoostingRegressor(**HBO_params_cpi)
        base_estimator2 = GradientBoostingRegressor(**HBO_params_power)
        surrogate_model_dict['different_model'] = True
        surrogate_model_dict['model_is_iterative'] = False
        surrogate_model_tag_real += '_v2'
    elif "GBRT-orh" == surrogate_model_tag:
        surrogate_model_tag_real += '_MART'
        # base_estimator = "GBRT"
        if 36864 == N_SPACE_SIZE:
            HBO_params_cpi = {'n_estimators': 139, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.6}
            HBO_params_power = {'n_estimators': 199, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.6}            
        else:
            HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
            HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        # base_estimator = GradientBoostingRegressor(loss="squared_error",
        #                                                       n_estimators=73,
        #                                                       learning_rate=0.1,
        #                                                       max_depth=19,
        #                                                       subsample=0.5,
        #                                                       )
        base_estimator = GradientBoostingRegressor(**HBO_params_cpi)
        base_estimator2 = GradientBoostingRegressor(**HBO_params_power)
        surrogate_model_dict['different_model'] = True
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
        surrogate_model_dict['model_is_iterative'] = False
        surrogate_model_tag_real += '_v2'
    elif "GBRT" == surrogate_model_tag:
        if 36864 == N_SPACE_SIZE:
            HBO_params_cpi = {'n_estimators': 139, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.6}
            HBO_params_power = {'n_estimators': 199, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.6}            
        else:
            HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
            HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        # base_estimator = GradientBoostingRegressor(loss="squared_error",
        #                                                       n_estimators=73,
        #                                                       learning_rate=0.1,
        #                                                       max_depth=19,
        #                                                       subsample=0.5,
        #                                                       )
        base_estimator = GradientBoostingRegressor(**HBO_params_cpi)
        base_estimator2 = GradientBoostingRegressor(**HBO_params_power)
        surrogate_model_dict['different_model'] = True
        #surrogate_model_dict['ucb'] = 0.001
        # surrogate_model_dict['ucb_scale'] = 1.01
        #surrogate_model_dict['semi_train'] = 1
        #surrogate_model_dict['semi_train_iter_max'] = 30
        #surrogate_model_dict['semi_train_adapt'] = True
        if surrogate_model_dict['semi_train']:
            surrogate_model_dict['cv_ranking'] = 'minsort'
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
        surrogate_model_dict['cv_ranking_beta'] = 0.1
        #surrogate_model_dict['acq_func'] = 'EHVI'
        surrogate_model_dict['model_is_iterative'] = False
        surrogate_model_tag_real += '_v2'
    elif "XGBoost" == surrogate_model_tag:
        from xgboost import XGBRegressor
        #HBO_params_cpi = {'n_estimators': 70, 'learning_rate': 0.2, 'max_depth': 48, 'booster': 'gbtree', 'subsample': 0.8}
        HBO_params_cpi = {'n_estimators': 174, 'learning_rate': 0.15, 'max_depth': 5, 'booster': 'dart', 'subsample': 0.8} 
        # HBO_params_power = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 23, 'objective': 'reg:squarederror', 'booster': 'gbtree', 'subsample': 0.5}
        #HBO_params_power = {'n_estimators': 97, 'learning_rate': 0.1, 'max_depth': 30, 'objective': 'reg:squarederror', 'booster': 'gbtree', 'subsample': 0.5}
        HBO_params_power = {'n_estimators': 132, 'learning_rate': 0.2, 'max_depth': 4, 'booster': 'dart', 'subsample': 0.7} 
        base_estimator = XGBRegressor(
            # max_depth=25,
            # learning_rate=0.1,
            # n_estimators=90,
            # objective='reg:squarederror',
            # booster='gbtree',
            **HBO_params_cpi,
            n_jobs=2,
            nthread=None,
        )
        base_estimator2 = XGBRegressor(
            **HBO_params_power,
            n_jobs=2,
            nthread=None,
        )
        surrogate_model_tag_real += '_v3'
        #surrogate_model_dict['initial_point_generator'] = "orthogonal"
        surrogate_model_dict['different_model'] = True
        surrogate_model_dict['model_is_iterative'] = False
    elif "LGBMRegressor" == surrogate_model_tag:
        from lightgbm import LGBMRegressor
        base_estimator = LGBMRegressor(
            n_estimators=71,
            learning_rate=1.0,
            n_jobs=2,
            nthread=None,
        )
        # num_leaves = 31
    elif "CatBoostRegressor" == surrogate_model_tag:
        from catboost import CatBoostRegressor
        HBO_params_cpi = {'n_estimators': 155, 'depth': 10, 'subsample': 1.0, 'early_stopping_rounds': 144,
                          'grow_policy': 'Lossguide'}
        HBO_params_power = {'n_estimators': 200, 'depth': 10, 'subsample': 0.7, 'early_stopping_rounds': 37,
                            'grow_policy': 'Lossguide'}
        base_estimator = CatBoostRegressor(
            # n_estimators=90,
            # depth=6,
            # subsample=0.8,
            # early_stopping_rounds=164,
            **HBO_params_cpi,
            verbose=False,
            thread_count=-1
        )
        base_estimator2 = CatBoostRegressor(
            # n_estimators=198,
            # depth=6,
            # subsample=0.7,
            # early_stopping_rounds=50,
            **HBO_params_power,
            verbose=False,
            thread_count=-1
        )
        surrogate_model_dict['different_model'] = True
    elif "AdaGBRT-no-iter" == surrogate_model_tag:
        HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
        HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        from sklearn.ensemble import AdaBoostRegressor
        HBO_params_ada_cpi = {'n_estimators': 40, 'learning_rate': 0.005}
        HBO_params_ada_power = {'n_estimators': 26, 'learning_rate': 0.01}
        base_estimator = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
            **HBO_params_ada_cpi,
        )
        base_estimator2 = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_power),
            **HBO_params_ada_power,
        )
        surrogate_model_dict['different_model'] = True
        surrogate_model_dict['model_is_iterative'] = False
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
    elif "AdaGBRT-base" == surrogate_model_tag:
        HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
        HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        from sklearn.ensemble import AdaBoostRegressor
        HBO_params_ada_cpi = {'n_estimators': 40, 'learning_rate': 0.005}
        HBO_params_ada_power = {'n_estimators': 26, 'learning_rate': 0.01}
        base_estimator = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
            **HBO_params_ada_cpi,
        )
        base_estimator2 = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_power),
            **HBO_params_ada_power,
        )
        surrogate_model_dict['different_model'] = True
    elif "AdaGBRT" == surrogate_model_tag:
        HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
        HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        from sklearn.ensemble import AdaBoostRegressor
        HBO_params_ada_cpi = {'n_estimators': 40, 'learning_rate': 0.005}
        HBO_params_ada_power = {'n_estimators': 26, 'learning_rate': 0.01}
        base_estimator = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
            **HBO_params_ada_cpi,
        )
        base_estimator2 = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_power),
            **HBO_params_ada_power,
        )
        #surrogate_model_dict['initial_point_generator'] = "orthogonal"
        surrogate_model_dict['different_model'] = True
    elif "BagGBRT" == surrogate_model_tag:
        if 36864 == N_SPACE_SIZE:
            HBO_params_cpi = {'n_estimators': 139, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.6}
            HBO_params_power = {'n_estimators': 199, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.6}            
        else:
            HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
            HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        HBO_params_ada_power = {'n_estimators': 20, 'n_jobs': 1}
        HBO_params_ada_cpi = {'n_estimators': 20}
        from sklearn.ensemble import BaggingRegressor
        base_estimator = BaggingRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
            **HBO_params_ada_cpi,
        )
        base_estimator2 = BaggingRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_power),
            **HBO_params_ada_power,
        )
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
        #surrogate_model_dict['initial_point_generator'] = "kmeans"
        surrogate_model_dict['different_model'] = True
        surrogate_model_dict['model_is_iterative'] = True
        if 1:
            surrogate_model_dict['ucb'] = 0.05
            surrogate_model_dict['ucb_v'] = 12
        if 1:
            surrogate_model_dict['filter'] = 2
        surrogate_model_tag_real += '_v2'
    elif "AdaGBRT-cv" == surrogate_model_tag:
        HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
        HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        from sklearn.ensemble import AdaBoostRegressor
        HBO_params_ada_cpi = {'n_estimators': 40, 'learning_rate': 0.005}
        HBO_params_ada_power = {'n_estimators': 26, 'learning_rate': 0.01}
        base_estimator = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
            **HBO_params_ada_cpi,
        )
        base_estimator2 = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_power),
            **HBO_params_ada_power,
        )
        # 'sobol', 'lhs', 'halton', 'hammersly','random', or 'grid', "orthogonal"
        surrogate_model_dict['initial_point_generator'] = 'orthogonal'
        #surrogate_model_tag_real += '_v2'
        surrogate_model_dict['different_model'] = True
        #surrogate_model_dict['ucb'] = 0.01
        #surrogate_model_dict['ucb_v'] = 27
        # surrogate_model_dict['ucb_scale'] = 1.01
        #surrogate_model_dict['semi_train'] = 1
        surrogate_model_dict['semi_train_iter_max'] = 30
        #surrogate_model_dict['semi_train_adapt'] = True
        if surrogate_model_dict['semi_train']:
            surrogate_model_dict['cv_ranking'] = 'minsort'

        #surrogate_model_dict['cv_ranking_beta'] = 0.1
        surrogate_model_dict['cv_ranking_beta_v'] = 12
        surrogate_model_dict['cv_pool_size'] = 40
        #surrogate_model_dict['acq_func'] = 'EHVI'

        surrogate_model_dict['non_uniformity_explore'] = 16
        surrogate_model_dict['hv_scale_v'] = 1
    else:
        print(f"no def surrogate_model_tag={surrogate_model_tag}")
        exit(1)

    if surrogate_model_dict['semi_train']:
        surrogate_model_tag_real += '_semiv6-' + str(surrogate_model_dict['semi_train'])
        surrogate_model_tag_real += "_w" + str(surrogate_model_dict['semi_train_iter_max'])
        if surrogate_model_dict['semi_train_adapt']:
            surrogate_model_tag_real += '_adapt'

    if surrogate_model_dict['model_is_iterative'] is False:
        surrogate_model_tag_real += "_no_iter"
    if base_estimator2 is not None:
        surrogate_model_dict['different_model'] = True
    if surrogate_model_dict['different_model']:
        surrogate_model_tag_real += "_diff_model"
    if 'HVI' != surrogate_model_dict['acq_func']:
        surrogate_model_tag_real += '_' + surrogate_model_dict['acq_func']
    if 'no' != surrogate_model_dict['cv_ranking']:
        surrogate_model_tag_real += "_cv" + surrogate_model_dict['cv_ranking']
    if surrogate_model_dict['cv_ranking_beta']:
        surrogate_model_tag_real += "_cvbetav" + str(surrogate_model_dict['cv_ranking_beta_v']) + "-" + str(surrogate_model_dict['cv_ranking_beta'])
    if surrogate_model_dict['ucb_v']:
        # surrogate_model_tag_real += "_ucb-" + str(surrogate_model_dict['ucb']) + '-' + str(surrogate_model_dict['ucb_scale'])
        surrogate_model_tag_real += "_ucbv" + str(surrogate_model_dict['ucb_v']) + "-" + str(surrogate_model_dict['ucb'])
    if surrogate_model_dict['non_uniformity_explore']:
        surrogate_model_tag_real += "_univMAX" + str(surrogate_model_dict['non_uniformity_explore'])
    if surrogate_model_dict['warm_start']:
        surrogate_model_tag_real += "_warm"
    if surrogate_model_dict['filter']:
        surrogate_model_tag_real += "_fivg" + str(surrogate_model_dict['filter'])        
    if "random" != surrogate_model_dict['initial_point_generator']:
        surrogate_model_tag_real += "_" + surrogate_model_dict['initial_point_generator']
    #if surrogate_model_dict['labeled_predict_mode'] is False:
    if surrogate_model_dict['hv_scale_v']:
        surrogate_model_tag_real += "_v5"
    else:
        surrogate_model_tag_real += "_v4"
    surrogate_model_tag_real += "_n" + str(N_SAMPLES_ALL)

    surrogate_model_dict['tag'] = surrogate_model_tag_real
    return base_estimator, base_estimator2, surrogate_model_tag_real, surrogate_model_dict


def ci(y):
    # 95% for 1.96
    return 1.96 * y.std(axis=0) / np.sqrt(len(y))


def save_line_file(case_name, surrogate_model_tag_real, result_array_samples, data_name):
    result_filename_prefix= ''
    if 2304 != N_SPACE_SIZE:
        result_filename_prefix += str(N_SPACE_SIZE) + "_"
    result_mape_line_file = open("log_summary_mape_line/" + result_filename_prefix + case_name + "_" + data_name + ".txt", "a")
    result_mape_line_file.write("%-40s %2d " % (surrogate_model_tag_real, len(result_array_samples),))
    for sample_iter in range(N_SAMPLES_INIT, N_SAMPLES_ALL + 1):
        # print(f"result_array_samples[{sample_iter}]={result_array_samples[sample_iter]}")
        result_mape_line_file.write(
            "%-10f %-10f " % (result_array_samples[sample_iter].mean(), ci(result_array_samples[sample_iter]),))
    result_mape_line_file.write('\n')
    result_mape_line_file.close()


result_array = np.zeros(n_experiment)
result_time_array = np.zeros(n_experiment)
result_mape_array1 = np.zeros(n_experiment)
result_mape_array2 = np.zeros(n_experiment)
result_unlabelled_mape_array1 = np.zeros(n_experiment)
result_unlabelled_mape_array2 = np.zeros(n_experiment)

if mape_line_analysis:
    result_array_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_time_array_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_mape_array1_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_mape_array2_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_unlabelled_mape_array1_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_unlabelled_mape_array2_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_non_uniformity_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_coverage = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_hv = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_hv_acq_stuck = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_hv_last_iter = np.zeros([N_SAMPLES_ALL + 1, n_experiment])

result_array_sample_map = {}

result_array_sample_map['time_train'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
result_array_sample_map['time_infer'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
result_array_sample_map['time_explore'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
result_array_sample_map['time_pareto'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
result_array_sample_map['time_adrs'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
result_array_sample_map['time_hv'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
result_array_sample_map['time_filter'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
result_array_sample_map['filter_ratio'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
result_array_sample_map['time_pred_pareto'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
result_array_sample_map['time_register'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
result_array_sample_map['time_valid'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
#result_array_sample_map['time_io_acc'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
#result_array_sample_map['time_dataset_acc'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])


init_effect_analysis = True
if init_effect_analysis:
    result_init_alg = np.zeros(n_experiment)


def reset_result():
    result_array = np.zeros(n_experiment)
    result_time_array = np.zeros(n_experiment)
    result_mape_array1 = np.zeros(n_experiment)
    result_mape_array2 = np.zeros(n_experiment)
    result_unlabelled_mape_array1 = np.zeros(n_experiment)
    result_unlabelled_mape_array2 = np.zeros(n_experiment)    

    if mape_line_analysis:
        result_array_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
        result_time_array_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
        result_mape_array1_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
        result_mape_array2_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
        result_unlabelled_mape_array1_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
        result_unlabelled_mape_array2_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
        result_non_uniformity_samples = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
        result_coverage = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
        result_hv = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
        result_hv_acq_stuck = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
        result_hv_last_iter = np.zeros([N_SAMPLES_ALL + 1, n_experiment])

    result_array_sample_map = {}
    result_array_sample_map['time_train'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_array_sample_map['time_infer'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_array_sample_map['time_explore'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_array_sample_map['time_pareto'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_array_sample_map['time_hv'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_array_sample_map['time_filter'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])    
    result_array_sample_map['filter_ratio'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_array_sample_map['time_pred_pareto'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    result_array_sample_map['time_register'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])   
    result_array_sample_map['time_valid'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment]) 
    #result_array_sample_map['time_io_acc'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])
    #result_array_sample_map['time_dataset_acc'] = np.zeros([N_SAMPLES_ALL + 1, n_experiment])


    if init_effect_analysis:
        result_init_alg = np.zeros(n_experiment)


def save_result(surrogate_model_tag_real):
    result_filename_prefix= ''
    if 2304 != N_SPACE_SIZE:
        result_filename_prefix += str(N_SPACE_SIZE) + "_"
    result_summary_file = open("log_summary/" + result_filename_prefix + case_name + "-summary.txt", "a")
    result_summary_file.write(
        "%-60s %-10f %-10f %-10f %-2d %-15s %10f %10f %10f %10f %10f %10f %10f %10f %10f \n" \
        % (surrogate_model_tag_real,
           result_array.mean(), result_array_ci,
           result_time_array.mean(),
           len(result_array), hostname,
           ci(result_time_array),
           result_mape_array1.mean(), ci(result_mape_array1),
           result_mape_array2.mean(), ci(result_mape_array2),
           result_unlabelled_mape_array1.mean(), ci(result_unlabelled_mape_array1),
           result_unlabelled_mape_array2.mean(), ci(result_unlabelled_mape_array2),
           )
    )
    result_summary_file.close()

    if mape_line_analysis:
        # result_array_samples_index = result_array_samples > 0
        # result_array_samples = result_array_samples[result_array_samples_index]
        # result_time_array_samples = result_time_array_samples[result_array_samples_index]
        # result_mape_array1_samples = result_mape_array1_samples[result_array_samples_index][N_SAMPLES_ALL:]
        # result_mape_array2_samples = result_mape_array2_samples[result_array_samples_index][N_SAMPLES_ALL:]
        print(f"mape_line: surrogate_model= {surrogate_model_tag_real}")
        save_line_file(case_name, surrogate_model_tag_real, result_array_samples, data_name="adrs")
        save_line_file(case_name, surrogate_model_tag_real, result_time_array_samples, data_name="time")
        save_line_file(case_name, surrogate_model_tag_real, result_mape_array1_samples, data_name="mape_cpi")
        save_line_file(case_name, surrogate_model_tag_real, result_mape_array2_samples, data_name="mape_power")
        save_line_file(case_name, surrogate_model_tag_real, result_unlabelled_mape_array1_samples, data_name="unlabelled_mape_cpi")
        save_line_file(case_name, surrogate_model_tag_real, result_unlabelled_mape_array2_samples, data_name="unlabelled_mape_power")
        save_line_file(case_name, surrogate_model_tag_real, result_non_uniformity_samples, data_name="non_uniformity")
        save_line_file(case_name, surrogate_model_tag_real, result_coverage, data_name="coverage")
        save_line_file(case_name, surrogate_model_tag_real, result_hv, data_name="hv")
        save_line_file(case_name, surrogate_model_tag_real, result_hv_acq_stuck, data_name="hv_acq_stuck")
        save_line_file(case_name, surrogate_model_tag_real, result_hv_last_iter, data_name="hv_last_iter")
        print(f"hv_acq_stuck= {np.mean(result_hv_acq_stuck[-1, :])}")
    
    save_line_file(case_name, surrogate_model_tag_real, result_array_sample_map['time_train'], data_name="time_train")
    save_line_file(case_name, surrogate_model_tag_real, result_array_sample_map['time_infer'], data_name="time_infer")
    save_line_file(case_name, surrogate_model_tag_real, result_array_sample_map['time_explore'], data_name="time_explore")
    save_line_file(case_name, surrogate_model_tag_real, result_array_sample_map['time_pareto'], data_name="time_pareto")
    save_line_file(case_name, surrogate_model_tag_real, result_array_sample_map['time_adrs'], data_name="time_adrs")
    save_line_file(case_name, surrogate_model_tag_real, result_array_sample_map['time_hv'], data_name="time_hv")
    save_line_file(case_name, surrogate_model_tag_real, result_array_sample_map['time_filter'], data_name="time_filter")
    save_line_file(case_name, surrogate_model_tag_real, result_array_sample_map['filter_ratio'], data_name="filter_ratio")
    save_line_file(case_name, surrogate_model_tag_real, result_array_sample_map['time_pred_pareto'], data_name="time_pred_pareto")
    save_line_file(case_name, surrogate_model_tag_real, result_array_sample_map['time_register'], data_name="time_register")
    save_line_file(case_name, surrogate_model_tag_real, result_array_sample_map['time_valid'], data_name="time_valid")
    #save_line_file(case_name, surrogate_model_tag_real, result_array_sample_map['time_io_acc'], data_name="time_io_acc")  
    #save_line_file(case_name, surrogate_model_tag_real, result_array_sample_map['time_dataset_acc'], data_name="time_dataset_acc")


def save_result_init_effect(surrogate_model_tag_real, surrogate_model_dict):
    if init_effect_analysis:
        global result_init_alg
        result_init_alg_index = result_init_alg > 0
        result_init_alg_valid = result_init_alg[result_init_alg_index]
        print(
            f"init_algo= {surrogate_model_dict['initial_point_generator']} result= {result_init_alg_valid.mean()} ci= {ci(result_init_alg_valid)}")
        result_init_alg_summary_file = open("log_summary" + "/init_algo_summary" + '_' + str(N_SPACE_SIZE) + ".txt", "a")
        result_init_alg_summary_file.write(
            "%-15s %-10f %-10f %2d %-10s %-40s \n" % (surrogate_model_dict['initial_point_generator'],
                                                      result_init_alg_valid.mean(), ci(result_init_alg_valid),
                                                      len(result_init_alg_valid),
                                                      case_name,
                                                      surrogate_model_tag_real),
        )
        result_init_alg_summary_file.close()


reset_result()

if __name__ == '__main__':
    for surrogate_model_tag in surrogate_model_tag_list:
        #if "BOOM-Explorer" in surrogate_model_tag:
        #    n_experiment = 1
        #else:
        #    n_experiment = 10

        if 0:
            if "ActBoost" in surrogate_model_tag:
                mape_line_analysis = True
            elif "GBRT" in surrogate_model_tag:
                mape_line_analysis = True
            elif "BOOM-Explorer" in surrogate_model_tag:
                mape_line_analysis = True
            else:
                mape_line_analysis = False

        thread_list = []
        reset_result()
        experiment_range = range(n_experiment)
        if exp_id is not None:
            experiment_range = range(exp_id, exp_id+1)
        for thread_i in experiment_range:
            # try:
            surrogate_model_config = get_surrogate_model(surrogate_model_tag)
            if 0 == thread_i or (exp_id is not None and exp_id == thread_i):
                _, _, surrogate_model_tag_real, surrogate_model_dict = surrogate_model_config
                print("running " + case_name + " " + surrogate_model_tag_real)
            problem_model = Problem_Model(surrogate_model_config, n=thread_i)
            problem_model.run()
            # thread_list.append(problem_model)
            # problem_model.start()
            # problem_model.join()
            # except:
            # print(f"thread {thread_i} exception")
        # for thread_list_entry in thread_list:
        #        thread_list_entry.join()

        # filter failed cases
        valid_index = result_array > 0
        result_array = result_array[valid_index]
        result_time_array = result_time_array[valid_index]
        result_array_ci = ci(result_array)
        # print(f"hostname={hostname}")
        print(f"surrogate_model= {surrogate_model_tag_real} result= {result_array.mean()} ci= {result_array_ci}")
        save_result(surrogate_model_tag_real=surrogate_model_tag_real)
        save_result_init_effect(surrogate_model_tag_real=surrogate_model_tag_real, surrogate_model_dict=surrogate_model_dict)
