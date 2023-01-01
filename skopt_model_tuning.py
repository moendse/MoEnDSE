import random
import socket
from datetime import datetime

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import skopt
from config import case_name, N_SPACE_SIZE
from get_real_pareto_frontier import transfer_version_to_var, get_pareto_optimality_from_file_ax_interface
from main import problem_space

from simulation_metrics import metrics_all, read_metrics
from skopt.plots import plot_evaluations

import warnings

from skopt_plot import case_names

#warnings.filterwarnings('ignore', category=ConvergenceWarning)

base_estimator_name = "BagGBRT"
#base_estimator_name = "AdaGBRT"
objective_metric = 'CPI'
#objective_metric = 'Power'
train_size = 100

sota_model_tuning = False
if "SemiBoost" == base_estimator_name:
    sota_model_tuning = True
elif "ActBoost" == base_estimator_name:
    sota_model_tuning = True

def get_base_estimator(base_estimator_name = "GBRT"):

    base_estimator_name_log = base_estimator_name
    surrogate_model_dict = {}
    surrogate_model_dict['kernel_train'] = False

    match base_estimator_name:
        case "PolyLinear":
            #degree = 1
            SPACE = [skopt.space.Integer(1, 8, name="poly__degree", prior='uniform'),]
            base_estimator = Pipeline([
                ("poly", PolynomialFeatures()),
                ("std_scaler", StandardScaler()),
                ("lin_reg", LinearRegression())
            ])
            print(f"keys={base_estimator.get_params().keys()}")
        case "Ridge":
            SPACE = [
                skopt.space.Categorical(categories=["rbf", "sigmoid", "linear"], name="kernel"),
            ]
            base_estimator = KernelRidge()
        case "SVR":
            from sklearn.svm import SVR
            SPACE = [
                skopt.space.Integer(1, 8, name="degree", prior='uniform'),
                skopt.space.Integer(1, 10, name="kernel__k2__length_scale", prior='uniform'),
                skopt.space.Categorical(categories=[2.5, 1.5], name="kernel__k2__nu"),
            ]
            kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 20.0))
            # noise_level = 0.0
            base_estimator = SVR(kernel=kernel)
            print(f"keys={base_estimator.get_params().keys()}")
        case "MLP":
            SPACE = [
                # skopt.space.Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
                # skopt.space.Categorical(categories=["(100, 30)", "(100, 100, 30)", "(100, 100, 100, 30)"], name="hidden_layer_sizes"),
                skopt.space.Categorical(categories=[0.00001, 0.001, 0.005, 0.01, 0.02], name="learning_rate_init"),
                # skopt.space.Real(0.001, 0.01, name='learning_rate_init', prior='log-uniform'),
                skopt.space.Categorical(categories=["relu", "logistic"], name="activation"),
                skopt.space.Categorical(categories=["adam", "sgd", "lbfgs"], name="solver"),
            ]
            hidden_layer_1 = 16
            hidden_size = (hidden_layer_1, hidden_layer_1 * 2, hidden_layer_1 * 2)
            base_estimator = MLPRegressor(hidden_layer_sizes=hidden_size,
                                          max_iter=10000)
            base_estimator_name_log += 'v2'
        case "ASPLOS06":
            SPACE = [
                skopt.space.Categorical(categories=["adam", "sgd"], name="solver"),
            ]
            base_estimator = MLPRegressor(hidden_layer_sizes=(16),
                                          # solver='sgd',  # ['adam', 'sgd', 'lbfgs'],
                                          activation='relu',
                                          max_iter=10000,
                                          )
        case "GP":
            SPACE = [
                skopt.space.Integer(2, 10, name='n_restarts_optimizer', prior='uniform'),
                skopt.space.Categorical(categories=[2.5, 1.5], name="kernel__k2__nu"),
                skopt.space.Categorical(categories=[False, True], name="normalize_y"),

            ]
            kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 20.0))
            # noise_level = 0.0
            base_estimator = GaussianProcessRegressor(kernel=kernel,
                                                      # alpha=noise_level ** 2,
                                                      # normalize_y=True,
                                                      )
            print(f"params_space={base_estimator.get_params().keys()}")
        case "GP_DKLv2":
            SPACE = [
                skopt.space.Integer(2, 10, name='n_restarts_optimizer', prior='uniform'),
                #skopt.space.Categorical(categories=[2.5, 1.5], name="kernel__nu"),
                skopt.space.Categorical(categories=[False, True], name="normalize_y"),

            ]
            from sklearn_DKL_GP import Sklearn_DKL_GP
            #kernel = Sklearn_DKL_GP(length_scale=1.0, length_scale_bounds=(1e-10, 10.0), nu=2.5)
            #kernel = Sklearn_DKL_GP(length_scale=0.1, length_scale_bounds=(1e-2, 10.0), nu=2.5)
            kernel = Sklearn_DKL_GP(nu=2.5)
            # noise_level = 0.0
            base_estimator = GaussianProcessRegressor(kernel=kernel,
                                                      # alpha=noise_level ** 2,
                                                      # normalize_y=True,
                                                      )
            print(f"params_space={base_estimator.get_params().keys()}")
            surrogate_model_dict['kernel_train'] = True
        case "RF":
            SPACE = [
                # skopt.space.Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
                skopt.space.Integer(50, 200, name="n_estimators", prior='uniform'),
                skopt.space.Integer(4, 30, name='max_depth', prior='uniform'),
                skopt.space.Integer(1, 3, name='min_samples_leaf', prior='uniform'),
                # skopt.space.Categorical(categories=[True, False], name="bootstrap")
            ]
            base_estimator = RandomForestRegressor(n_jobs=2)
        case "ET":
            SPACE = [
                skopt.space.Integer(50, 200, name="n_estimators", prior='uniform'),
                skopt.space.Integer(4, 30, name='max_depth', prior='uniform'),
                skopt.space.Integer(1, 3, name='min_samples_leaf', prior='uniform'),
                skopt.space.Categorical(categories=['auto', 'sqrt', 'log2'], name="max_features"),
            ]
            base_estimator = ExtraTreesRegressor(n_jobs=2)
        case "GBRT":
            SPACE = [
                # skopt.space.Categorical(categories=["quantile", "squared_error"], name="loss"),
                skopt.space.Integer(20, 200, name="n_estimators", prior='uniform'),
                skopt.space.Categorical(categories=[0.01, 0.1, 0.2, 0.3, 1.0], name="learning_rate"),
                skopt.space.Integer(2, 30, name='max_depth', prior='uniform'),
                skopt.space.Categorical(categories=[0.5, 0.6, 0.7, 0.8], name="subsample"),
            ]
            base_estimator = GradientBoostingRegressor()
        case "AdaBoost_DTR":
            SPACE = [
                skopt.space.Integer(2, 200, name='n_estimators', prior='uniform'),
                skopt.space.Categorical(categories=[0.001, 0.005, 0.01, 1.0], name="learning_rate"),
                skopt.space.Categorical(categories=[DecisionTreeRegressor(max_depth=8),
                                                    DecisionTreeRegressor(max_depth=16),
                                                    DecisionTreeRegressor(max_depth=32),
                                                    DecisionTreeRegressor(max_depth=64), ],
                                        name="base_estimator"),
            ]
            # dt_stump = DecisionTreeRegressor(max_depth=8, min_samples_leaf=1)
            base_estimator = AdaBoostRegressor(
                # base_estimator=dt_stump,
            )
        case "ActBoost_MLPv3":
            SPACE = [
                skopt.space.Integer(10, 100, name='n_estimators', prior='uniform'),
                skopt.space.Categorical(categories=[0.001, 0.005, 0.01, 1.0], name="learning_rate"),
                #skopt.space.Categorical(categories=["relu", "logistic"], name="activation"),
            ]
            # hidden_layer_1 = 16
            hidden_size = (16, 32, 32)
            dt_stump = MLPRegressor(hidden_layer_sizes=hidden_size, max_iter=10000,
                                    solver='lbfgs',  # ['adam', 'sgd', 'lbfgs'],
                                    activation='logistic',
                                    )
            # dt_stump.fit(train_X, train_Y)
            base_estimator = AdaBoostRegressor(
                base_estimator=dt_stump,
            )
            base_estimator_name_log += 'v2'
        case "ActBoost_v4":
            SPACE = [
                skopt.space.Integer(10, 100, name='n_estimators', prior='uniform'),
                skopt.space.Categorical(categories=[0.001, 0.005, 0.01, 1.0], name="learning_rate"),
            ]
            # hidden_layer_1 = 16
            hidden_size = (8, 6)
            dt_stump = MLPRegressor(hidden_layer_sizes=hidden_size, max_iter=10000)
            # dt_stump.fit(train_X, train_Y)
            base_estimator = AdaBoostRegressor(
                base_estimator=dt_stump,
            )
        case "SemiBoost":
            SPACE = [
                skopt.space.Categorical(categories=[0.001, 0.005, 0.01, 1.0], name="learning_rate"),
            ]
            hidden_layer_1 = 8
            dt_stump = MLPRegressor(hidden_layer_sizes=(hidden_layer_1, hidden_layer_1),
                                    max_iter=10000,
                                    solver='sgd',  # ['adam', 'sgd', 'lbfgs'],
                                    activation='relu',
                                    )
            # dt_stump.fit(train_X, train_Y)
            base_estimator = AdaBoostRegressor(
                base_estimator=dt_stump,
                learning_rate=0.001,
                n_estimators=20,
            )            
        case "XGBoost":
            SPACE = [
                skopt.space.Integer(20, 200, name="n_estimators", prior='uniform'),
                skopt.space.Categorical(categories=[0.01, 0.1, 0.15, 0.2, 0.3], name="learning_rate"),
                skopt.space.Integer(4, 50, name='max_depth', prior='uniform'),
                # skopt.space.Categorical(categories=['reg:squarederror'], name="objective"),
                skopt.space.Categorical(categories=['gbtree', 'gblinear', 'dart'], name="booster"),
                skopt.space.Categorical(categories=[0.5, 0.6, 0.7, 0.8], name="subsample"),
            ]
            base_estimator = XGBRegressor(
                n_jobs=2,
                nthread=None,
            )
        case "AdaXGBoost":
            SPACE = [
                skopt.space.Integer(20, 200, name='n_estimators', prior='uniform'),
                skopt.space.Categorical(categories=[0.001, 0.005, 0.01, 1.0], name="learning_rate"),
            ]
            HBO_params_cpi = {'n_estimators': 70, 'learning_rate': 0.2, 'max_depth': 48, 'booster': 'gbtree',
                              'subsample': 0.8}
            HBO_params_power = {'n_estimators': 97, 'learning_rate': 0.1, 'max_depth': 30,
                                'objective': 'reg:squarederror', 'booster': 'gbtree', 'subsample': 0.5}
            if "CPI" == objective_metric:
                base_estimator = AdaBoostRegressor(
                    base_estimator=XGBRegressor(**HBO_params_cpi),
                )
            else:
                base_estimator = AdaBoostRegressor(
                    base_estimator=XGBRegressor(**HBO_params_power),
                )
        case "LGBMRegressor":
            SPACE = [
                skopt.space.Integer(20, 100, name="n_estimators", prior='uniform'),
                skopt.space.Categorical(categories=[0.001, 0.005, 0.01, 0.1, 1.0], name="learning_rate"),
            ]
            base_estimator = LGBMRegressor(
                n_jobs=2,
                nthread=None,
            )
        case "CatBoostRegressor":
            SPACE = [
                skopt.space.Integer(20, 200, name="n_estimators", prior='uniform'),
                skopt.space.Integer(6, 10, name='depth', prior='uniform'),
                # skopt.space.Categorical(categories=[0.01, 0.05, 0.1, 0.2, 0.3], name="learning_rate"),
                skopt.space.Categorical(categories=[0.7, 0.8, 0.9, 1.0], name="subsample"),
                # skopt.space.Categorical(categories=['RMSE', 'MAPE'], name="loss_function"),
                skopt.space.Integer(20, 200, name="early_stopping_rounds", prior='uniform'),
                skopt.space.Categorical(categories=["SymmetricTree", "Depthwise", "Lossguide"], name="grow_policy"),
            ]
            base_estimator = CatBoostRegressor(verbose=False, thread_count=-1)
        case "BagGBRT":
            SPACE = [
                skopt.space.Integer(20, 40, name='n_estimators', prior='uniform'),
                skopt.space.Categorical(categories=[0.5, 0.7, 0.8, 1.0], name="max_samples"),
                skopt.space.Categorical(categories=[0.5, 0.7, 0.8, 1.0], name="max_features"),
                #skopt.space.Integer(0, 1, name="bootstrap"),
            ]
            if 0:
                HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                                  'subsample': 0.8}
                HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                                    'subsample': 0.5}
            else:
                HBO_params_cpi = {'n_estimators': 139, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.6}
                HBO_params_power = {'n_estimators': 199, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.6}
            base_estimator = BaggingRegressor(
                base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
            )
            base_estimator_name_log += 'v2'
        case "AdaGBRT":
            SPACE = [
                skopt.space.Integer(20, 200, name='n_estimators', prior='uniform'),
                skopt.space.Categorical(categories=[0.01, 0.05, 0.1, 0.2, 0.5], name="learning_rate"),
            ]
            if "CPI" == objective_metric:
                HBO_params_cpi = {'n_estimators': 198, 'learning_rate': 0.1, 'max_depth': 12, 'subsample': 0.5} 
                base_estimator = AdaBoostRegressor(
                    base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
                )
            else:
                HBO_params_power = {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.6} 
                base_estimator = AdaBoostRegressor(
                    base_estimator=GradientBoostingRegressor(**HBO_params_power),
                )         
        case "_":
            print(f"no match base_estimator_name={base_estimator_name}")
            base_estimator = None
            exit(1)
    base_estimator_name_log += '-' + objective_metric
    return base_estimator, base_estimator_name_log, surrogate_model_dict, SPACE

base_estimator, base_estimator_name_log, surrogate_model_dict, SPACE = get_base_estimator(base_estimator_name = base_estimator_name)

HPO_PARAMS = {
    'n_calls': 50,
    'n_random_starts': 10, #10
    'base_estimator': 'GP',
    'acq_func': "EI",  # 'EI',
    # 'noise': 0.013,
}


def to_named_params(results, search_space):
    params = results
    param_dict = {}
    params_list = [(dimension.name, param) for dimension, param in zip(search_space, params)]
    for item in params_list:
        param_dict[item[0]] = item[1]
    return param_dict


@skopt.utils.use_named_args(SPACE)
def objective(**params):
    all_params = {**params}
    return evaluator.evaluate_params(all_params)


def calculate_rmse(model, X, y):
    y_hat = model.predict(X)
    y_true = np.asarray(y)
    # print(f"y_hat= {y_hat}\ny_true= {y_true}")
    rmse = np.sqrt(((y_true - y_hat) ** 2).mean())
    return rmse


class ParamsEvaluate():
    def __init__(self, X_train, X_val, y_train, y_val, pareto_data):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        [self.real_pareto_points_x, self.real_pareto_points_y, self.config_vector_list] = pareto_data
        self.n = 0

    def select_model(self, model):
        self.model = model

    def evaluate_params(self, params):
        self.n += 1
        model_origin = self.model.set_params(**params)

        if True:
            # X_train,X_test, y_train, y_test = train_test_split(train_data, train_target, train_size=0.1)
            # shufsp1 = ShuffleSplit(train_size=0.1, test_size=0.9, n_splits=10)
            shufsp1 = ShuffleSplit(train_size=train_size, n_splits=10)
            scoring = make_scorer(mean_absolute_percentage_error, greater_is_better=True, needs_proba=False,
                                  needs_threshold=False)
            model = clone(model_origin)
            if surrogate_model_dict['kernel_train']:
                for train_index, test_index in shufsp1.split(self.X_train, self.y_train):
                    #print(f"train_index={train_index}")
                    model.kernel.my_train(np.asarray(self.X_train)[train_index[:25]],
                                          np.asarray(self.y_train)[train_index[:25]])
                    break
            scores = cross_val_score(model, self.X_train, self.y_train, cv=shufsp1, n_jobs=1, scoring=scoring)
            if surrogate_model_dict['kernel_train']:
                print(f"model_theta={model.kernel.theta}")
            score_return = scores.mean()
            global scores_return_min
            if score_return < scores_return_min.mean():
                scores_return_min = scores
                scores_return_min_index = self.n
            # print(f"scores ={scores} std={scores.std()}")
            print("Iteration %d score= %0.5f (+/- %0.5f ) %s" % (self.n, scores.mean(), scores.std() * 2, params))
            result_log.write("Iteration %d score= %0.5f (+/- %0.5f )" % (self.n, scores.mean(), scores.std() * 2))
        else:
            model.fit(self.X_train, self.y_train)
            y_train_predict, y_train_predict_std = model.predict(self.X_train, return_std=True)
            # print(f"y_train_predict_std={y_train_predict_std}")
            y_val_predict = model.predict(self.X_val)
            y_pareto_predict = model.predict(self.config_vector_list)
            MAPE_train = mean_absolute_percentage_error(y_train_predict, self.y_train)
            score_return = MAPE_val = mean_absolute_percentage_error(y_val_predict, self.y_val)
            y_val_pareto = self.real_pareto_points_x if objective_metric == 'CPI' else self.real_pareto_points_y
            MAPE_val_pareto = mean_absolute_percentage_error(y_pareto_predict, y_val_pareto)
            # print("values: validation/train/pareto")
            # result_log.write("values: validation/train/pareto\n")
            print(
                "Iteration {} MAPE = {:.5f} / {:.5f} / {:.5f} {}".format(self.n, MAPE_val, MAPE_train, MAPE_val_pareto,
                                                                         params))
            result_log.write(
                "Iteration {} MAPE = {:.5f} / {:.5f} / {:.5f} ".format(self.n, MAPE_val, MAPE_train, MAPE_val_pareto))

        if False:
            rmse_train = calculate_rmse(model, self.X_train, self.y_train)
            rmse_val = calculate_rmse(model, self.X_val, self.y_val)
            print("Iteration {} with RMSE = {:.5f} / {:.5f} (validation/train) at {}" \
                  .format(self.n, rmse_val, rmse_train, str(datetime.now().time())[:8]))

        result_log.write(f"{params}\n")
        return score_return


def get_dataset(metrics_all):
    X_train = []
    X_val = []
    y_train = []
    y_val = []
    random.shuffle(metrics_all)
    for train_index, design_point in enumerate(metrics_all):
        if len(design_point['version']) < 30:
            continue
        else:
            x = transfer_version_to_var(design_point['version'])
            if train_index <= int(len(metrics_all) * 1):
                x_trans = []
                for space_trans, space_var in zip(problem_space, x):
                    x_trans.append(space_trans.transform(space_var))
                X_train.append(x_trans)
                y_train.append(design_point[objective_metric])
                # print(f"x={x}")
            else:
                X_val.append(x)
                y_val.append(design_point[objective_metric])
    print(f"all size ={len(metrics_all)} X_train size= {len(X_train)}, X_val size= {len(X_val)}")
    return X_train, X_val, y_train, y_val


print(f"train_size={train_size}")

#for case_name in case_names:
for case_name in [case_name]:
    if 2304 != N_SPACE_SIZE:
        prefix = str(N_SPACE_SIZE) + '_'
    else:
        prefix = ''      
    base_estimator_name_log_case = prefix + case_name + "-" + base_estimator_name_log  
    print(base_estimator_name_log_case)    
    result_log = open("log_tune_model/" + base_estimator_name_log_case + ".log", "w")
    pareto_data = get_pareto_optimality_from_file_ax_interface(case_name)

    metrics_all = read_metrics('data_all_simpoint/', case_name)
    X_train, X_val, y_train, y_val = get_dataset(metrics_all)

    scores_return_min = np.asarray([999.99])
    scores_return_min_index = -1

    evaluator = ParamsEvaluate(X_train, X_val, y_train, y_val, pareto_data)
    evaluator.select_model(model=base_estimator)

    results = skopt.gp_minimize(objective, SPACE, n_jobs=2, acq_optimizer='sampling', **HPO_PARAMS)

    param_dict = to_named_params(results.x, SPACE)
    print("----------------------------------------------------------------------------------------------------\n")
    print(f"best func_value= {results.fun} param_dict={param_dict}")

    hostname = socket.getfqdn(socket.gethostname())

    result_log.write(f"results={results}\n")
    result_log.write(
        f"best func_value= {results.fun} (+/- {scores_return_min.std() * 2} ) iter= {scores_return_min_index} param_dict= {param_dict} hostname= {hostname}\n")
    result_log.close()

    result_summary_log = open("log_tune_model/log_tune_model_summary_" + base_estimator_name + '-' + objective_metric + ".log", "a")
    result_summary_log.write(
        f"%-10f (+/- %-10f ) %-35s %-15s %s \n"
        % (results.fun, scores_return_min.std() * 2, base_estimator_name_log_case, hostname, param_dict))
    result_summary_log.close()

# plot_evaluations(results)
# plt.show()
