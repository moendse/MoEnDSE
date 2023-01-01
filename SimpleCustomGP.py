from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.containers import TrainingData
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel, MaternKernel, IndexKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
#from gpytorch.lazy import NonLazyTensor
#from sklearn.gaussian_process.kernels import Matern

import torch

algo_kernel = ""
#algo_kernel = "RBF"
#algo_kernel = "Matern"
#algo_kernel = "MLP"
#algo_kernel = "AdaBoost"

def set_algo_kernel(kernal_type):
    global algo_kernel
    #print(f"algo_kernel={kernal_type}")
    algo_kernel = kernal_type

class CustomKernal(Kernel):

    def __init__(self, train_X, train_Y, kernal_type):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__()
        #print(f"train_X={train_X}")
        #print(f"train_Y={train_Y}")
        #print(f"kernal_type={kernal_type}")
        match kernal_type:
            case "MLP":
                from ANN_model import MLP_Predictor, Loss_Fun
                self.model = MLP_Predictor(
                    in_channel = train_X.shape[-1]
                    , out_channel = 1
                    , hidden_size = 50 #useless now, define in MLP_Predictor
                    , drop_rate = 0.01, use_bias=True, use_drop=False
                    , initial_lr = 0.001
                    , momentum = 0.4
                    , loss_fun = torch.nn.MSELoss() #Loss_Fun()
                )
                self.model.train()
                self.model.my_train(train_X, train_Y)
            case "AdaBoost":
                from AdaBoost_Model import AdaBoost_Model             
                self.model = AdaBoost_Model(train_X, train_Y)
                #self.model.train()            
            case _:
                print(f"no def kernal_type={kernal_type}")
                exit(1)

    def two_dim(self, x1, x2):
        cov_matrix = torch.empty(x1.size()[0], x2.size()[0])

        x1_predictions = torch.empty(x1.size()[0])
        #print(f"x1_predictions size={x1_predictions.size()}")
        x2_predictions = torch.empty(x2.size()[0])
        #print(f"x2_predictions size={x2_predictions.size()}")
        #self.model.eval()
        x1_predictions = self.model.forward(x1).flatten()
        x2_predictions = self.model.forward(x2).flatten()
        for x1_index in range(len(x1)):
            x1_prediction_extend = torch.ones(len(x2)) * x1_predictions[x1_index]
            #cov_matrix[x1_index] = 1 - (abs(x1_prediction_extend - x2_predictions) / max_value)
            if True:
                kernel = MaternKernel(ard_num_dims=1, nu=5/2)
                #kernel = MaternKernel(ard_num_dims=1, nu=5/2, batch_shape = torch.Size)
                kerval_value = kernel(x1_prediction_extend, x2_predictions).evaluate_kernel().evaluate()[0]
                #print(f"kerval_value={kerval_value}")
                cov_matrix[x1_index] = kerval_value
            else:
                cov_matrix[x1_index] = 2 * torch.sigmoid(0 - abs(x1_prediction_extend - x2_predictions))
        #print(cov_matrix)
        return cov_matrix

    def forward(self, x1, x2, diag=False, **params):
        if 3 == len(x1.size()):
            cov_matrix = torch.empty(x1.size()[0], x1.size()[1], x2.size()[1])
            #print(f"x1={x1.size()}\n")
            #print(f"x2={x2.size()}\n")
            #print(f"cov_matrix={cov_matrix.size()}\n")
            for first_dim_iter, x1_element, x2_element in zip(range(len(x1)), x1, x2):
                cov_matrix[first_dim_iter] = self.two_dim(x1_element, x2_element)
        else:
            cov_matrix = self.two_dim(x1, x2)
        #print(cov_matrix)
        return cov_matrix

class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        #print(f"train_X: {train_X}")
        self.mean_module = ConstantMean()
        global algo_kernel
        #print(f"algo_kernel={algo_kernel}")
        match algo_kernel:
            case "RBF":
                base_kernel = RBFKernel(ard_num_dims=train_X.shape[-1])
            case "Matern":
                base_kernel = MaternKernel(ard_num_dims=train_X.shape[-1])
            case _:
                base_kernel = CustomKernal(train_X=train_X, train_Y=train_Y, kernal_type = algo_kernel)
        #print(f"algo_kernel={algo_kernel}")
        if True:
            self.covar_module = ScaleKernel(
                base_kernel=base_kernel,
            )
        else:
            self.covar_module = base_kernel
        #self.input_transform = InputTransform()
        #self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        result = MultivariateNormal(mean_x, covar_x)
        #print(f"mean_x={mean_x.size()}, covar_x={covar_x.size()}")
        return result

    @classmethod
    def construct_inputs(cls, training_data: TrainingData, **kwargs):
        r"""Construct kwargs for the `SimpleCustomGP` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        return {"train_X": training_data.X, "train_Y": training_data.Y}