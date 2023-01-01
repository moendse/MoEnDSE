import numpy as np
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.containers import TrainingData
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.means import ConstantMean
from xgboost import XGBRegressor
from torch import Tensor
from typing import Any

class botorch_model_wrapper(SingleTaskGP):
    _num_outputs = 2  # to inform GPyTorchModel API

    def __init__(self, train_X: Tensor, train_Y: Tensor, model):
        super().__init__(train_X, train_Y)
        # print(f"_aug_batch_shape={self._aug_batch_shape}")
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_X.shape[-1],
                batch_shape=self._aug_batch_shape,
            ),
            batch_shape=self._aug_batch_shape,
        )
        self.model = model
        # self.to(x_train)  # make sure we're on the right device/dtype

    def forward(self, x: Tensor) -> MultivariateNormal:
        print(f"forward")
        y_pred = self.model.predict(x)
        print(f"y_pred={y_pred}")
        mean_x = (y_pred)
        covar_x = self.covar_module(x)
        print(f"covar_x={covar_x}")
        return MultivariateNormal(mean_x, covar_x)

    def posterior(
            self,
            X: Tensor,
            output_indices=None,
            observation_noise=False,
            posterior_transform=None,
            **kwargs: Any,
    ) -> GPyTorchPosterior:
        print(f"posterior")
        y_pred = self.model.predict(X)
        return GPyTorchPosterior(y_pred)

    def forward2(self, X_test):
        # dt_stump_err = 1.0 - self.dt_stump.score(X_test,y_test)
        # print(dt_stump_err)

        y_pred = self.regresspr.predict(X_test)
        # for y in y_pred:
        print(y_pred)
        result = y_pred
        # result = torch.empty(X_test.size()[0], X_test.size()[0], X_test.size()[0])
        # print(f"y_pred={y_pred.size()}, result={result.size()}")
        return result

    def predict(self, X_test):
        # ada_real_err_train=np.zeros((self.n_estimators,))
        y_pred = self.regresspr.predict(X_test)
        return y_pred

    @classmethod
    def construct_inputs(cls, training_data: TrainingData, **kwargs):
        r"""Construct kwargs for the `AdaBoost_Model` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        return {"train_X": training_data.X, "train_Y": training_data.Y}
