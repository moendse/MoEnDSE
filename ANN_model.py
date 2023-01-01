import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type(torch.DoubleTensor)


class Loss_Fun(nn.Module):

    def __init__(self):
        super(Loss_Fun, self).__init__()

    def forward(self, y, label):
        return torch.abs((y / label) - 1)


class MLP_Predictor(nn.Module):
    def __init__(self, in_channel, out_channel, drop_rate, use_bias, use_drop, initial_lr, momentum,
                 loss_fun):
        super(MLP_Predictor, self).__init__()
        hidden_size_1 = 1000
        hidden_size_2 = 500
        hidden_size_3 = 50
        self.layer1 = nn.Linear(in_channel, hidden_size_1, bias=use_bias)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2, bias=use_bias)
        self.layer3 = nn.Linear(hidden_size_2, hidden_size_3, bias=use_bias)
        self.layer4 = nn.Linear(hidden_size_3, out_channel, bias=use_bias)
        #self.layer2 = nn.Linear(hidden_size_1, out_channel, bias=use_bias)
        self.use_drop = use_drop
        self.dropout = nn.Dropout(drop_rate)

        # self.optimizer = optimizer
        #self.optimizer = torch.optim.SGD(self.parameters(), lr = initial_lr, momentum = momentum)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=initial_lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 2000, gamma=1.5, last_epoch=-1)
        self.loss_fun = loss_fun
        # self.bench_each_norm = bench_each_norm
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        # self.bn3 = nn.BatchNorm1d(hidden_size)
        # self.batch_size = batch_size
        # self.CPI_mode = CPI_mode

    def predict(self, _input):
        self.eval()
        return self.forward(_input)

    def forward(self, _input):
        x = torch.Tensor(_input)
        # x = x.to(torch.float32)
        # print("layer0: x", x)
        #x = torch.sigmoid(self.layer1(x))
        x = F.relu(self.layer1(x))
        #x = self.layer1(x)
        x = F.relu(self.layer2(x))
        #x = torch.sigmoid(x)
        #x = self.layer2(x)
        # print("layer2 x", x)
        # x = F.relu(self.layer3(x))
        #x = torch.sigmoid(x)
        x = self.layer3(x)
        # x = torch.sigmoid(x)
        #x = torch.relu(x)
        x = self.layer4(x)
        if self.use_drop:
            y = self.dropout(x)
            print("use_drop error")
            exit(1)
        else:
            y = x
        #print(f"y={y.flatten()}")
        return y

    def my_train(self, train_data, label):
        # data_shuffle(batch_input)
        # [final_data, min_max_scaler] = data_preprocess(label, feature_range = (min(label),max(label)))
        # final_data = min_max_scaler.transform(label)
        self.train()
        self.optimizer.zero_grad()
        train_iter = 0
        train_iter_max = 1000
        train_loss = -1
        while train_iter < train_iter_max:
            output_nn = self.forward(train_data)
            # output_nn = min_max_scaler.inverse_transform(output_nn)
            #output_nn_copy = copy.deepcopy(output_nn)
            loss = self.loss_fun(torch.as_tensor(output_nn).flatten(), torch.as_tensor(label).flatten())
            #loss = self.loss_fun(torch.as_tensor(output_nn), torch.as_tensor(label))
            #loss = loss.sum()
            if torch.isnan(loss):
                print(f"train loss is NaN!!!!")
                exit(1)
            loss.backward()
            self.optimizer.step()
            train_loss = loss.item()
            if train_iter + 1 == train_iter_max:
            #if 0 == (train_iter & 0x3f):
                print(f"nn={torch.as_tensor(output_nn).flatten()}")
                #print(f"label={label}")
                print(f"train_loss={train_loss}")
            #if 0 == (train_iter & 0x2f):
            #   print(f"train_iter={train_iter}, train_loss={train_loss}")
            train_iter += 1
        #print(f"train_iter={train_iter}, train_loss={train_loss}")
        return train_loss
