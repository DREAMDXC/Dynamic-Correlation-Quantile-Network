
"""
@Date: 2025/01/23

@Author: DREAM_DXC

@Summary: Dynamic correlation quantile network

"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from Lib import dataload
import scipy.stats as st

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--farm_number", type=int, default=1, help="number of wind farm")
parser.add_argument("--batch_size", type=int, default=50, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="RMSprop: learning rate")
parser.add_argument("--target_step", type=int, default=24, help="define the target temporal step")
parser.add_argument("--scenario_number", type=int, default=500, help="define the scenario number")
parser.add_argument("--seed", default=0, type=int)

args = parser.parse_args()
print(args)

# Set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

#gpu or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("train_type:",device.type)

def dataload_data(input_data, target_data, BATCH_SIZE):
    torch_dataset = Data.TensorDataset(input_data, target_data)
    train_loader = Data.DataLoader(
        dataset=torch_dataset,  # torch Tensor Dataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # random shuffle for training
    )
    return train_loader
"""---------------------------------------------Define IQN Networks----------------------------------------------"""

# Implicit Quantile Network Decoder Network
class IQN_Decoder(nn.Module):
    def __init__(self):
        super(IQN_Decoder, self).__init__()

        self.input = 6
        self.channel = 32

        self.input = nn.Conv2d(
                in_channels=self.input,  # input height
                out_channels=self.channel,  # n_filters
                kernel_size=(1, 1),  # filter size
                stride=(1, 1),  # filter movement/step
                padding=(0, 0),
                dilation=(1, 1),)

        self.TCN = nn.ModuleList()
        self.TCN.append((nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel,  # input height
                out_channels=self.channel,  # n_filters
                kernel_size=(1, 3),  # filter size
                stride=(1, 1),  # filter movement/step
                padding=(0, 0),
                dilation=(1, 1),),
            nn.LeakyReLU(0.2),
        )))
        self.TCN.append((nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel,  # input height
                out_channels=self.channel,  # n_filters
                kernel_size=(1, 3),  # filter size
                stride=(1, 1),  # filter movement/step
                padding=(0, 0),
                dilation=(1, 2),),
            nn.LeakyReLU(0.2),
        )))
        self.TCN.append((nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel,  # input height
                out_channels=self.channel,  # n_filters
                kernel_size=(1, 3),  # filter size
                stride=(1, 1),  # filter movement/step
                padding=(0, 0),
                dilation=(1, 3), ),
            nn.LeakyReLU(0.2),
        )))

        self.out_p = nn.ModuleList()
        self.out_p.append(nn.Sequential(
            nn.Conv2d(self.channel, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(0.2)
        ))
        self.out_p.append(nn.Sequential(
            nn.Conv2d(self.channel, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(0.2)
        ))
        self.out_p.append(nn.Sequential(
            nn.Conv2d(self.channel, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(0.2)
        ))

        self.quantile = nn.ModuleList()
        self.quantile.append(nn.Sequential(
                nn.Conv2d(1, self.channel, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
                nn.LeakyReLU(0.2)
        ))
        self.quantile.append(nn.Sequential(
                nn.Conv2d(1, self.channel, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
                nn.LeakyReLU(0.2)
        ))
        self.quantile.append(nn.Sequential(
                nn.Conv2d(1, self.channel, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
                nn.LeakyReLU(0.2)
        ))

        self.padding = nn.ModuleList()
        self.padding.append(nn.ReplicationPad2d(padding=(1, 1, 0, 0)))   # Left, right, up, down
        self.padding.append(nn.ReplicationPad2d(padding=(2, 2, 0, 0)))
        self.padding.append(nn.ReplicationPad2d(padding=(3, 3, 0, 0)))

        self.weight = nn.Parameter(torch.FloatTensor(1, 1, 3, args.target_step))
        torch.nn.init.uniform_(self.weight.data, a=0, b=1)

    def forward(self, x, quantile):
        # TCN size regular : output temporal size = [input_size + padding_size_left + padding_size_right -dilation*(kernel_size -1)]/stride +1
        x = self.input(x)
        res_x = x
        x = self.padding[0](x)
        x = self.TCN[0](x)
        q_x = torch.mul(x, self.quantile[0](quantile))
        p = self.out_p[0](q_x)   # size (batch,1,1,args.target_step)
        out_p = p

        for i in range(1,3):
            x = x + res_x
            x = self.padding[i](x)
            x = self.TCN[i](x)
            res_x = x
            q_x = torch.mul(x, self.quantile[i](quantile))
            p = self.out_p[i](q_x)
            out_p = torch.cat((out_p, p), 2)  # size(batch,1,3,args.target_step)

        weight = self.weight.repeat(out_p.size(0), 1, 1, 1)
        scenario = F.sigmoid(torch.sum(out_p * weight, dim=2))
        scenario = scenario.view(scenario.size(0), 1, args.target_step)

        return scenario

# Implicit Quantile Network Encoder Network
class IQN_Encoder(nn.Module):
    def __init__(self):
        super(IQN_Encoder, self).__init__()

        self.input = 6
        self.channel = 32

        self.input = nn.Conv2d(
                in_channels=self.input,  # input height
                out_channels=self.channel,  # n_filters
                kernel_size=(1, 1),  # filter size
                stride=(1, 1),  # filter movement/step
                padding=(0, 0),
                dilation=(1, 1),)

        self.TCN = nn.ModuleList()
        self.TCN.append((nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel,  # input height
                out_channels=self.channel,  # n_filters
                kernel_size=(1, 3),  # filter size
                stride=(1, 1),  # filter movement/step
                padding=(0, 0),
                dilation=(1, 1),),
            nn.LeakyReLU(0.2),
        )))
        self.TCN.append((nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel,  # input height
                out_channels=self.channel,  # n_filters
                kernel_size=(1, 3),  # filter size
                stride=(1, 1),  # filter movement/step
                padding=(0, 0),
                dilation=(1, 2),),
            nn.LeakyReLU(0.2),
        )))
        self.TCN.append((nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel,  # input height
                out_channels=self.channel,  # n_filters
                kernel_size=(1, 3),  # filter size
                stride=(1, 1),  # filter movement/step
                padding=(0, 0),
                dilation=(1, 3), ),
            nn.LeakyReLU(0.2),
        )))

        self.out_p = nn.ModuleList()
        self.out_p.append(nn.Sequential(
            nn.Conv2d(self.channel, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(0.2)
        ))
        self.out_p.append(nn.Sequential(
            nn.Conv2d(self.channel, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(0.2)
        ))
        self.out_p.append(nn.Sequential(
            nn.Conv2d(self.channel, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(0.2)
        ))

        self.quantile = nn.ModuleList()
        self.quantile.append(nn.Sequential(
                nn.Conv2d(1, self.channel, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
                nn.LeakyReLU(0.2)
        ))
        self.quantile.append(nn.Sequential(
                nn.Conv2d(1, self.channel, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
                nn.LeakyReLU(0.2)
        ))
        self.quantile.append(nn.Sequential(
                nn.Conv2d(1, self.channel, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
                nn.LeakyReLU(0.2)
        ))

        self.padding = nn.ModuleList()
        self.padding.append(nn.ReplicationPad2d(padding=(1, 1, 0, 0)))   # Left, right, up, down
        self.padding.append(nn.ReplicationPad2d(padding=(2, 2, 0, 0)))
        self.padding.append(nn.ReplicationPad2d(padding=(3, 3, 0, 0)))

        self.weight = nn.Parameter(torch.FloatTensor(1, 1, 3, args.target_step))
        torch.nn.init.uniform_(self.weight.data, a=0, b=1)

    def forward(self, x, forecast):

        # TCN size regular : output temporal size = [input_size + padding_size_left + padding_size_right -dilation*(kernel_size -1)]/stride +1
        x = self.input(x)
        res_x = x
        x = self.padding[0](x)
        x = self.TCN[0](x)
        q_x = torch.mul(x, self.quantile[0](forecast))
        p = self.out_p[0](q_x)   # size (batch,1,1,args.target_step)
        out_p = p

        for i in range(1,3):
            x = x + res_x
            x = self.padding[i](x)
            x = self.TCN[i](x)
            res_x = x
            q_x = torch.mul(x, self.quantile[i](forecast))
            p = self.out_p[i](q_x)
            out_p = torch.cat((out_p, p), 2)  # size(batch,1,3,args.target_step)

        weight = self.weight.repeat(out_p.size(0), 1, 1, 1)
        quantile = F.sigmoid(torch.sum(out_p * weight, dim=2))
        quantile = quantile.view(quantile.size(0), 1, args.target_step)

        return quantile

"""---------------------------------------------Define DCN Networks----------------------------------------------"""

class DCN(nn.Module):
    def __init__(self):
        super(DCN, self).__init__()


        self.conv1 = nn.Sequential(nn.ReplicationPad2d(padding=(1, 1, 0, 0)),
                                   nn.Conv2d(in_channels=6, out_channels=8,kernel_size=(1, 3),stride=(1, 1),padding=(0, 0),dilation=(1, 1)),
                                   nn.LeakyReLU(0.2))

        self.conv2 = nn.Sequential(nn.ReplicationPad2d(padding=(1, 1, 0, 0)),
                                   nn.Conv2d(in_channels=8, out_channels=16,kernel_size=(1, 3),stride=(1, 1),padding=(0, 0),dilation=(1, 1)),
                                   nn.LeakyReLU(0.2))

        self.conv3 = nn.Sequential(nn.ReplicationPad2d(padding=(1, 1, 0, 0)),
                                   nn.Conv2d(in_channels=16, out_channels=32,kernel_size=(1, 3),stride=(1, 1),padding=(0, 0),dilation=(1, 1)),
                                   nn.LeakyReLU(0.2))

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x).squeeze(2) # batch channel target_step

        x = F.softplus(torch.bmm(x.permute(0,2,1),x))

        matrix = torch.reshape(x, (x.size(0), 24, 24))
        triangular_matrix = torch.tril(matrix, diagonal=0)

        # row norm of triangular_matrix
        # The sum of squares of each row is equal to 1 (variance of MultivariateNormal)
        norm = torch.sqrt(torch.sum(triangular_matrix*triangular_matrix,dim=2,keepdim=True))
        triangular_matrix = triangular_matrix/norm

        return triangular_matrix

def lossfunction_probabilistic(target_data, forecast, quantile):
    """
       The quantile loss function for a given quantile tau:
       L(y_true, y_pred) = (tau - I(y_pred < y_true)) * (y_pred - y_true)
       Where I is the indicator function.
    """
    dy = forecast - target_data

    tau = quantile

    loss = torch.mean(torch.sum((1.0 - tau) * F.relu(dy) + tau * F.relu(-dy), dim=(1,2)))  # loss

    return  loss

def lossfunction_reconstruction(quantile1,quantile2):

    loss = torch.mean(torch.sum(torch.abs(quantile1-quantile2),dim=(1,2)))

    return loss

def cholesky(A):
    n = len(A)
    L = np.zeros(A.shape)
    L[0,0] = np.sqrt(A[0,0])
    for i in range(1,n):
        for j in range(0,i):  # j<i
            L[i,j] = (A[i,j] - np.sum(L[i,:j]*L[j,:j]))/L[j,j]
        L[i,i] = np.sqrt(A[i,i] - np.sum(L[i,:i]**2))
    return L,L.T

def lossfunction_MLE_Copula(triangular_matrix,z):

    cov_matrix = torch.bmm(triangular_matrix,triangular_matrix.permute(0, 2, 1))

    log_det = torch.mean(torch.log(torch.det(cov_matrix) + 1e-30))

    # inverse of cov_matrix
    inver_cov = torch.linalg.inv(cov_matrix)
    z_cov_zt = torch.mean(torch.bmm(torch.bmm(z, inver_cov), z.permute(0, 2, 1)))

    loss = log_det + z_cov_zt

    return loss
"""---------------------------------------------traning and testing---------------------------------------------"""

def train(Valid_input,Valid_target):

    Valid_input = Valid_input.to(device).unsqueeze(dim=2)
    Valid_target = Valid_target.to(device).view(Valid_target.size(0), 1, args.target_step)

    Forecast_Net = IQN_Decoder().to(device)
    print('Forecast Model Struct:', Forecast_Net)
    Quantile_Net = IQN_Encoder().to(device)
    print('Quantile Model Struct:', Quantile_Net)
    DCN_Net = DCN().to(device)
    print('Copula Model Struct:', DCN_Net)

    print("Total number of paramerters in networks is {} ".format(sum(x.numel() for x in Forecast_Net.parameters())))
    print("Total number of paramerters in networks is {} ".format(sum(x.numel() for x in Quantile_Net.parameters())))
    print("Total number of paramerters in networks is {} ".format(sum(x.numel() for x in DCN_Net.parameters())))

    opt_forecast_net = torch.optim.Adam(Forecast_Net.parameters(), lr=args.lr, weight_decay=0.002)
    opt_quantile_net = torch.optim.Adam(Quantile_Net.parameters(), lr=args.lr, weight_decay=0.002)
    opt_dcn_net = torch.optim.Adam(DCN_Net.parameters(), lr=args.lr, weight_decay=0.002)

    time_start = time.time()

    # First training forecast network
    for epoch in range(args.epoch):
        total_loss = 0

        for step, (Train_input, Train_target) in enumerate(train_loader):

            current_batch_size = Train_input.shape[0]

            # data to device
            Train_input = Train_input.to(device)
            Train_target = Train_target.to(device)

            Train_target = Train_target.view(Train_target.size(0), 1, args.target_step)

            Quantile = torch.rand(Train_input.size(0),1,args.farm_number,args.target_step).to(device)
            Quantile = torch.clamp(Quantile,min=1e-5,max=1-(1e-5))
            Train_input = Train_input.unsqueeze(dim=2)

            # decode data
            forecast_scenario = Forecast_Net(Train_input,Quantile)
            forecast_scenario = torch.clamp(forecast_scenario,min=1e-4,max=1-(1e-4))

            Quantile = Quantile.squeeze(dim=1)
            # QR loss
            loss = lossfunction_probabilistic(Train_target, forecast_scenario, Quantile)
            total_loss += loss.item() * current_batch_size

            opt_forecast_net.zero_grad()
            loss.backward()
            opt_forecast_net.step()

        total_loss = total_loss / len(train_loader.dataset)

        if epoch % 10 == 0:

            Quantile = torch.rand(Valid_input.size(0),1,args.farm_number,args.target_step).to(device)
            Quantile = torch.clamp(Quantile,min=1e-5,max=1-(1e-5))

            # decode data
            forecast_scenario = Forecast_Net(Valid_input,Quantile)
            forecast_scenario = torch.clamp(forecast_scenario,min=1e-4,max=1-(1e-4))

            Quantile = Quantile.squeeze(dim=1)
            # QR loss
            valid_loss = lossfunction_probabilistic(Valid_target, forecast_scenario, Quantile).cpu().detach().numpy()

            print('epoch:', epoch, 'QR loss:', total_loss, 'Valid loss:',valid_loss)

    # Second training quantile reconstruct network
    for epoch in range(args.epoch):
        total_loss = 0

        for step, (Train_input, _) in enumerate(train_loader):

            # data to device
            Train_input = Train_input.to(device)

            Quantile = torch.rand(Train_input.size(0),1,args.farm_number,args.target_step).to(device)
            Train_input = Train_input.unsqueeze(dim=2)
            # encode data
            forecast_scenario = Forecast_Net(Train_input,Quantile).detach()

            Quantile_re = Quantile_Net(Train_input,forecast_scenario.unsqueeze(dim=1))
            Quantile = Quantile.squeeze(dim=1)

            # mle loss
            loss_re = lossfunction_reconstruction(Quantile, Quantile_re)
            total_loss += loss_re.item() * current_batch_size

            opt_quantile_net.zero_grad()
            loss_re.backward()
            opt_quantile_net.step()

        total_loss = total_loss / len(train_loader.dataset)

        if epoch % 10 == 0:
            Quantile = torch.rand(Valid_input.size(0), 1, args.farm_number, args.target_step).to(device)
            forecast_scenario = Forecast_Net(Valid_input, Quantile).detach()

            Quantile_re = Quantile_Net(Valid_input, forecast_scenario.unsqueeze(dim=1))
            Quantile = Quantile.squeeze(dim=1)

            # mle loss
            valid_loss = lossfunction_reconstruction(Quantile, Quantile_re).cpu().detach().numpy()

            print('epoch:', epoch, 'RE loss:', total_loss, 'Valid loss:',valid_loss)

    # third training DCN network
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epoch):
        total_loss = 0

        for step, (Train_input, Train_target) in enumerate(train_loader):

            Train_input = Train_input.unsqueeze(dim=2).to(device)

            Copula_cdf = Quantile_Net(Train_input, Train_target.to(device).unsqueeze(1)).cpu().detach().numpy().squeeze()
            Copula_cdf = np.clip(Copula_cdf, a_min=1e-5, a_max=1 - (1e-5))
            Copula_pdf = st.norm.ppf(Copula_cdf)
            Copula_pdf = torch.from_numpy(Copula_pdf).float().unsqueeze(1).to(device)

            matrix = DCN_Net(Train_input)

            loss_mle = lossfunction_MLE_Copula(matrix,Copula_pdf)
            total_loss += loss_mle.item() * current_batch_size

            opt_dcn_net.zero_grad()
            with torch.autograd.detect_anomaly():
                loss_mle.backward()
            opt_dcn_net.step()

        total_loss = total_loss / len(train_loader.dataset)

        if epoch % 10 == 0:

            Copula_cdf = Quantile_Net(Valid_input,Valid_target.unsqueeze(1)).cpu().detach().numpy().squeeze()
            Copula_cdf = np.clip(Copula_cdf, a_min=1e-5, a_max=1 - (1e-5))
            Copula_pdf = st.norm.ppf(Copula_cdf)
            Copula_pdf = torch.from_numpy(Copula_pdf).float().unsqueeze(1).to(device)

            matrix = DCN_Net(Valid_input)

            valid_loss = lossfunction_MLE_Copula(matrix, Copula_pdf).cpu().detach().numpy()

            print('epoch:', epoch, 'MLE loss:', total_loss, 'Valid loss:',valid_loss)

    time_end = time.time()
    print('totally cost:', time_end - time_start)

    data_file_name = 'Forecast_Net{ep}.pkl'.format(ep=args.epoch)
    torch.save(Forecast_Net.state_dict(), data_file_name)  # save model parameters

    data_file_name = 'Quantile_Net{ep}.pkl'.format(ep=args.epoch)
    torch.save(Quantile_Net.state_dict(), data_file_name)  # save model parameters

    data_file_name = 'DCN_Net{ep}.pkl'.format(ep=args.epoch)
    torch.save(DCN_Net.state_dict(), data_file_name)  # save model parameters

"""---------------------------------------------main function----------------------------------------------------"""

if __name__ == '__main__':

    Train_input, Train_target, Valid_input, Valid_target, Test_input, Test_target = dataload.dataloder('Wind')

    train_loader = dataload_data(input_data=Train_input, target_data=Train_target, BATCH_SIZE=args.batch_size)

    train(Valid_input,Valid_target)
