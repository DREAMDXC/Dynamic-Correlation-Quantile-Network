"""
@Date: 2023/04/21

@Author: DREAM_DXC

@Summary: define a index for day-ahead scenario generation

include three index class：
1、deterministic forecasting index
2、probabilistic forecasting index
3、scenario forecasting index

"""

import torch
import pandas as pd
import numpy as np
import os

"""---------------------------------------deterministic_evaluation_index---------------------------------------------"""
# data size:(batch_number:B, farms_number:I, target_step:J)
# mean absolute error (MAE) = 1/B * 1/I * 1/J * sigma(abs(forecast - tagart))
# root mean square error (RMSE) = sigma[sqrt(1/I *1/B * 1/J * sigma((forecast - tagart)^2))]

# single time mean absolute error (ST_MAE) = 1/B * 1/I * sigma(abs(forecast - tagart))
# single time root mean square error (ST_RMSE) = sigma[sqrt(1/I *1/B * sigma((forecast - tagart)^2))]

# single farm mean absolute error (SF_MAE) = 1/B * 1/J * sigma(abs(forecast - tagart))
# single farm root mean square error (SF_RMSE) = sqrt(1/B * 1/J * sigma((forecast - tagart)^2))

# GB/T 19963.1-2021
# 风电场超短期风电功率预测结果 第4h平均预测准确率不低于87％
# 风电场短期风电功率预测结果 日前预测平均准确率不低于83％
# 风电场中期风电功率预测结果 第10日平均准确率不低于70％

class deterministic_evaluation_index():

    def MAE(self, forecast, target):
        MAE = np.mean(np.abs(forecast - target))
        MAE = np.expand_dims(MAE, axis=0)
        print("MAE:", MAE)

        ST_MAE = np.mean(np.abs(forecast - target), axis=(0, 1))
        print("ST_MAE:", ST_MAE)

        SF_MAE = np.mean(np.abs(forecast - target), axis=(0, 2))
        print("SF_MAE:", SF_MAE)

        return MAE, ST_MAE, SF_MAE

    def RMSE(self, forecast, target):
        RMSE = np.mean(np.sqrt(np.mean(np.power(forecast - target, 2), axis=(1, 2))))
        RMSE = np.expand_dims(RMSE, axis=0)
        print("RMSE:", RMSE)

        ST_RMSE = np.mean(np.sqrt(np.mean(np.power(forecast - target, 2), axis=1)), axis=0)
        print("ST_RMSE:", ST_RMSE)

        SF_RMSE = np.mean(np.sqrt(np.mean(np.power(forecast - target, 2), axis=2)), axis=0)
        print("SF_RMSE:", SF_RMSE)

        return RMSE, ST_RMSE, SF_RMSE

    def accuracy_rate(self, RMSE):
        # accuracy rate (AR) based on RMSE
        AR = (1 - RMSE)
        print("AR:", AR * 100, "％")

        return AR

    def pass_rate(self, forecast, target):
        pass_max_error = 0.25

        # percent of pass (PP) based on MAE
        error = np.abs(forecast - target)
        index = error <= pass_max_error
        PP = np.sum(index) / np.size(error)
        PP = np.expand_dims(PP, axis=0)
        print("PP:", PP * 100, "％")

        ST_PP = np.sum(index, axis=(0, 1)) / np.product(error.shape[0] * error.shape[1])
        print("ST_PP:", ST_PP * 100, "％")

        SF_PP = np.sum(index, axis=(0, 2)) / np.product(error.shape[0] * error.shape[2])
        print("SF_PP:", SF_PP * 100, "％")

        return PP, ST_PP, SF_PP

    def max_error(self, forecast, target):
        # max error (ME)
        error = np.abs(forecast - target)

        ME = np.max(error)
        ME = np.expand_dims(ME, axis=0)
        print("ME:", ME)

        ST_ME = np.max(error, axis=(0, 1))
        print("ST_ME:", ST_ME)

        SF_ME = np.max(error, axis=(0, 2))
        print("SF_ME:", SF_ME)

        return ME, ST_ME, SF_ME

    def main(self, forecast, target):

        print(' ♪(^∇^*) !! Deterministic Forecasting Index：')

        MAE, ST_MAE, SF_MAE = self.MAE(forecast, target)
        RMSE, ST_RMSE, SF_RMSE = self.RMSE(forecast, target)
        AR = self.accuracy_rate(RMSE)
        PP, ST_PP, SF_PP = self.pass_rate(forecast, target)
        ME, ST_ME, SF_ME = self.max_error(forecast, target)

        xlsfile = 'deterministic_evaluation_index.xlsx'
        if os.path.exists(xlsfile):
            origin = pd.read_excel(xlsfile).values
            data = np.array([MAE, RMSE, AR, PP, ME]).T
            data = np.concatenate((origin, data), axis=0)
        else:
            data = np.array([MAE, RMSE, AR, PP, ME]).T

        xlsfile = pd.ExcelWriter('deterministic_evaluation_index.xlsx')

        form_header = ['MAE', 'RMSE', 'AR', 'PP', 'ME']
        data_df = pd.DataFrame(data)
        data_df.to_excel(xlsfile, sheet_name='index', float_format='%.5f',
                         header=form_header, index=False, index_label=False)  # float_format control float

        xlsfile.close()

"""---------------------------------------probabilistic_evaluation_index---------------------------------------------"""

class probabilistic_evaluation_index():
    # scenario size: (batch number:B, scenario number:N, farms_number:I, target_step:J)
    # forecast size: (batch_number:B, farms_number:I, target_step:J)

    def Quantile_line(self,scenario,target):
        # numpy.reshape(a, newshape, order='C')
        # ‘C’ means read in line order, with the last axis index changing fastest
        # ‘F’ means read by column order, with the first index changing fastest
        # scenario size:(scenario number:N,farms_number:I, target_step:J*B)
        scenario = scenario.transpose(1, 2, 0, 3).reshape((scenario.shape[1], scenario.shape[2],
                                                           scenario.shape[3] * scenario.shape[0]), order='C')

        # target size:(farms_number:I,target_step:J*B)
        target = target.transpose(1, 0, 2).reshape((target.shape[1], target.shape[2] *
                                                    target.shape[0]), order='C')

        Quantile = np.sort(scenario, axis=0)
        interval = scenario.shape[0] // 20
        Quantile_dot = np.arange(interval, scenario.shape[0], interval) - 1
        Quantile_line = Quantile[Quantile_dot, :, :]

        return Quantile_line,target

    def CRPS(self,scenario,target):

        # Continuous Ranked Probability Score, CRPS
        # http://www.lokad.com/cn/%E8%BF%9E%E7%BB%AD-%E6%A6%82%E7%8E%87-%E6%8E%92%E4%BD%8D-%E5%88%86%E6%95%B0
        # CRPS = E|X1-x|-1/2*E|X1-X2|

        CRPS_target = torch.from_numpy(target).float()  # size(batch_size,farms_number,target_step)
        CRPS_target = CRPS_target.unsqueeze(dim=1).repeat(1, scenario.shape[1], 1, 1)

        CRPS_scenario = torch.from_numpy(scenario).float()  # size(batch_size,scenario_number,farms_number,target_step)

        crps = torch.zeros([target.shape[0], target.shape[1], target.shape[2]])

        X1_x = torch.abs(CRPS_target - CRPS_scenario)
        EX1_x = torch.mean(X1_x, dim=1, keepdim=True)  # size(batch_size,1,farms_number,target_step)

        for k in range(target.shape[1]):
            for j in range(target.shape[2]):
                for i in range(target.shape[0]):
                    X1 = CRPS_scenario[i, :, k, j].unsqueeze(dim=1).repeat(1, scenario.shape[1])
                    X2 = X1.transpose(0, 1)
                    X1_X2 = torch.abs(X1 - X2)
                    EX1_X2 = torch.mean(X1_X2)

                    crps[i, k, j] = EX1_x[i, 0, k, j] - 0.5 * EX1_X2

        CRPS = torch.mean(crps).numpy()
        CRPS = np.expand_dims(CRPS, axis=0)
        print("CRPS:", CRPS)

        ST_CRPS = torch.mean(crps, dim=[0, 1]).numpy()
        print("ST_CRPS:", ST_CRPS)

        SF_CRPS = torch.mean(crps, dim=[0, 2]).numpy()
        print("ST_CRPS:", SF_CRPS)

        return CRPS,ST_CRPS,SF_CRPS

    def Pinball_score(self,Quantile_line,target):

        # pinball score
        # http://www.lokad.com/cn/Pinball-%e6%8d%9f%e5%a4%b1%e5%87%bd%e6%95%b0-%e5%ae%9a%e4%b9%89
        # PS = Φ(quantile-target)*τ + Φ(target-quantile)*(1-τ)
        # Φ(x) is the indicative function, x greater than 0, Φ(x) is 1, x less than 0, Φ(x) is 0

        dy = Quantile_line - target  # quantile farm_number time_step
        tau = np.expand_dims(np.arange(0.05, 1.00, 0.05), axis=(1, 2))
        tau = np.tile(tau, (1, target.shape[0], target.shape[1]))
        I_dy = np.clip(dy, a_min=0, a_max=None)
        L_dy = np.clip(-dy, a_min=0, a_max=None)
        PS = np.mean(np.sum((1.0 - tau) * I_dy + tau * L_dy, axis=1))  # loss
        PS = np.expand_dims(PS, axis=0)
        print("pinball score:", PS)

        return PS

    def Winkler_score(self,Quantile_line,target):

        # Winkler score & PICP & PIAW
        # 10％~90％ confidence interval

        PICP = np.zeros((target.shape[0], 9))
        PIAW = np.zeros((target.shape[0], 9))
        Winkler = np.zeros((target.shape[0], 9))

        for j in range(target.shape[0]):
            print('farm %s index' % (j + 1))
            for i in range(9):
                # (predict interval average width, PIAW) (predict interval coverage probability, PICP) of farm j
                coverage = np.zeros((target.shape[1]))
                PIAW[j, i] = np.mean(Quantile_line[18 - i, j, :] - Quantile_line[i, j:])
                print('%s' % (90 - i * 10), '% PIAW:', PIAW[j, i], 'p.u.')

                coverage[np.where(
                    (Quantile_line[i, j, :] <= target[j, :]) & (target[j, :] <= Quantile_line[18 - i, j, :]))] = 1
                coverate = np.sum(coverage) / target.shape[1]
                print('%s' % (90 - i * 10), '% PICP:', coverate * 100, '%')
                PICP[j, i] = coverate

                # Winkler score
                Up = np.mean(np.clip(target[j, :] - Quantile_line[18 - i, j, :], a_min=0, a_max=None))
                Down = np.mean(np.clip(Quantile_line[i, j:] - target[j, :], a_min=0, a_max=None))
                Winkler[j, i] = PIAW[j, i] + 2 * Up / (0.1 + i * 0.1) + 2 * Down / (0.1 + i * 0.1)
                print('%s' % (90 - i * 10), '% Winkler:', Winkler[j, i])

        return Winkler,PICP,PIAW

    def main(self,scenario,target):

        print(' ♪(^∇^*) !! Probabilistic Forecasting Index：')

        CRPS, ST_CRPS, SF_CRPS = self.CRPS(scenario, target)

        Quantile_line, target = self.Quantile_line(scenario, target)
        PS = self.Pinball_score(Quantile_line, target)
        Winkler, PICP, PIAW = self.Winkler_score(Quantile_line, target)

        xlsfile = 'probabilistic_evaluation_index.xlsx'
        if os.path.exists(xlsfile):
            origin = pd.read_excel(xlsfile, 'index').values
            data = np.array([CRPS, PS]).T
            data = np.concatenate((origin, data), axis=0)

            origin = pd.read_excel(xlsfile, 'PICP').values
            PICP = np.concatenate((origin, PICP), axis=0)

            origin = pd.read_excel(xlsfile, 'PIAW').values
            PIAW = np.concatenate((origin, PIAW), axis=0)

            origin = pd.read_excel(xlsfile, 'Winkler').values
            Winkler = np.concatenate((origin, Winkler), axis=0)

        else:
            data = np.array([CRPS, PS]).T

        xlsfile = pd.ExcelWriter('probabilistic_evaluation_index.xlsx')

        form_header = ['CRPS', 'PS']
        data_df = pd.DataFrame(data)
        data_df.to_excel(xlsfile, sheet_name='index', float_format='%.5f', header=form_header, index=False,
                         index_label=False)

        form_header = ['90％', '80％', '70％', '60％', '50％', '40％', '30％', '20％', '10％']
        data_df = pd.DataFrame(PICP)
        data_df.to_excel(xlsfile, 'PICP', float_format='%.5f', header=form_header, index=False,
                         index_label=False)
        data_df = pd.DataFrame(PIAW)
        data_df.to_excel(xlsfile, 'PIAW', float_format='%.5f', header=form_header, index=False,
                         index_label=False)
        data_df = pd.DataFrame(Winkler)
        data_df.to_excel(xlsfile, 'Winkler', float_format='%.5f', header=form_header, index=False,
                         index_label=False)  # float_format control fl
        xlsfile.close()

    def main_quantile(self,quantile,scenario,target):
        # Used in quantile regression models

        quantile = np.expand_dims(quantile, axis=2)
        quantile = quantile.transpose(1, 2, 0, 3).reshape((quantile.shape[1], quantile.shape[2],quantile.shape[3] * quantile.shape[0]),order='C')

        print(' ♪(^∇^*) !! Probabilistic Forecasting Index：')

        CRPS, ST_CRPS, SF_CRPS = self.CRPS(scenario,target)

        Quantile_line, target = self.Quantile_line(scenario,target)
        PS = self.Pinball_score(quantile,target)
        Winkler, PICP, PIAW = self.Winkler_score(quantile,target)

        xlsfile = pd.ExcelWriter('probabilistic_evaluation_index.xlsx')
        data = np.array([CRPS, PS]).T
        form_header = ['CRPS', 'PS']
        data_df = pd.DataFrame(data)
        data_df.to_excel(xlsfile, sheet_name='index', float_format='%.5f', header=form_header, index=False,
                         index_label=False)  # float_format control float

        form_header = ['90％', '80％', '70％', '60％', '50％', '40％', '30％', '20％', '10％']
        data_df = pd.DataFrame(PICP)
        data_df.to_excel(xlsfile, 'PICP', float_format='%.5f', header=form_header, index=False,
                         index_label=False)  # float_format control float
        data_df = pd.DataFrame(PIAW)
        data_df.to_excel(xlsfile, 'PIAW', float_format='%.5f', header=form_header, index=False,
                         index_label=False)  # float_format control float
        data_df = pd.DataFrame(Winkler)
        data_df.to_excel(xlsfile, 'Winkler', float_format='%.5f', header=form_header, index=False,
                         index_label=False)  # float_format control float
        xlsfile.close()

"""------------------------------------------scenario_evaluation_index-----------------------------------------------"""

class scenario_evaluation_index():
    # temporal scenario index for 1 station

    def euclidean_dist_numpy(self, x, y):  # numpy distance matrix calculate
        # x,y (batch,scenario number,size) distance (batch,scenario number,number)

        m, n = x.shape[1], y.shape[1]

        xx = np.power(x, 2).sum(2, keepdims=True)  # keepdim = True ,Keep this dimension
        xx = np.tile(xx, (1, 1, n))  # repeat narray dim and number

        yy = np.power(y, 2).sum(2, keepdims=True)
        yy = np.tile(yy, (1, 1, m)).transpose(0, 2, 1)

        dist = xx + yy

        dot = np.einsum('ijk,ikl -> ijl', x, y.transpose(0, 2, 1))
        dist = dist - 2 * dot

        dist = np.clip(dist, a_min=1e-12, a_max=None)
        dist = np.sqrt(dist)

        return dist

    def energy_score(self,scenario,target):
        # energy score
        # paper: P. Pinson and R. Girard, “Evaluating the quality of scenarios of short-term wind power generation,”
        # Appl. Energy, vol. 96, pp. 12–20, 2012, doi: 10.1016/j.apenergy.2011.11.004.

        # scenario (batch,scenario number,1,step) target (batch,1,step)
        scenario = np.squeeze(scenario, axis=2)

        # calculate energy score
        Y1_y = self.euclidean_dist_numpy(scenario, target)
        Y1_Y2 = self.euclidean_dist_numpy(scenario, scenario)

        ES = np.mean(np.mean(Y1_y, axis=(1, 2)) - 0.5 * np.mean(Y1_Y2, axis=(1, 2)))
        ES = np.expand_dims(ES, axis=0)
        print('Energy Score:', ES)

        return ES

    def variogram_score(self,scenario,target):
        # Variogram score
        # paper:  M. Scheuerer and T. M. Hamill, “Variogram-based proper scoring rules for probabilistic forecasts of
        # multivariate quantities,” Mon. Weather Rev., vol. 143, no. 4, pp. 1321–1334, 2015, doi: 10.1175/MWR-D-14-00269.1.

        s = np.tile(scenario, (1, 1, scenario.shape[3], 1))
        Yt1_Yt2 = np.sqrt(np.abs(s - s.transpose(0, 1, 3, 2))) # p=0.5
        # Yt1_Yt2 = np.abs(s - s.transpose(0, 1, 3, 2)) # p=1

        t = np.tile(target, (1, target.shape[2], 1))
        yt1_yt2 = np.sqrt(np.abs(t - t.transpose(0, 2, 1)))  # p=0.5
        # yt1_yt2 = np.abs(t - t.transpose(0, 2, 1)) # p=1

        VS = np.mean(np.sum(np.power(yt1_yt2 - np.mean(Yt1_Yt2, axis=1), 2), axis=(1, 2)))
        VS = np.expand_dims(VS, axis=0)
        print('Variogram Score:', VS)

        return VS

    def ramp_score(self,scenario,target):
        # ramp score proposed by DREAM_DXC
        # The average ramp difference between variables
        s = np.tile(scenario, (1, 1, scenario.shape[3], 1))
        Yt1_Yt2 = s - s.transpose(0, 1, 3, 2)

        t = np.tile(target, (1, target.shape[2], 1))
        yt1_yt2 = t - t.transpose(0, 2, 1)

        RS = np.mean(np.mean(np.abs(yt1_yt2 - np.mean(Yt1_Yt2, axis=1)),axis=(1,2)))
        RS = np.expand_dims(RS, axis=0)
        print('Ramp Score:', RS)

        return RS

    def scenarios_coverate(self,scenario,target):
        # coverate of all scenario set

        # scenario size:(scenario number:N,farms_number:I, target_step:J*B)
        scenario = scenario.transpose(1, 2, 0, 3).reshape((scenario.shape[1], scenario.shape[2],
                                                           scenario.shape[3] * scenario.shape[0]), order='C')

        # target size:(farms_number:I,target_step:J*B)
        target = target.transpose(1, 0, 2).reshape((target.shape[1], target.shape[2] *
                                                    target.shape[0]), order='C')

        sort = np.sort(scenario, axis=0)

        coverage = np.zeros((target.shape[1]))
        PIAW = np.mean(sort[scenario.shape[0]-1, 0, :] - sort[0, 0, :])
        print('%s' % 100, '% PIAW:', PIAW, 'p.u.')

        coverage[np.where(
            (sort[0, 0, :] <= target[0, :]) & (target[0, :] <= sort[scenario.shape[0]-1, 0, :]))] = 1
        coverate = np.sum(coverage) / target.shape[1]
        print('%s' % 100, '% PICP:', coverate * 100, '%')

    def main(self,scenario, target):

        print(' ♪(^∇^*) !! Scenario Forecasting Index：')

        ES = self.energy_score(scenario, target)
        VS = self.variogram_score(scenario, target)
        RS = self.ramp_score(scenario, target)

        self.scenarios_coverate(scenario, target)

        xlsfile = 'scenario_evaluation_index.xlsx'
        if os.path.exists(xlsfile):
            origin = pd.read_excel(xlsfile, 'index').values
            data = np.array([ES, VS, RS]).T
            data = np.concatenate((origin, data), axis=0)

        else:
            data = np.array([ES, VS, RS]).T

        xlsfile = pd.ExcelWriter('scenario_evaluation_index.xlsx')

        form_header = ['ES', 'VS', 'RS']
        data_df = pd.DataFrame(data)
        data_df.to_excel(xlsfile, sheet_name='index', float_format='%.5f', header=form_header, index=False,
                         index_label=False)  # float_format control float
        xlsfile.close()
