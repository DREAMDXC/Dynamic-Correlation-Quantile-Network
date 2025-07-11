"""
@Date: 2023/04/22

@Author: DREAM_DXC

@Summary: define a figure plot for day-ahead scenario generation

"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--test_number", type=int, default=0, help="index of test sample")
parser.add_argument("--test_start", type=int, default=0, help="index of start test sample")
parser.add_argument("--test_end", type=int, default=7, help="index of end test sample")
parser.add_argument("--target_step", type=int, default=24, help="sample time step")

args = parser.parse_args()
print(args)

def Qunatile(scenario,forecast,target):

    # numpy.reshape(a, newshape, order='C')
    # ‘C’ means read in line order, with the last axis index changing fastest
    # ‘F’ means read by column order, with the first index changing fastest

    # scenario size:(scenario number:N,farms_number:I, target_step:J*B)
    scenario = scenario.transpose(1, 2, 0, 3).reshape((scenario.shape[1], scenario.shape[2],
                                                       scenario.shape[3] * scenario.shape[0]), order='C')

    # target size:(farms_number:I,target_step:J*B)
    target = target.transpose(1, 0, 2).reshape((target.shape[1], target.shape[2] *
                                                target.shape[0]), order='C')

    # forecast size:(farms_number:I,target_step:J*B)
    forecast = forecast.transpose(1, 0, 2).reshape((forecast.shape[1], forecast.shape[2] *
                                                    forecast.shape[0]), order='C')

    Quantile = np.sort(scenario, axis=0)
    interval = scenario.shape[0] // 20
    Quantile_dot = np.arange(interval, scenario.shape[0], interval) - 1
    Quantile_line = Quantile[Quantile_dot, :, :]

    return Quantile_line,scenario,forecast,target

def probabilistic_plot(Quantile_line,forecast,target):
    # quantile size (quantile number, farm number, time step)

    # all samples plot
    for j in range(target.shape[0]):

        # plot Quantile figure
        plt.subplot(target.shape[0], 1, j + 1)
        plt.title("farm %s" % (j + 1))
        color = ['0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1']
        x = np.linspace(1, target.shape[1], target.shape[1])

        for i in range(9):
            plt.fill_between(x, Quantile_line[i, j, :], Quantile_line[18 - i, j, :], facecolor=color[i],
                             label=("%s" % (90 - i * 10)))
        plt.plot(x, target[j, :], linewidth=2.0)
        plt.plot(x, forecast[j, :], linewidth=2.0)
        plt.legend()
        plt.show()

    # one sample plot
    for j in range(target.shape[0]):

        # plot Quantile figure
        plt.subplot(target.shape[0], 1, j + 1)
        # plt.title("farm %s" % (j + 1))
        color = ['0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1']
        x = np.linspace(1, args.target_step, args.target_step)

        for i in range(9):
            plt.fill_between(x, Quantile_line[i, j,
                                args.test_number * args.target_step:(args.test_number + 1) * args.target_step]
                             , Quantile_line[18 - i, j,
                               args.test_number * args.target_step:(args.test_number + 1) * args.target_step],
                             facecolor=color[i], label=("%s" % (90 - i * 10)))
        plt.plot(x, target[j, args.test_number * args.target_step:(args.test_number + 1) * args.target_step], linewidth=2.0)
        plt.plot(x, forecast[j, args.test_number * args.target_step:(args.test_number + 1) * args.target_step], linewidth=2.0)
        plt.legend()

        plt.savefig('Sample probabilistic plot.svg', dpi=600, format='svg')
        plt.show()

    # Continuous sample test plot
    for j in range(target.shape[0]):

        # plot Quantile figure
        # plt.subplot(target.shape[0], 1, j + 1)
        plt.figure(figsize=(6, 3))
        # plt.title("farm %s" % (j + 1))
        color = ['0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1']
        x = np.linspace(1, args.target_step*(args.test_end-args.test_start), args.target_step*(args.test_end-args.test_start))
        for i in range(9):
            plt.fill_between(x, Quantile_line[i, j, args.test_start*args.target_step:args.test_end*args.target_step]
                             , Quantile_line[18 - i, j, args.test_start*args.target_step:args.test_end*args.target_step],
                             facecolor=color[i],label=("%s" % (90 - i * 10)))
        plt.plot(x, target[j, args.test_start*args.target_step:args.test_end*args.target_step], linewidth=2.0)
        plt.plot(x, forecast[j, args.test_start*args.target_step:args.test_end*args.target_step], linewidth=2.0)
        plt.legend()

        plt.xlim((1, x.shape[0]))
        plt.ylim((0, 1))

        plt.xlabel('time/h')
        plt.ylabel('p.u.')

        plt.savefig('Continuous sample probabilistic plot.svg', dpi=600, format='svg')
        plt.show()

def scenario_plot(scenario,forecast,target):

    # scenario size (scenario number, farm number, time step)

    # one sample plot
    time = np.linspace(1, args.target_step, args.target_step)

    for j in range(scenario.shape[1]):
        plt.subplot(scenario.shape[1], 1, j + 1)
        plt.title("farm %s" % (j + 1))
        for i in range(scenario.shape[0]):

            plt.plot(time,scenario[i, j, args.test_number * args.target_step:(args.test_number + 1) * args.target_step]
                     , linewidth=0.2, color='silver')

    plt.plot(time, target[j, args.test_number * args.target_step:(args.test_number + 1) * args.target_step], linewidth=2.0)
    plt.plot(time, forecast[j, args.test_number * args.target_step:(args.test_number + 1) * args.target_step],
             linewidth=2.0)
    plt.savefig('Sample scenario plot.svg', dpi=600, format='svg')
    plt.show()

    xlsfile = pd.ExcelWriter('scenario.xlsx')
    data_df = pd.DataFrame(scenario[:, j, args.test_number * args.target_step:(args.test_number + 1) * args.target_step])
    data_df.to_excel(xlsfile, 'scenario', float_format='%.5f', header=None, index=False, index_label=False)
    xlsfile.close()

    # Continuous sample test plot
    plt.figure(figsize=(6, 3))
    time = np.linspace(1, args.target_step*(args.test_end-args.test_start), args.target_step*(args.test_end-args.test_start))

    for j in range(scenario.shape[1]):
        plt.subplot(scenario.shape[1], 1, j + 1)
        # plt.title("farm %s" % (j + 1))
        for i in range(scenario.shape[0]):
            plt.plot(time, scenario[i, j, args.test_start*args.target_step:args.test_end*args.target_step]
                     , linewidth=0.2, color='silver')

    plt.plot(time, target[j, args.test_start*args.target_step:args.test_end*args.target_step],
             linewidth=2.0)
    plt.plot(time, forecast[j, args.test_start*args.target_step:args.test_end*args.target_step],
             linewidth=2.0)

    plt.xlim((1, time.shape[0]))
    plt.ylim((0, 1))

    plt.xlabel('time/h')
    plt.ylabel('p.u.')

    plt.savefig('Continuous scenario plot.svg', dpi=300, format='svg')
    plt.show()

def mian_plot(scenario,forecast,target):

    Quantile_line,scenario,forecast,target = Qunatile(scenario, forecast, target)

    probabilistic_plot(Quantile_line, forecast, target)
    scenario_plot(scenario, forecast, target)

def mian_quantile_plot(Quantile_line,scenario,forecast,target):
    Quantile_line = np.expand_dims(Quantile_line,axis=2)
    Quantile_line = Quantile_line.transpose(1, 2, 0, 3).reshape((Quantile_line.shape[1], Quantile_line.shape[2],
                                                                 Quantile_line.shape[3] * Quantile_line.shape[0]), order='C')

    # scenario size:(scenario number:N,farms_number:I, target_step:J*B)
    scenario = scenario.transpose(1, 2, 0, 3).reshape((scenario.shape[1], scenario.shape[2],
                                                       scenario.shape[3] * scenario.shape[0]), order='C')

    # target size:(farms_number:I,target_step:J*B)
    target = target.transpose(1, 0, 2).reshape((target.shape[1], target.shape[2] *
                                                target.shape[0]), order='C')

    # forecast size:(farms_number:I,target_step:J*B)
    forecast = forecast.transpose(1, 0, 2).reshape((forecast.shape[1], forecast.shape[2] *
                                                    forecast.shape[0]), order='C')

    probabilistic_plot(Quantile_line, forecast, target)
    scenario_plot(scenario, forecast, target)