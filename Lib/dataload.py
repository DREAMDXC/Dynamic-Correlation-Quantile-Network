"""
@Date: 2024/02/04

@Author: DREAM_DXC

@Summary: define a dataload for short-term forecast GEFCOM

"""

import os
import torch
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--interval", type=int, default=1, help="define the interval (The original data is sampled 15 minutes)")
parser.add_argument("--sample_step", type=int, default=24, help="define the sampling step")
parser.add_argument("--input_step", type=int, default=24, help="define the input temporal step")
parser.add_argument("--target_step", type=int, default=24, help="define the target temporal step")

args = parser.parse_args()
print(args)

def norm_speed(data):
    where_are_nan = np.isnan(data)
    data[where_are_nan] = 0  # delete data if data = nan

    speed1 = np.sqrt(np.power(data[:,0:1],2)+np.power(data[:,1:2],2))
    speed2 = np.sqrt(np.power(data[:, 2:3], 2) + np.power(data[:, 3:4], 2))
    sin_speed1 = np.sin(data[:,1:2]/speed1)
    cos_speed1 = np.cos(data[:,0:1]/speed1)
    sin_speed2 = np.sin(data[:,3:4]/speed2)
    cos_speed2 = np.cos(data[:,2:3]/speed2)
    data = np.concatenate((speed1,speed2, sin_speed1, cos_speed1,sin_speed2, cos_speed2), axis=1)

    return data

def Data_Preproccess(category,index,interval,step,input_time_step,target_time_step,data_file_path):

    farm_number = len(index)

    if category == 'Wind':
        for i in range(farm_number):
            WIND_fileNameList = "\GEFCOMWindData\GEFCom2014zone{index}.csv".format(index=index[i])
            WIND_data = pd.read_csv(data_file_path + WIND_fileNameList).values

            if i == 0:
                POWER = WIND_data[:, 2:3]
                NWP = WIND_data[:, 3:]  # 每个场站有4个NWP特征
                NWP = norm_speed(np.array(NWP, dtype=np.float64))

            if i > 0:
                POWER = np.concatenate((POWER, WIND_data[:, 2:3]), axis=1)
                X = norm_speed(np.array(WIND_data[:, 3:], dtype=np.float64))
                NWP = np.concatenate((NWP, X), axis=1)


    NWP = np.array(NWP, dtype=np.float64)
    where_are_nan = np.isnan(NWP)
    print('NWP nan number:', len(NWP[where_are_nan]))

    NWP,NWP_max,NWP_min = norm(NWP)
    where_are_nan = np.isnan(NWP)
    print('After data cleaning NWP nan number:', len(NWP[where_are_nan]))

    POWER = np.array(POWER, dtype=np.float64)
    where_are_nan = np.isnan(POWER)
    print('POWER nan number:',len(POWER[where_are_nan]))
    POWER = pd.DataFrame(POWER)
    POWER = POWER.fillna(method='bfill').values
    POWER = np.clip(POWER, a_min=0 , a_max=None)
    POWER = np.array(POWER, dtype=np.float64)
    where_are_nan = np.isnan(POWER)
    print('After data cleaning POWER nan number:', len(POWER[where_are_nan]))

    Len_data = POWER.shape[0]  # data length
    feature_number = NWP.shape[1]
    x = np.arange(0, Len_data, interval)
    POWER = POWER[x]  # get the interval data
    NWP = NWP[x]

    num = (( Len_data - target_time_step) // step) + 1  # the finally data cant use to train ,lack the target data ,so lack 1 num
    print("sample-number:",num)

    # get nwp input data
    input_node = np.arange(0, input_time_step, 1)
    y = input_node
    for i in range(1, num):
        y = np.append(y, input_node + i * step, axis=0)
    Input_NWP_data = NWP[y]
    Input_NWP_data = torch.from_numpy(Input_NWP_data).float()
    Input_NWP_data = Input_NWP_data.view(num, input_time_step, feature_number)
    Input_NWP_data = Input_NWP_data.permute(0, 2, 1) # (B,f*10,16)
    Input_NWP_data = Input_NWP_data.clamp(min=1e-6,max=1-(1e-6)) # for numerical stability

    # get power target data
    Target_data = POWER[y]
    Target_data = Target_data # 归一化
    Target_data = torch.from_numpy(Target_data).float()
    Target_data = Target_data.view(num, target_time_step, farm_number)
    Target_data = Target_data.permute(0, 2, 1) # (B,10,16)
    Target_data = Target_data.clamp(min=1e-6,max=1-(1e-6)) # for numerical stability

    # Train-Valid-Test 731 sample

    Valid_node = 438
    Test_node = 511

    Normfactor = {}

    Train_input = Input_NWP_data[0:Valid_node,:,:]
    Train_target = Target_data[0:Valid_node, :, :]

    Valid_input = Input_NWP_data[Valid_node:Test_node,:,:]
    Valid_target = Target_data[Valid_node:Test_node, :, :]

    Test_input = Input_NWP_data[Test_node:,:,:]
    Test_target = Target_data[Test_node:, :, :]

    Normfactor['nwp_max'] = NWP_max
    Normfactor['nwp_min'] = NWP_min

    return Train_input,Train_target,Valid_input,Valid_target,Test_input,Test_target,Normfactor

def norm(data):

    min = np.amin(data,axis=0)
    max = np.amax(data,axis=0)
    data = (data - min) / (max - min)

    return data,max,min

def dataloder(category):
    pwd = os.getcwd()
    grader_father = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")

    index = ['1']

    Train_input,Train_target,Valid_input,Valid_target,Test_input,Test_target,Normfactor  = Data_Preproccess(category, index, args.interval,
                                                                          args.sample_step, args.input_step,
                                                                          args.target_step,
                                                                          grader_father)

    return Train_input,Train_target,Valid_input,Valid_target,Test_input,Test_target
