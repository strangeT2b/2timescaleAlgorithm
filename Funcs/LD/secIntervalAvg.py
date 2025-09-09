#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   secIntervalAvg.py
@Time    :   2025/03/15 22:24:50
@Author  :   勾子瑭
'''

# here put the import lib
import numpy as np
import gurobipy as gp
import pandas as pd
import matplotlib.pyplot as plt

def secIntervalAvg(secInterval, data):
    # data_avg = pd.DataFrame(columns=[f'N{i}' for i in range(2)])
    # cnt = 0
    # for i in range(data.__len__()):
    #     if cnt ==0:
    #         data_avg.loc[data_avg.__len__(),:] = 0 # 创建一个新的行
    #     cnt += 1
    #     data_avg.loc[data_avg.__len__()-1,:] += data.loc[i,:].values
    #     if cnt == secInterval:
    #         cnt = 0

    # # 结束之后，整体取平均
    # data_avg = data_avg/secInterval
    # return data_avg

    data.reset_index(drop = True, inplace = True)
    data['groupBy'] = data.index.values.astype('int')//secInterval
    data = data.groupby('groupBy').mean().reset_index(drop = True)
    # data.drop(columns = ['groupBy'], inplace = True)
    return data
    pass