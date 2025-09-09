#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LoadDistribute.py
@Time    :   2025/03/10 16:41:24
@Author  :   勾子瑭
'''
"""
用来统一进行牵引站的负载分配，与 process & main 函数分开使用
"""
# here put the import lib
"""
通过反比进行负载分配
"""

import numpy as np
import gurobipy as gp
import pandas as pd
import matplotlib.pyplot as plt

from Funcs.LD import loadDistributePowerFlowOrg
from MyProcess import Process
from Funcs.controlParams import Params

def loadDistributeFanbi(DFs, sec_interval, params, save, address_output):
    # res = pd.DataFrame(columns=['name', 'class', 'upPre', 'upPost', 'downPre', 'downPost', 'preTss',
    #    'distance_preTss', 'postTss', 'distance_postTss', 'P', 'load', 'location',
    #    'distance_upPre', 'distance_upPost', 'distance_downPre',
    #    'distance_downPost', 'stopping', 'upStop', 'downStop', 'E', 'Rs', 'V',
    #    'I_rated'])
    load_avg = pd.DataFrame(columns=[f'N{i}' for i in range(1, params.nums_Tss+1)])
    cnt = 0

    for i, df in enumerate(DFs):
        # print(f'第{i}次迭代')
        p = df
        p['load']=0

        if cnt==0:
            load_avg.loc[load_avg.__len__(),:] = 0 # 创建一个新的行
        cnt += 1
        for i in p.index:
            # 牵引站 stopping P
            if p.loc[i,'class'] == 1:
                if p.loc[i,'stopping'] == 1:
                    continue
                p.loc[i,'load'] += params.stoppingP if p.loc[i,'upStop'] > 0 else 0
                p.loc[i,'load'] += params.stoppingP if p.loc[i,'downStop'] > 0 else 0
                continue
            # 机车功率按照反比分配到两侧的牵引站上
            P_train = p.loc[i,'P']
            preTss = p.loc[p['name']==p.loc[i,'preTss']].index
            distance_preTss = p.loc[i,'distance_preTss']
            postTss = p.loc[p['name']==p.loc[i,'postTss']].index
            distance_postTss = p.loc[i,'distance_postTss']
            # print(P_train,preTss,distance_preTss,postTss,distance_postTss)
            p.loc[preTss,'load'] += P_train*distance_postTss/(distance_postTss+distance_preTss)
            p.loc[postTss,'load'] += P_train*distance_preTss/(distance_postTss+distance_preTss)
        # print('新增功率为：',p.loc[p['class']==1,'load'].values)
        load_avg.iloc[load_avg.__len__()-1,:] += p.loc[p['class']==1,'load'].values
        if cnt == sec_interval:
            cnt =0
        # print(p['load'])
        # res = pd.concat([res,p],axis=0, ignore_index=True)
    # 结束之后，整体取平均
    load_avg = load_avg/sec_interval

    if save == True:
        load_avg.to_csv(address_output, index = False)
    return load_avg
