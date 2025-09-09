#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   power_flow_org.py
@Time    :   2025/02/20 20:26:54
@Author  :   
'''

# here put the import lib
import time
import re
import numpy as np
import gurobipy as gp
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import copy
def power_flow_checkOnce(checkData, checkTime, DFs, DFs_noStop, Y, params, feasibleTol = 0.1):
    """
    太精确的话，等式约束貌似没有办法严格成立，所以下面我们考虑偏差小雨0.001(0.1%)就算成立
    """
    tempData = copy.deepcopy(checkData)
    df = DFs[checkTime]
    df = df[['name', 'class', 'P', 'location', 'stopping', 'upStop', 'downStop']]
    df_noStop = DFs_noStop[checkTime]
    df_noStop = df_noStop[['name', 'class', 'P', 'location', 'stopping', 'upStop', 'downStop']]
    yMatrix = Y[checkTime]

    for key, value in tempData.items():
        # if 'H' not in str(value.index[0]):
        #     value.index = value.index.map(lambda x : 'H'+str(x))
        # if key not in ['vs', 'p_ins', 'SOCs', 'p_gs', 'p_bs', 'p_chs', 'p_dischs']:
        #     continue
        # if key in ['vs', 'p_ins', 'SOCs', 'p_chs', 'p_dischs']:
        #     value.columns = value.columns.map(lambda x: int(x.split('_N')[1]))  # 右侧提取数字并转整数
        value = value.loc[checkTime,:]
        # 如果确认所有name都可以转为整数
        df['name'] = df['name'].astype(int)          # 左侧保持整数
        df_noStop['name'] = df_noStop['name'].astype(int)          # 左侧保持整数
        value.name = key
        df = df.join(value, on='name')
        df_noStop = df_noStop.join(value, on='name')
        
    checkMap = dict()
    #-------- 电压限值检测
    isFeasibleTssVoltage = all((params.v_tss_min-feasibleTol<=df.loc[idx, 'vs'] and df.loc[idx, 'vs']<=params.v_tss_max+feasibleTol) for idx in df.index if df.loc[idx,'class']==1)
    isFeasibleTrainVoltage = all((params.v_train_min-feasibleTol<=df.loc[idx, 'vs'] and df.loc[idx, 'vs'] <= params.v_train_max+feasibleTol) for idx in df.index if df.loc[idx,'class']==0)
    checkMap['牵引站电压限值'] = list((idx, df.loc[idx, 'vs']) for idx in df.index if df.loc[idx,'class']==1)
    checkMap['机车电压限值'] = list((idx, df.loc[idx, 'vs']) for idx in df.index if df.loc[idx,'class']==0)
    
    #-------- 功率限值检测
    # 牵引站反向潮流检测
    isFeasibleTssP = all((df.loc[idx, 'p_ins']>=-feasibleTol) for idx in df.index if df.loc[idx,'class']==1)
    checkMap['牵引站反向潮流'] = list((idx, df.loc[idx, 'p_ins']) for idx in df.index if df.loc[idx,'class']==1)

    #-------- 电池充放电功率限值检测
    isFeasibleBatteryCharge = all((df.loc[idx, 'p_chs']>=-feasibleTol and df.loc[idx, 'p_chs'] <= params.p_b_max+feasibleTol) for idx in df.index if df.loc[idx,'class']==1)
    isFeasibleBatteryDisCharge = all((df.loc[idx, 'p_dischs']>=params.p_b_min-feasibleTol and df.loc[idx, 'p_dischs'] <= feasibleTol) for idx in df.index if df.loc[idx,'class']==1)
    isFeasibleToBattery = all((df.loc[idx, 'p_bs']>=params.p_b_min-feasibleTol and df.loc[idx, 'p_bs'] <= params.p_b_max+feasibleTol) for idx in df.index if df.loc[idx,'class']==1)
    checkMap['电池充电限值'] = list((idx, df.loc[idx, 'p_chs']) for idx in df.index if df.loc[idx,'class']==1)
    checkMap['电池放电限值'] = list((idx, df.loc[idx, 'p_dischs']) for idx in df.index if df.loc[idx,'class']==1)
    checkMap['光伏->电池限值'] = list((idx, df.loc[idx, 'p_bs']) for idx in df.index if df.loc[idx,'class']==1)

    #-------- 潮流方程检测（重要
    # 功率电压检测 p_in = E*(E-V)/R
    # isFeasiblePinAndVoltage = all(abs(params.E*(params.E-df.loc[idx,'vs'])/params.Rs-df.loc[idx,'p_ins'])/df.loc[idx,'p_ins'] <= 0.001 for idx in df.index if df.loc[idx,'class']==1)
    isFeasiblePinAndVoltage = all(
        (abs(params.E*(params.E-df.loc[idx,'vs'])/params.Rs - df.loc[idx,'p_ins']) <= feasibleTol)
        for idx in df.index if df.loc[idx,'class'] == 1)
    checkMap['牵引站输出功率方程'] = list(
        (params.E*(params.E-df.loc[idx,'vs'])/params.Rs) 
        for idx in df.index if df.loc[idx,'class'] == 1)

    ## 牵引站潮流
    isFeasibleTssPowerFlow = all(abs(-sum(yMatrix[idx,j]*df_noStop.loc[idx,'vs']*df_noStop.loc[j,'vs'] for j in df_noStop.index)- \
                                      (df_noStop.loc[idx,'vs']*(params.E-df_noStop.loc[idx,'vs'])/params.Rs \
                                    + df_noStop.loc[idx,'p_gs'] \
                                    - params.stoppingP*(int(df_noStop.loc[idx,'upStop']!=0)+int(df_noStop.loc[idx,'downStop']!=0))))<feasibleTol\
                                    for idx in df_noStop.index if df_noStop.loc[idx,'class']==1)
    checkMap['牵引站潮流'] = list(abs(-sum(yMatrix[idx,j]*df_noStop.loc[idx,'vs']*df_noStop.loc[j,'vs'] for j in df_noStop.index)- \
                                      (df_noStop.loc[idx,'vs']*(params.E-df_noStop.loc[idx,'vs'])/params.Rs \
                                    + df_noStop.loc[idx,'p_gs'] \
                                    - params.stoppingP*(int(df_noStop.loc[idx,'upStop']!=0)+int(df_noStop.loc[idx,'downStop']!=0))))\
                                    for idx in df_noStop.index if df_noStop.loc[idx,'class']==1)

    ## 机车潮流
    isFeasibleTrainPowerFlow =  all(abs(sum(yMatrix[idx,j]*df_noStop.loc[idx,'vs']*df_noStop.loc[j,'vs'] for j in df_noStop.index)- \
                                      df_noStop.loc[idx,'P']) <= feasibleTol \
                                    for idx in df_noStop.index if df_noStop.loc[idx,'class']==0)
    checkMap['机车潮流'] = list(abs(sum(yMatrix[idx,j]*df_noStop.loc[idx,'vs']*df_noStop.loc[j,'vs'] for j in df_noStop.index)- \
                                      df_noStop.loc[idx,'P']) \
                                    for idx in df_noStop.index if df_noStop.loc[idx,'class']==0)

    res = isFeasibleTssVoltage and isFeasibleTrainVoltage 
    res = res and isFeasibleTssP 
    res = res and isFeasibleBatteryCharge and isFeasibleBatteryDisCharge and isFeasibleToBattery
    res = res and isFeasiblePinAndVoltage
    res = res and isFeasibleTssPowerFlow and isFeasibleTrainPowerFlow

    isFeasibleMap = {'牵引站电压限值':isFeasibleTssVoltage
                     ,'机车电压限值':isFeasibleTrainVoltage
                     ,'牵引站反向潮流':isFeasibleTssP
                     ,'电池充电限值':isFeasibleBatteryCharge
                     ,'电池放电限值':isFeasibleBatteryDisCharge
                     ,'光伏->电池限值':isFeasibleToBattery
                     ,'牵引站输出功率方程':isFeasiblePinAndVoltage
                     ,'牵引站潮流':isFeasibleTssPowerFlow
                     ,'机车潮流':isFeasibleTrainPowerFlow}
    resDF = pd.DataFrame(isFeasibleMap, index = [checkTime])
    # return res, isFeasibleMap, df
    if res:
        return res, resDF
    else:
        print(isFeasibleMap)
        for k,v in isFeasibleMap.items():
            if not v:
                print(checkMap[k])
        return res, resDF
