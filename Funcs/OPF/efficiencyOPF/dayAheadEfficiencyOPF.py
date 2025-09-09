#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dayAhead_OPF_avg.py
@Time    :   2025/03/04 22:01:35
@Author  :   勾子瑭
'''

# here put the import lib
"""
对 dayAhead_OPF_org.py 做了修改，由于 day ahead dispatch 只需要每个牵引站的粗时间粒度的功率平衡方程，因此不在需要繁杂的 DFs(牵引网络拓扑信息)
只需要保留时间粒度，每个牵引站的平均功率即可,免去了繁杂的冗余特征信息 命名为 load_avg.csv 
index | avg_pIn_N1 | avg_pIn_N2 |...
而对于如何产生 load_avg.csv 则需要根据不同的取平均的方法分别定义不同的函数

此函数 接收平均化的 PV 以及 牵引站负载，约束条件就是最简单的、只包含牵引站套利的线性方程
"""

import gurobipy as gp
import re
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

def dayAhead_OPF_efficiency_avg( schedule_horizon, load_avg, PV_Price, params, delta_t = None, in_PV = False, schedule_start=0, plot=False, path=""):

    current_t = schedule_start
    horizon = schedule_horizon
    current_t += 0
    print('第 {} 日前调度开始\n'.format(current_t))
    # DF and Y computing time
    time_DF_read = time.time() # DF and Y computing time
    if delta_t == None:
        delta_t = params.delta_t

    # horizon list 
    # if in_PV == True:
    #     dfs_pv = PV_Price.loc[current_t:current_t+horizon-1, :].values # convert to unit MW
    dfs_pv = PV_Price.loc[current_t:current_t+horizon-1,:]
    dfs_pv.reset_index(drop=True, inplace = True)

    if "PRICE" in dfs_pv.columns:
        dfs_pv.drop(columns=["PRICE"], inplace = True)
    if "time" in dfs_pv.columns:
        dfs_pv.drop(columns=["time"], inplace = True)
    dfs_pv.columns = dfs_pv.columns.astype("int")
    
    df_price = PV_Price.loc[current_t:current_t+horizon-1, 'PRICE'].values
    load_avg = load_avg.loc[current_t:current_t+horizon-1,:]
    time_df_read = time.time() # dfs time

    ####### GUROBI #######
    nums_noStop = np.zeros(shape=horizon,dtype='int64')

    soc_0 = np.ones(params.nums_Tss)*50
    
    # print('shape of P is {}\n'.format(np.shape(P)))
    print('every t nums_noStop is {}'.format(nums_noStop))

    # model
    model = gp.Model()

    # var [[[] for i in range(nums_noStop[t])] for t in range(horizon)]
    v = {}
    soc = {}
    p_b = {}
    p_ch = {}
    p_disch = {}
    p_g = {}
    p_in = {}

    obj_expr = gp.LinExpr()
    obj_L1 = gp.LinExpr()
    l_expr = gp.LinExpr()
    print(delta_t)

    for t in range(horizon):
        # variables
        # 由于与潮流计算已经无关，因此只需要对牵引站部分进行建模
        for i in range(len(load_avg.columns)):
            soc[t,i] = model.addVar(lb=params.soc_min, ub=params.soc_max, vtype=gp.GRB.CONTINUOUS, name='SOC_{}_H{}'.format(load_avg.columns[i],t))
            p_b[t,i] = model.addVar(lb=params.p_b_min, ub=params.p_b_max, vtype=gp.GRB.CONTINUOUS, name='P_Battery_{}_H{}'.format(load_avg.columns[i],t))
            p_ch[t,i] = model.addVar(lb=params.p_ch_min*0.9, ub=params.p_ch_max*0.9, vtype=gp.GRB.CONTINUOUS, name='P_Battery_Charge_{}_H{}'.format(load_avg.columns[i],t))
            p_disch[t,i] = model.addVar(lb=params.p_disch_min*0.9, ub=params.p_disch_max*0.9, vtype=gp.GRB.CONTINUOUS, name='P_Battery_DisCharge_{}_H{}'.format(load_avg.columns[i],t))

            p_g[t,i] = model.addVar(lb=params.p_g_min, ub=params.p_g_max, vtype=gp.GRB.CONTINUOUS, name='P_to_Grid_{}_H{}'.format(load_avg.columns[i],t))
            p_in[t,i] = model.addVar(lb=params.p_in_min, ub=params.p_in_max, vtype=gp.GRB.CONTINUOUS, name='P_from_Source_{}_H{}'.format(load_avg.columns[i],t))
            v[t,i] = model.addVar(lb=params.v_tss_min, ub=params.v_tss_max, name='Volt_{}_H{}'.format(str(load_avg.columns[i]),t))
            ## objective function expr
            obj_expr.addTerms(df_price[t]*delta_t,p_in[t,i])
            ## obj L1
            obj_L1.addTerms(delta_t, p_ch[t,i])
            obj_L1.addTerms(delta_t, p_disch[t,i])

        # constraints
        for i in range(len(load_avg.columns)):

            ### 光伏发电和电池充放电效率损失
            # p_b = p_ch/e + p_disch*e
            model.addConstr(p_b[t,i] == p_ch[t,i]/params.battery_efficiency + p_disch[t,i]*params.battery_efficiency, name='PB_charge_discharge_N{}_H{}'.format(load_avg.columns[i],t))
            ### soc
            if t == 0:
                model.addConstr(soc[t,i] == soc_0[i] + 100*((p_ch[t,i]+p_disch[t,i])*delta_t)/(params.battery_max-params.battery_min), name='SOC_change_N{}_H{}'.format(load_avg.columns[i],t))
            elif t >= 1:
                model.addConstr(soc[t,i] == soc[t-1,i] + 100*((p_ch[t,i]+p_disch[t,i])*delta_t)/(params.battery_max-params.battery_min), name='SOC_change_N{}_H{}'.format(load_avg.columns[i],t))

            # soc_sta = soc_end
            if t == horizon-1:
                model.addConstr(soc[t,i] == soc_0[i], name = 'initial_soc=end_soc_{}'.format(load_avg.columns[i]))

            ### pv >= p + g
            l_expr.addTerms(-1, p_b[t,i])
            l_expr.addTerms(-1, p_g[t,i])
            l_expr.addConstant(dfs_pv.iloc[t,i])
            model.addConstr(l_expr == 0, name='PV_Balance_{}_H{}'.format(load_avg.columns[i],t))
            l_expr.clear()

            ### b + g >= 0
            model.addConstr(p_b[t,i] + p_g[t,i] >= min(0, dfs_pv.iloc[t,i]), name='BESS_Feasible_{}_H{}'.format(load_avg.columns[i],t))

            ### p_in
            model.addConstr(p_in[t,i]*params.Rs == params.E*(params.E - v[t,i]), name='P_IN_{}_H{}'.format(load_avg.columns[i],t))

            ### power load_avg balance equation of station, instead of power flow
            if in_PV:
                p_sum = dfs_pv.iloc[t,i]
            else:
                p_sum = 0
            # PV 好像算重复了啊？
            p_sum = 0
            l_expr.addConstant(p_sum) # l_expr = pv + p_in + p_g  - (stopping)load_avg

            l_expr.addTerms(1, p_in[t,i]) # p_in
            l_expr.addTerms(1,p_g[t,i]) # p_g

            l_expr.addConstant(-load_avg.iloc[t,i])

            model.addConstr(l_expr == 0, name='PowerEq_N{}_H{}'.format(load_avg.columns[i],t))
            l_expr.clear()

    # objective function
    model.setObjective(params.realTime_weight_p_in * obj_expr + params.realTime_weight_L1 * obj_L1, sense=gp.GRB.MINIMIZE)
    time_model = time.time()

    model.setParam('OutputFlag',1)
    model.setParam('PreSolve',2)
    model.setParam('NumericFocus',3)
    model.setParam('FeasibilityTol',1e-8)
    model.params.MIPGap = params.grb_gap

    # model.write("/Users/town/Desktop/STU/Code/Jupyter_Code/Master/Main/2Stage/Funcs/OPF/efficiencyOPF/log/dayAhead.LP")
    if path != "":
        model.setParam('LogFile', path)
    model.optimize()
    print(model.Status)
    time_slove = time.time()


    # ------ 获取求解信息
    # # print('the objective value is : {}'.format(model.ObjVal))
    vs = pd.DataFrame(columns=['Volt_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    SOCs = pd.DataFrame(columns=['SOC_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    SOCs.iloc[0:] = soc_0
    p_gs = pd.DataFrame(columns=['P_to_Grid_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    p_ins = pd.DataFrame(columns=['P_from_Source_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    p_bs = pd.DataFrame(columns=['P_Battery_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    p_chs = pd.DataFrame(columns=['P_Battery_Charge_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    p_dischs = pd.DataFrame(columns=['P_Battery_DisCharge_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    p_bras = pd.DataFrame(columns=['P_brake_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    for x in model.getVars():
        if 'Volt' in x.VarName:
            match = re.match(r'Volt_N(\d+)_H(\d+)',x.VarName)
            vs.loc[int(match.group(2)),'Volt_N'+match.group(1)] = x.X
        elif 'SOC' in x.VarName:
            match = re.match(r'SOC_N(\d+)_H(\d+)',x.VarName)
            SOCs.loc[int(match.group(2)),'SOC_N'+match.group(1)] = x.X
        elif 'P_to_Grid' in x.VarName:
            match = re.match(r'P_to_Grid_N(\d+)_H(\d+)',x.VarName)
            p_gs.loc[int(match.group(2)),'P_to_Grid_N'+match.group(1)] = x.X
        elif 'P_from_Source' in x.VarName:
            match = re.match(r'P_from_Source_N(\d+)_H(\d+)',x.VarName)
            p_ins.loc[int(match.group(2)),'P_from_Source_N'+match.group(1)] = x.X
        elif 'P_Battery_N' in x.VarName:
            match = re.match(r'P_Battery_N(\d+)_H(\d+)',x.VarName)
            p_bs.loc[int(match.group(2)),'P_Battery_N'+match.group(1)] = x.X
        elif 'P_Battery_Charge' in x.VarName:
            match = re.match(r'P_Battery_Charge_N(\d+)_H(\d+)',x.VarName)
            p_chs.loc[int(match.group(2)),'P_Battery_Charge_N'+match.group(1)] = x.X
        elif 'P_Battery_DisCharge' in x.VarName:
            match = re.match(r'P_Battery_DisCharge_N(\d+)_H(\d+)',x.VarName)
            p_dischs.loc[int(match.group(2)),'P_Battery_DisCharge_N'+match.group(1)] = x.X
        elif 'P_brake' in x.VarName:
            match = re.match(r'P_brake_N(\d+)_H(\d+)',x.VarName)
            p_bras.loc[int(match.group(2)),'P_brake_N'+match.group(1)] = x.X
        else:
            print('遗漏了变量')
    SOCs.reset_index(inplace=True, drop = True)
    SOCs.index = (range(1, 1+len(SOCs)))
    SOCs.loc[0,:] = soc_0
    SOCs.sort_index(inplace=True)

    time_list = [ time_df_read-time_DF_read
                , time_model-time_df_read
                , time_slove-time_model]
    # print('读取 DFs 耗时: {} s\n计算 dfs 和 Y 耗时: {} s\n建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[0],time_list[1],time_list[2],time_list[3]))
    print('建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[1],time_list[2]))

    model.dispose()

    return {'vs':vs
            ,'p_ins': p_ins
            ,'SOCs': SOCs
            ,'p_gs': p_gs
            ,'p_bs': p_bs
            ,'p_chs': p_chs
            ,'p_dischs': p_dischs
            ,'p_bras': p_bras}
