#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LD_PF_org.py
@Time    :   2025/03/10 15:59:23
@Author  :   勾子瑭
'''

"""
利用 nonlinear power flow 计算出每一时刻牵引站的负载大小
"""
# here put the import lib
import time
import re
import numpy as np
import gurobipy as gp
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from Funcs.admittanceMatrix import admittanceMatrix


def loadDistributePowerFlowOrg(LD_start, LD_horizon, DFs, DFs_noStop, PV_Price, Ys, params, sec_interval,  in_PV = False, stepByStep = False):
    ### simulation
    current_t = LD_start
    horizon = LD_horizon
    current_t += 0
    # DF and Y computing time
    time_DF_read = time.time() # DF and Y computing time

    # horizon list 
    dfs = DFs[current_t:current_t+horizon]
    if in_PV == True:
        dfs_pv = PV_Price.loc[current_t:current_t+horizon-1, 'PV'].values # convert to unit MW
    dfs_noStop = DFs_noStop[current_t:current_t+horizon]
    # ys = Ys[current_t:current_t+horizon]



    ####### GUROBI #######
    nums_noStop = np.zeros(shape=horizon,dtype='int64')
    nums_Tss = dfs[0]['class'].sum()
    # PF_vs_org = pd.DataFrame(columns=['Volt_N{}'.format(i) for i in range(1,nums_Tss+1)])
    PF_p_ins = pd.DataFrame(columns=['P_from_Source_N{}'.format(i) for i in range(1,nums_Tss+1)]
                            ,index= [i for i in range(LD_horizon)]
                            , dtype='float')
    
    P = {}
    soc_0 = np.ones(nums_Tss)*50

    for t in range(horizon):
        dfs[t].reset_index(drop=True, inplace=True)
        dfs_noStop[t].reset_index(drop=True, inplace=True)

        nums_noStop[t] = len(dfs_noStop[t].index.values)
        P[t] = dfs_noStop[t]['P'].values # convert to unit MW

    for t in range(horizon):

        # model
        model = gp.Model()
        # var [[[] for i in range(nums_noStop[t])] for t in range(horizon)]
        v = {}
        p_in = {}

        obj_expr = gp.LinExpr()
        q_expr = gp.QuadExpr()
        l_expr = gp.LinExpr()


        scaleNum = 1000

        ys = admittanceMatrix(dfs[t], dfs_noStop[t])
        clear_output(wait=True)
        print(f'这是第 {t} 次求解\n')
        # variables
        for i in dfs_noStop[t].index:
            if dfs_noStop[t].loc[i,'class']==1:
                v[t,i] = model.addVar(lb=params.v_tss_min, ub=params.v_tss_max, vtype=gp.GRB.CONTINUOUS, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                p_in[t,i] = model.addVar(lb=params.p_in_min, ub=params.p_in_max, vtype=gp.GRB.CONTINUOUS, name='P_from_Source_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
            elif dfs_noStop[t].loc[i,'class']==0:
                v[t,i] = model.addVar(lb=params.v_tss_min, ub=params.v_tss_max, vtype=gp.GRB.CONTINUOUS, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                pass

        # constraints
        for i in dfs_noStop[t].index:
            if dfs_noStop[t].loc[i,'class'] == 1:
                # ### p_in
                l_expr = gp.LinExpr()
                l_expr.addTerms(params.Rs,p_in[t,i])
                l_expr.addTerms(params.E,v[t,i])
                model.addConstr(scaleNum*l_expr == scaleNum*params.E*params.E, name='P_IN_N{}_H{}'.format(i,t))
                l_expr.clear()

                ### power flow of station
                q_expr.addTerms(-1/params.Rs,v[t,i],v[t,i])
                for j in range(nums_noStop[t]):
                    q_expr.addTerms(ys[i,j],v[t,i],v[t,j])

                # E*V/Rs
                l_expr.addTerms(params.E/params.Rs,v[t,i])
                
                # 牵引站功率（pv - stopping
                if in_PV:
                    p_sum = dfs_pv[t]
                else:
                    p_sum = 0
                p_sum = 0
                if dfs_noStop[t].loc[i,'upStop'] > 0: # stopping train
                    p_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'upStop'],'P'].values[0]
                if dfs_noStop[t].loc[i,'downStop'] > 0 and dfs_noStop[t].loc[i,'downStop'] != dfs_noStop[t].loc[i,'upStop']:
                    p_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'downStop'],'P'].values[0]
                l_expr.addConstant(p_sum)

                model.addQConstr(-q_expr == l_expr, name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                q_expr.clear()
                l_expr.clear()
                p_sum = 0

            elif dfs_noStop[t].loc[i,'class'] == 0:
                ### train braking power balance

                ### power flow of train
                for j in range(nums_noStop[t]):
                    q_expr.addTerms(ys[i,j],v[t,i],v[t,j])
                model.addQConstr(q_expr == P[t][i], name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                q_expr.clear()
                l_expr.clear()

        # objective function
        model.setObjective(1, sense=gp.GRB.MINIMIZE)

        model.setParam('OutputFlag',0)
        model.setParam('NonConvex',2)
        model.params.MIPGap = 0.01

        model.optimize()

        # 获取求解信息
        for x in model.getVars():
            # if 'Volt' in x.VarName:
            #     match = re.match(r'Volt_N(\d+)_H(\d+)',x.VarName)
            #     PF_vs_org.loc[int(match.group(2)),'Volt_N'+match.group(1)] = x.X
            #     # PF_vs_org.append(x.X)
            if 'P_from_Source' in x.VarName:
                match = re.match(r'P_from_Source_N(\d+)_H(\d+)',x.VarName)
                PF_p_ins.loc[int(match.group(2)), 'P_from_Source_N'+match.group(1)] = x.X
                # PF_p_ins.append(x.X)
            # else:
            #     print('遗漏了变量')

        model.dispose()

    #     # stopping train voltage recover
    #     for col in PF_vs_org.columns[11:]:
    #         match = re.match('Volt_N(\d+)',col)
    #         num_train = int(match.group(1))
    #         for idx in PF_vs_org[col].index:
    #             if pd.isna(PF_vs_org.loc[idx,col]):
    #                 match = re.match('H(\d+)',idx)
    #                 num_horizon = int(match.group(1))
    #                 current_DF = DFs[num_horizon]
    #                 if num_train in current_DF['name'].values:
    #                     stoppingTss = current_DF.loc[current_DF['name']==num_train,'stopping'].values[0]
    #                     if stoppingTss > 0 :
    #                         # print(stoppingTss)
    #                         PF_vs_org.loc[idx,col] = PF_vs_org.loc[idx,'Volt_N'+str(stoppingTss)]
    #             else: pass

    # PF_data = {'vs':PF_vs_org.astype('float')
    #            ,'p_ins':PF_p_ins.astype('float')}
    
    PF_data = {'p_ins':PF_p_ins.astype('float')}

    print('---------- 潮流计算完成，开始分配负载 ----------\n')
    load_avg = PF_data['p_ins'].reset_index(drop=True)
    load_avg.columns = [f'N{i}' for i in range(1, params.nums_Tss+1)]
    load_avg.reset_index(drop=True, inplace=True)
    load_avg['groupBy'] = load_avg.index.values.astype('int')//sec_interval
    load_avg.groupby('groupBy').mean().reset_index(drop=True, inplace=True)
    load_avg.drop(columns=['groupBy'], inplace=True)


    return PF_data, load_avg
    pass