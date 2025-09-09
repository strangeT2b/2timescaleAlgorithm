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

def power_flow_org(power_flow_start, power_flow_horizon, DFs, DFs_noStop, PV_Price, Ys, params,  in_PV = False):
    ### simulation
    current_t = power_flow_start
    horizon = power_flow_horizon
    current_t += 0

    # DF and Y computing time
    time_DF_read = time.time() # DF and Y computing time

    # horizon list 
    dfs = DFs[current_t:current_t+horizon]
    # dfs_pv = PV_Price.loc[current_t:current_t+horizon-1, 'PV'].values # convert to unit MW
    dfs_noStop = DFs_noStop[current_t:current_t+horizon]
    ys = Ys[current_t:current_t+horizon]
    PV_Price = PV_Price.iloc[current_t:current_t+horizon,:]
    pv = PV_Price.drop(columns=["PRICE"])
    price = PV_Price["PRICE"]

    ####### GUROBI #######
    nums_noStop = np.zeros(shape=horizon,dtype='int64')
    nums_Tss = dfs[0]['class'].sum()

    # var [[[] for i in range(nums_noStop[t])] for t in range(horizon)]
    v = {}
    p_in = {}

    PF_vs_org = pd.DataFrame(columns=['Volt_N{}'.format(i) for i in range(1,nums_Tss+1)])
    PF_p_ins = pd.DataFrame(columns=['P_from_Source_N{}'.format(i) for i in range(1,nums_Tss+1)])

    P = {}
    for t in range(horizon):
        nums_noStop[t] = len(dfs_noStop[t].index.values)
        P[t] = dfs_noStop[t]['P'].values # convert to unit MW

    for t in range(horizon):
        clear_output(wait=True)
        print(f'开始进行第 {t} 个时刻的潮流求解')

        # model
        model = gp.Model()
        # var [[[] for i in range(nums_noStop[t])] for t in range(horizon)]
        v = {}
        p_in = {}
        p_g = {}
        p_brake = {}

        q_expr = gp.QuadExpr()
        l_expr = gp.LinExpr()

        # variables
        for i in dfs_noStop[t].index:
            if dfs_noStop[t].loc[i,'class']==1:
                v[t,i] = model.addVar(lb=0, ub=2, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                p_in[t,i] = model.addVar(lb=-10, ub=10, vtype=gp.GRB.CONTINUOUS, name='P_from_Source_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
            elif dfs_noStop[t].loc[i,'class']==0:
                v[t,i] = model.addVar(lb=0, ub=2, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                # p_brake[t,i] = model.addVar(lb=params.p_brake_min, ub=params.p_brake_max, vtype=gp.GRB.CONTINUOUS,name='P_brake_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                pass

        # constraints
        for i in dfs_noStop[t].index:

            if dfs_noStop[t].loc[i,'class'] == 1:
                # ### p_in
                l_expr = gp.LinExpr()
                l_expr.addTerms(params.Rs,p_in[t,i])
                l_expr.addTerms(params.E,v[t,i])
                model.addConstr(l_expr == params.E*params.E, name='P_IN_N{}_H{}'.format(i,t))
                l_expr.clear()
                
                ### power flow of station
                #### q_expr = YVV - V_i^2/params.Rs 
                q_expr.addTerms(-1/params.Rs,v[t,i],v[t,i])
                for j in range(nums_noStop[t]):
                    # print(t,i,j)
                    q_expr.addTerms(ys[t][i,j],v[t,i],v[t,j])

                p_sum = 0 # PV 算重复了
                if dfs_noStop[t].loc[i,'upStop'] > 0:
                    p_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'upStop'],'P'].values[0]
                if dfs_noStop[t].loc[i,'downStop'] > 0 and dfs_noStop[t].loc[i,'downStop'] != dfs_noStop[t].loc[i,'upStop']:
                    p_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'downStop'],'P'].values[0]
                p_sum += pv.iloc[t,i]
                l_expr.addTerms(params.E/params.Rs,v[t,i])
                # print(p_sum)
                l_expr.addConstant(p_sum)

                model.addQConstr(-q_expr == l_expr, name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                q_expr.clear()
                l_expr.clear()


            elif dfs_noStop[t].loc[i,'class'] == 0:
                # ### train braking power balance
                # train_name = int(dfs_noStop[t].loc[i,'name'])
                # model.addConstr(p_brake[t,i] == p_brakes.loc[t,train_name], name = "train brake input_N{}_{}".format(dfs_noStop[t].loc[i,'name'],t))
                # # print(f"p_brake == {p_brakes.loc[t,train_name]}")


                ### power flow of train
                for j in range(nums_noStop[t]):
                    q_expr.addTerms(ys[t][i,j],v[t,i],v[t,j])
                # model.addQConstr(q_expr  == P[t][i] + p_brake[t,i], name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                model.addQConstr(q_expr  == P[t][i], name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                q_expr.clear()
                l_expr.clear()



        # objective function
        model.setObjective(1, sense=gp.GRB.MINIMIZE)

        model.setParam('OutputFlag',0)
        model.setParam('NonConvex',2)
        model.setParam('PreSolve',1)
        model.setParam('NumericFocus',0)
        model.setParam('FeasibilityTol',1e-9)
        model.params.MIPGap = 0.001
        time_model = time.time()
        # model.write('/Users/town/Desktop/STU/Code/Jupyter_Code/Master/Main/2Stage/GrbLog/PF_log/PF_org.LP')
        # model.write('/Users/town/Desktop/STU/Code/Jupyter_Code/Master/Main/2Stage/GrbLog/PF_log/PF_org.MPS')

        model.optimize()
        time_slove = time.time()
        # model.write('/Users/town/Desktop/STU/Code/Jupyter_Code/Master/Main/2Stage/GrbLog/PF_log/PF_org.SOL')

        # 获取求解信息
        print('the objective value is : {}'.format(model.ObjVal))
        for x in model.getVars():
            if 'Volt' in x.VarName:
                match = re.match(r'Volt_N(\d+)_H(\d+)',x.VarName)
                PF_vs_org.loc[int(match.group(2)),'Volt_N'+match.group(1)] = x.X
                # PF_vs_org.append(x.X)
            elif 'P_from_Source' in x.VarName:
                match = re.match(r'P_from_Source_N(\d+)_H(\d+)',x.VarName)
                PF_p_ins.loc[int(match.group(2)),'P_from_Source_N'+match.group(1)] = x.X
                # PF_p_ins.append(x.X)
            else:
                print('遗漏了变量')

        # stopping train voltage recover
        for col in PF_vs_org.columns[11:]:
            match = re.match('Volt_N(\d+)',col)
            num_train = int(match.group(1))
            for idx in PF_vs_org[col].index:
                if pd.isna(PF_vs_org.loc[idx,col]):
                    match = re.match('H(\d+)',idx)
                    num_horizon = int(match.group(1))
                    current_DF = DFs[num_horizon]
                    if num_train in current_DF['name'].values:
                        stoppingTss = current_DF.loc[current_DF['name']==num_train,'stopping'].values[0]
                        if stoppingTss > 0 :
                            # print(stoppingTss)
                            PF_vs_org.loc[idx,col] = PF_vs_org.loc[idx,'Volt_N'+str(stoppingTss)]
                else: pass
        # time_list = [ time_df_read-time_DF_read
        #             , time_model-time_df_read
        #             , time_slove-time_model]
        # print('建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[1],time_list[2]))

    PF_data = {'vs':PF_vs_org.astype('float')
               ,'p_ins':PF_p_ins.astype('float')}
    return PF_data