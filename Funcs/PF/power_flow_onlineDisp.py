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

def power_flow_onlineDisp( power_flow_horizon, DFs, DFs_noStop, PV_Price, Ys, online_data, params, in_PV=False, power_flow_start=0, plot=False):
    E = params.E
    Rs = params.Rs
    nums_Tss = params.nums_Tss

    ### simulation
    current_t = power_flow_start
    horizon = power_flow_horizon
    current_t += 0
    print('开始进行潮流方程代回求解\n')
    # DF and Y computing time
    time_DF_read = time.time() # DF and Y computing time

    # horizon list 
    dfs = DFs[current_t:current_t+horizon]
    dfs_noStop = DFs_noStop[current_t:current_t+horizon]
    # dfs_pv = PV_Price.loc[current_t:current_t+horizon-1, 'PV'].values # convert to unit MW
    dfs_pv = PV_Price.loc[current_t:current_t+horizon-1,:]
    if "PRICE" in dfs_pv.columns:
        dfs_pv.drop(columns=["PRICE"], inplace = True)
    if "time" in dfs_pv.columns:
        dfs_pv.drop(columns=["time"], inplace = True)
    dfs_pv.columns = dfs_pv.columns.astype("int")

    ys = Ys[current_t:current_t+horizon]
    time_df_read = time.time() # dfs time
    pbs = online_data['p_bs'].copy()
    # print(pbs)
    if type(pbs.columns[0]) == 'str' and '_N' in pbs.columns[0]:
        pbs.columns = pbs.columns.map(lambda x : int(x.split('N')[1]))
    pbs.columns = pbs.columns.astype(int)
    if type(pbs.index[0]) == 'str' and 'H' in pbs.index[0]:
        pbs.index = pbs.index.map(lambda x : int(x.split('H')[1]))
    pbs.index = pbs.index.astype(int)

    pgs = online_data['p_gs'].copy()
    if type(pgs.columns[0]) == 'str' and '_N' in pgs.columns[0]:
        pgs.columns = pgs.columns.map(lambda x : int(x.split('N')[1]))
    pgs.columns = pgs.columns.astype(int)
    if type(pgs.index[0]) == 'str' and 'H' in pgs.index[0]:
        pgs.index = pgs.index.map(lambda x : int(x.split('H')[1]))
    pgs.index = pgs.index.astype(int)

    p_brakes = online_data['p_bras'].copy()
    p_brakes.fillna(value = 0, inplace = True)
    if type(p_brakes.columns[0]) == 'str' and '_N' in p_brakes.columns[0]:
        p_brakes.columns = p_brakes.columns.map(lambda x : int(x.split('N')[1]))
    if type(p_brakes.index[0]) == 'str' and 'H' in p_brakes.index[0]:
        p_brakes.index = p_brakes.index.map(lambda x : int(x.split('H')[1]))
    # vs = online_data['vs'].copy()
    # if type(vs.columns[0]) == 'str' and '_N' in vs.columns[0]:
    #     vs.columns = vs.columns.map(lambda x : int(x.split('N')[1]))
    # if type(vs.index[0]) == 'str' and 'H' in vs.index[0]:
    #     vs.index = vs.index.map(lambda x : int(x.split('H')[1]))  
    PF_vs_org = pd.DataFrame(columns=['Volt_N{}'.format(i) for i in range(1,nums_Tss+1)])
    PF_p_ins = pd.DataFrame(columns=['P_from_Source_N{}'.format(i) for i in range(1,nums_Tss+1)])
    PF_p_gs = pd.DataFrame(columns=['P_to_Grid_N{}'.format(i) for i in range(1,nums_Tss+1)])
    ####### GUROBI #######
    nums_noStop = np.zeros(shape=horizon,dtype='int64')
    nums_Tss = dfs[0]['class'].sum()

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
                v[t,i] = model.addVar(lb=params.v_tss_min, ub=params.v_tss_max, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                p_in[t,i] = model.addVar(lb=params.p_in_min, ub=params.p_in_max, vtype=gp.GRB.CONTINUOUS, name='P_from_Source_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                p_g[t,i] = model.addVar(lb=params.p_g_min, ub=params.p_g_max, vtype=gp.GRB.CONTINUOUS, name='P_to_Grid_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

            elif dfs_noStop[t].loc[i,'class']==0:
                v[t,i] = model.addVar(lb=params.v_train_min, ub=params.v_train_max, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                p_brake[t,i] = model.addVar(lb=params.p_brake_min, ub=params.p_brake_max, vtype=gp.GRB.CONTINUOUS,name='P_brake_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                pass

        # constraints
        for i in dfs_noStop[t].index:

            if dfs_noStop[t].loc[i,'class'] == 1:
                # ### p_in
                l_expr = gp.LinExpr()
                l_expr.addTerms(Rs,p_in[t,i])
                l_expr.addTerms(E,v[t,i])
                model.addConstr(l_expr == E*E, name='P_IN_N{}_H{}'.format(i,t))
                l_expr.clear()


                # Pgs
                tss_name = int(dfs_noStop[t].loc[i,'name'])
                # print(t)
                # print(tss_name)
                # print(tss_name)
                model.addConstr(p_g[t,i] == pgs.loc[t,tss_name], name="tss pgs input_N{}_H{}".format(dfs_noStop[t].loc[i,'name'],t))
                # print(f"p_g == {pgs.loc[t,tss_name]}")
                
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
                
                l_expr.addTerms(params.E/params.Rs,v[t,i])
                l_expr.addTerms(1,p_g[t,i])
                # print(p_sum)
                l_expr.addConstant(p_sum)

                model.addQConstr(-q_expr == l_expr, name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                q_expr.clear()
                l_expr.clear()


            elif dfs_noStop[t].loc[i,'class'] == 0:
                ### train braking power balance
                train_name = int(dfs_noStop[t].loc[i,'name'])
                model.addConstr(p_brake[t,i] == p_brakes.loc[t,train_name], name = "train brake input_N{}_{}".format(dfs_noStop[t].loc[i,'name'],t))
                # print(f"p_brake == {p_brakes.loc[t,train_name]}")


                ### power flow of train
                for j in range(nums_noStop[t]):
                    q_expr.addTerms(ys[t][i,j],v[t,i],v[t,j])
                model.addQConstr(q_expr  == P[t][i] + p_brake[t,i], name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                q_expr.clear()
                l_expr.clear()

        # objective function
        model.setObjective(1, sense=gp.GRB.MINIMIZE)

        time_model = time.time()
        model.setParam('OutputFlag',0)
        model.setParam('NonConvex',2)
        model.setParam('FeasibilityTol', 1e-6)  # 收紧可行性容忍度
        # model.setParam('NumericFocus', 3)       # 增强数值稳定性
        # model.setParam('BarConvTol', 1e-10)     # 对偶收敛容忍度
        model.params.MIPGap = params.grb_gap

        model.optimize()
        if model.status == gp.GRB.INFEASIBLE:
            model.computeIIS()
            model.write("/Users/town/Desktop/STU/Code/Jupyter_Code/Master/Main/2Stage/Funcs/PF/log/infeasible_PF.ilp")
            raise Exception("Model is infeasible")

        time_slove = time.time()

        # 获取求解信息
        # print('the objective value is : {}'.format(model.ObjVal))

        for x in model.getVars():
            if 'Volt' in x.VarName:
                match = re.match(r'Volt_N(\d+)_H(\d+)',x.VarName)
                PF_vs_org.loc['H'+match.group(2),'Volt_N'+match.group(1)] = x.X
                # PF_vs_org.append(x.X)
            elif 'P_from_Source' in x.VarName:
                match = re.match(r'P_from_Source_N(\d+)_H(\d+)',x.VarName)
                PF_p_ins.loc['H'+match.group(2),'P_from_Source_N'+match.group(1)] = x.X
                # PF_p_ins.append(x.X)
            elif 'P_to_Grid' in x.VarName:
                match = re.match(r'P_to_Grid_N(\d+)_H(\d+)',x.VarName)
                PF_p_gs.loc['H'+match.group(2),'P_to_Grid_N'+match.group(1)] = x.X
                # PF_p_gs.append(x.X)
            else:
                print('遗漏了变量')

# stopping train voltage recover
    for col in PF_vs_org.columns[11:]:
        match = re.match('Volt_N(\d+)',col)
        num_train = int(match.group(1))
        for idx in PF_vs_org[col].index:
            if pd.isna(PF_vs_org.loc[idx,col]):
                if type(idx)==str:
                    match = re.match('H(\d+)',idx)
                    num_horizon = int(match.group(1))
                else:
                    num_horizon = int(idx)
                current_DF = DFs[num_horizon]
                if num_train in current_DF['name'].values:
                    stoppingTss = current_DF.loc[current_DF['name']==num_train,'stopping'].values[0]
                    if stoppingTss > 0 :
                        # print(stoppingTss)
                        PF_vs_org.loc[idx,col] = PF_vs_org.loc[idx,'Volt_N'+str(stoppingTss)]
            else: pass



    time_list = [ time_df_read-time_DF_read
                , time_model-time_df_read
                , time_slove-time_model ]
    # print('读取 DFs 耗时: {} s\n计算 dfs 和 Y 耗时: {} s\n建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[0],time_list[1],time_list[2],time_list[3]))
    print('建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[1],time_list[2]))
    if "p_bs" in online_data:
        PF_data = {'vs':PF_vs_org.astype('float')
                ,'p_gs': online_data['p_gs']
                ,'p_bs': online_data['p_bs']
                ,'SOCs': online_data['SOCs']
                ,'p_ins':PF_p_ins.astype('float')}
    else:
        PF_data = {'vs':PF_vs_org.astype('float')
                   ,'p_gs': online_data['p_gs']
                   ,'SOCs': online_data['SOCs']
                   ,'p_ins':PF_p_ins.astype('float')}
    
    if 'p_chs' in online_data.keys():
        PF_data['p_chs'] = online_data['p_chs']
    if 'p_dischs' in online_data.keys():
        PF_data['p_dischs'] = online_data['p_dischs']
    print(f'共求解 {power_flow_horizon} 时刻的潮流')
    return PF_data