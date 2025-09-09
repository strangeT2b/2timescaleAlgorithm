#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   offLine_OPF_org.py
@Time    :   2025/02/18 22:07:59
@Author  :   勾子瑭
'''

# here put the import lib

import gurobipy as gp
import re
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt


def offline_OPF_efficiency_org( schedule_horizon, DFs, DFs_noStop, PV_Price, Ys, params, in_PV = False, schedule_start=0, plot=False, path=""):
    current_t = schedule_start
    horizon = schedule_horizon
    current_t += 0
    print('这是第 {} 次求解\n'.format(current_t))
    # DF and Y computing time
    time_DF_read = time.time() # DF and Y computing time

    # horizon list 
    dfs = DFs[current_t:current_t+horizon]
    # dfs_pv = PV_Price.loc[current_t:current_t+horizon-1, 'PV'].values # convert to unit MW
    dfs_pv = PV_Price.loc[current_t:current_t+horizon-1,:]
    if "PRICE" in dfs_pv.columns:
        dfs_pv.drop(columns=["PRICE"], inplace = True)
    if "time" in dfs_pv.columns:
        dfs_pv.drop(columns=["time"], inplace = True)

    df_price = PV_Price.loc[current_t:current_t+horizon-1, 'PRICE'].values
    dfs_noStop = DFs_noStop[current_t:current_t+horizon]
    ys = Ys[current_t:current_t+horizon]
    time_df_read = time.time() # dfs time

    ####### GUROBI #######
    scaleNum = 10000
    scaleNum_pvBalance = 100
    nums_noStop = np.zeros(shape=horizon,dtype='int64')
    params.nums_Tss = dfs[0]['class'].sum()

    P = {}
    soc_0 = np.ones(params.nums_Tss)*50

    for t in range(horizon):
        dfs[t].reset_index(drop=True, inplace = True)
        dfs_noStop[t].reset_index(drop=True, inplace = True)
        nums_noStop[t] = len(dfs_noStop[t].name)
        P[t] = dfs_noStop[t]['P'].values # convert to unit MW
    
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
    p_brake = {}

    obj_expr = gp.LinExpr()
    obj_L1 = gp.LinExpr()
    q_expr = gp.QuadExpr()
    l_expr = gp.LinExpr()

    for t in range(horizon):
        # variables
        for i in dfs_noStop[t].index:
            if dfs_noStop[t].loc[i,'class']==1:
                soc[t,i] = model.addVar(lb=params.soc_min, ub=params.soc_max, vtype=gp.GRB.CONTINUOUS, name='SOC_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                p_b[t,i] = model.addVar(lb=params.p_b_min, ub=params.p_b_max, vtype=gp.GRB.CONTINUOUS, name='P_Battery_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                p_ch[t,i] = model.addVar(lb=params.p_ch_min, ub=params.p_ch_max, vtype=gp.GRB.CONTINUOUS, name='P_Battery_Charge_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                p_disch[t,i] = model.addVar(lb=params.p_disch_min, ub=params.p_disch_max, vtype=gp.GRB.CONTINUOUS, name='P_Battery_DisCharge_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                p_g[t,i] = model.addVar(lb=params.p_g_min, ub=params.p_g_max, vtype=gp.GRB.CONTINUOUS, name='P_to_Grid_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                p_in[t,i] = model.addVar(lb=params.p_in_min, ub=params.p_in_max, vtype=gp.GRB.CONTINUOUS, name='P_from_Source_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                v[t,i] = model.addVar(lb=params.v_tss_min, ub=params.v_tss_max, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                ## objective function expr
                obj_expr.addTerms(df_price[t]*params.delta_t,p_in[t,i])
                ## objective function expr
                obj_L1.addTerms(df_price[t]*params.delta_t,p_in[t,i])
                ## obj L1
                obj_L1.addTerms(params.delta_t, p_ch[t,i])
                obj_L1.addTerms(params.delta_t, p_disch[t,i])
            elif dfs_noStop[t].loc[i,'class']==0:
                v[t,i] = model.addVar(lb=params.v_train_min, ub=params.v_train_max, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                p_brake[t,i] = model.addVar(lb=params.p_brake_min, ub=params.p_brake_max, vtype=gp.GRB.CONTINUOUS,name='P_brake_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                pass

        # constraints
        for i in dfs_noStop[t].index:

            if dfs_noStop[t].loc[i,'class'] == 1:
                ### 光伏发电和电池充放电效率损失
                # p_b = p_ch/e + p_disch*e
                model.addConstr(p_b[t,i] == p_ch[t,i]/params.battery_efficiency + p_disch[t,i]*params.battery_efficiency, name='PB_charge_discharge_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                ### soc
                if t == 0:
                    model.addConstr(scaleNum*soc[t,i] == scaleNum*soc_0[i] + scaleNum*100*((p_ch[t,i]+p_disch[t,i])*params.delta_t)/(params.battery_max-params.battery_min), name='SOC_change_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                elif t >= 1:
                    model.addConstr(scaleNum*soc[t,i] == scaleNum*soc[t-1,i] + scaleNum*100*((p_ch[t,i]+p_disch[t,i])*params.delta_t)/(params.battery_max-params.battery_min), name='SOC_change_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                # soc_start = soc_end
                if t == horizon-1:
                    model.addConstr(soc[t,i] == soc_0[i], name = 'initial_soc=end_soc_N{}'.format(i))

                ### pv == p + g
                l_expr.addTerms(-1, p_b[t,i])
                l_expr.addTerms(-1, p_g[t,i])
                l_expr.addConstant(dfs_pv.iloc[t,i])
                model.addConstr(scaleNum_pvBalance*l_expr == 0, name='PV_Balance_N{}_H{}'.format(i,t))
                l_expr.clear()

                ### b + g >= 0
                model.addConstr(scaleNum*p_b[t,i] + scaleNum*p_g[t,i] >= scaleNum*min(0, dfs_pv.iloc[t,i]), name='BESS_Feasible_{}_H{}'.format(i,t))
                
                ### p_in
                model.addConstr(scaleNum*p_in[t,i]*params.Rs == scaleNum*params.E*(params.E - v[t,i]), name='P_IN_{}_H{}'.format(i,t))

                ### power flow of station
                #### q_expr = YVV - V_i^2/params.Rs 
                q_expr.addTerms(-1/params.Rs,v[t,i],v[t,i])
                for j in range(nums_noStop[t]):
                    q_expr.addTerms(ys[t][i,j],v[t,i],v[t,j])

                # p_sum = P[t][i] # pv
                # change 25.3.8 for PV(之前遗漏了PV)
                if in_PV:
                    p_sum = dfs_pv.iloc[t,i]
                else:
                    p_sum = 0
                p_sum = 0
                if dfs_noStop[t].loc[i,'upStop'] > 0:
                    p_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'upStop'],'P'].values[0]
                if dfs_noStop[t].loc[i,'downStop'] > 0 and dfs_noStop[t].loc[i,'downStop'] != dfs_noStop[t].loc[i,'upStop']:
                    p_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'downStop'],'P'].values[0]
                
                l_expr.addTerms(-params.E/params.Rs,v[t,i])
                l_expr.addTerms(-1,p_g[t,i]) # p_g - p_s 方向是不是反了?
                l_expr.addConstant(-p_sum)

                model.addQConstr(q_expr == l_expr, name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                q_expr.clear()
                l_expr.clear()
                p_sum = 0
# 注意符号！！
            elif dfs_noStop[t].loc[i,'class'] == 0:
                ### train braking power balance

                ### power flow of train
                for j in range(nums_noStop[t]):
                    q_expr.addTerms(ys[t][i,j],v[t,i],v[t,j])
                model.addQConstr(q_expr == P[t][i]+p_brake[t,i], name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                q_expr.clear()
                l_expr.clear()

    # objective function
    model.setObjective(params.realTime_weight_p_in * obj_expr + params.realTime_weight_L1 * obj_L1, sense=gp.GRB.MINIMIZE)
    time_model = time.time()

    for k,v in params.offlineNonlinearGpParams.items():
        model.setParam(k, v)
    model.setParam("TimeLimit", 10800)
    model.params.MIPGap = 0.01
    if path != "":
        model.setParam('LogFile', path)
    model.optimize()
    model.write("/home/gzt/Codes/2STAGE/Funcs/OPF/log/offline_org.LP")    

    time_slove = time.time()

    # # 获取求解信息
    # # print('the objective value is : {}'.format(model.ObjVal))
    vs = pd.DataFrame(columns=['Volt_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    SOCs = pd.DataFrame(columns=['SOC_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
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

    # stopping train voltage recoverß
    for col in vs.columns[11:]:
        match = re.match('Volt_N(\d+)',col)
        num_train = int(match.group(1))
        for idx in vs[col].index:
            if pd.isna(vs.loc[idx,col]):
                # match = re.match('H(\d+)',idx)
                # num_horizon = int(match.group(1))
                num_horizon = idx
                current_DF = DFs[num_horizon]
                if num_train in current_DF['name'].values:
                    stoppingTss = current_DF.loc[current_DF['name']==num_train,'stopping'].values[0]
                    if stoppingTss > 0 :
                        # print(stoppingTss)
                        vs.loc[idx,col] = vs.loc[idx,'Volt_N'+str(stoppingTss)]
            else: pass
    
    ### plot
    if plot == True:
        SOCs.plot(figsize=(20,6))
        plt.show()

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
            ,'p_chs':p_chs
            ,'p_dischs':p_dischs
            ,'p_bras': p_bras}


