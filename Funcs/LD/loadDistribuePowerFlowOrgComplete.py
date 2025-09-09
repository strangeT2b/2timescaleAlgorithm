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


def loadDistributePowerFlowOrgComplete(LD_start, LD_horizon, DFs, DFs_noStop, PV_Price, Ys, Ycs, Yrs, params, sec_interval,  in_PV = False, stepByStep = False):
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
        # ys = Ys[t]
        # ycs = Ycs
        # yrs = Yrs
    for t in range(horizon):
        dfs[t].reset_index(drop=True, inplace=True)
        dfs_noStop[t].reset_index(drop=True, inplace=True)

        nums_noStop[t] = len(dfs_noStop[t].index.values)
        P[t] = dfs_noStop[t]['P'].values # convert to unit MW

    # var [[[] for i in range(nums_noStop[t])] for t in range(horizon)]
    v = {}
    v_c = {}
    v_r = {}
    v_diff = {}
    p_in = {}
    aux = {}
    p_brake = {}


    PF_v_cs = pd.DataFrame(columns=['Volt_c_N{}'.format(i) for i in range(1,nums_Tss+1)])
    PF_v_rs = pd.DataFrame(columns=['Volt_r_N{}'.format(i) for i in range(1,nums_Tss+1)])
    PF_v_diffs = pd.DataFrame(columns=['Volt_diff_N{}'.format(i) for i in range(1,nums_Tss+1)])
    PF_p_ins = pd.DataFrame(columns=['P_from_Source_N{}'.format(i) for i in range(1,nums_Tss+1)])
    PF_p_bras = pd.DataFrame(columns=['P_brake_N{}'.format(i) for i in range(1,nums_Tss+1)])

    for t in range(horizon):
        clear_output(wait=True)
        print('这是第 {} 时刻求解\n'.format(t))
        time_df_read = time.time() # dfs time
        # model
        model = gp.Model()
        obj_expr = gp.LinExpr()
        q_expr = gp.QuadExpr()
        l_expr = gp.LinExpr()

        ys = Ys[t]
        ycs = Ycs[t]
        yrs = Yrs[t]

        # variables
        for i in dfs_noStop[t].index:
            if dfs_noStop[t].loc[i,'class']==1:
                v_c[t,i] = model.addVar(lb=params.v_tss_min, ub=params.v_tss_max, name='Volt_c_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                v_r[t,i] = model.addVar(lb=0, ub=params.v_tss_max, name='Volt_r_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                v_diff[t,i] = model.addVar(lb=params.v_tss_min, ub=params.v_tss_max, name='Volt_diff_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))

                p_in[t,i] = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='P_from_Source_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

            elif dfs_noStop[t].loc[i,'class']==0:
                v_c[t,i] = model.addVar(lb=params.v_train_min, ub=params.v_train_max, name='Volt_c_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                v_r[t,i] = model.addVar(lb=-0.1, ub=params.v_train_max, name='Volt_r_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                v_diff[t,i] = model.addVar(lb=params.v_train_min, ub=params.v_train_max, name='Volt_diff_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                p_brake[t,i] = model.addVar(lb=params.p_brake_min, ub=params.p_brake_max, vtype=gp.GRB.CONTINUOUS,name='P_brake_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                obj_expr.addTerms(0, p_brake[t,i])

        # constraints
        for i in dfs_noStop[t].index:
            # print(f"下面为 i = {i} 的信息:\n")
            # print("params.Rs, params.E:", params.Rs, params.E)
            # print("v bounds:", params.v_tss_min, params.v_tss_max, params.v_train_min, params.v_train_max)
            # print("P for node:", P[t][i])  # 这就是 PF_N12_H2 里的 RHS = 5 的来源
            # print("ycs row for i:", ycs[i,:])

            # v_diff 变量
            model.addConstr(v_diff[t,i] == v_c[t,i]-v_r[t,i], name='V_Diff_eq_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

            if dfs_noStop[t].loc[i,'class'] == 1:
                # 牵引站电位关系
                model.addConstr(v_r[t,i]==0, name="voltage_Potential_tss_N{}_H{}".format(dfs_noStop[t].loc[i,'name'],t))

                ### p_in
                l_expr = gp.LinExpr()
                l_expr.addTerms(params.Rs,p_in[t,i])
                l_expr.addTerms(params.E,v_diff[t,i])
                model.addConstr(1e3*l_expr == 1e3*params.E*params.E, name='P_IN_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                l_expr.clear()

                ### power flow of station 
                #### q_expr = YVV - V_i^2/Rs 
                q_expr.addTerms(-1/params.Rs,v_diff[t,i], v_diff[t,i])
                for j in range(nums_noStop[t]):
                    l_expr.addTerms(ycs[i,j],v_c[t,j])
                aux[t,i] = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="aux_N{}_H{}".format(dfs_noStop[t].loc[i,'name'],t))
                model.addConstr(aux[t,i]==l_expr, name="YV_c_N{}_H{}".format(dfs_noStop[t].loc[i,'name'],t))
                q_expr.addTerms(1, aux[t,i], v_diff[t,i])
                l_expr.clear()

                p_sum = 0
                # stopping train load
                if dfs_noStop[t].loc[i,'upStop'] > 0:
                    p_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'upStop'],'P'].values[0]
                if dfs_noStop[t].loc[i,'downStop'] > 0 and dfs_noStop[t].loc[i,'downStop'] != dfs_noStop[t].loc[i,'upStop']:
                    p_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'downStop'],'P'].values[0]
                
                l_expr.addTerms(-params.E/params.Rs,v_diff[t,i])
                # print(p_sum)
                l_expr.addConstant(p_sum)

                model.addQConstr(q_expr == l_expr, name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                q_expr.clear()
                l_expr.clear()

            elif dfs_noStop[t].loc[i,'class'] == 0:
                # # 电位关系
                left_expr = gp.LinExpr()
                right_expr = gp.LinExpr()
                for j in range(nums_noStop[t]):
                    left_expr.addTerms(ycs[i,j], v_c[t,j])
                    right_expr.addTerms(yrs[i,j], v_r[t,j])
                model.addConstr(left_expr == -right_expr, name="voltage_Potential_train_N{}_H{}".format(dfs_noStop[t].loc[i,'name'],t))

                ### power flow of train
                aux[t,i] = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="aux_N{}_H{}".format(dfs_noStop[t].loc[i,'name'],t))
                model.addConstr(aux[t,i]==left_expr, name="YV_c_N{}_H{}".format(dfs_noStop[t].loc[i,'name'],t))
                q_expr.addTerms(1, aux[t,i], v_diff[t,i])
                model.addQConstr(q_expr == P[t][i]+p_brake[t,i], name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                
                q_expr.clear()
                l_expr.clear()
                left_expr.clear()
                right_expr.clear()

        
        # objective function
        model.setObjective(obj_expr, sense=gp.GRB.MINIMIZE)
        model.setParam('ScaleFlag', 2)  # 启用自动缩放
        model.setParam('OutputFlag',0)
        model.setParam('NonConvex',2)
        # model.setParam('PreSolve',1)
        # model.setParam('NumericFocus',3)
        # model.setParam('FeasibilityTol',1e-3)
        model.params.MIPGap = 0.01
        time_model = time.time()
        # model.write('/home/gzt/Codes/2STAGE/Funcs/LD/PF_org.LP')
        # model.write('/home/gzt/Codes/2STAGE/Funcs/LD/PF_org.MPS')
        
        for v in model.getVars():
            print(v.VarName, "LB=", v.LB, "UB=", v.UB)

        model.optimize()
        time_slove = time.time()
        # model.write('/Users/town/Desktop/STU/Code/Jupyter_Code/Master/Main/2Stage/GrbLog/PF_log/PF_org.SOL')
        print(f"第 {t} 时刻 {model.Status}")

        if model.status == gp.GRB.INFEASIBLE:
            model.computeIIS()
            model.write("/home/gzt/Codes/2STAGE/Funcs/LD/load_distribution_infeasible_model.ilp")  # 输出冲突约束
            print(f"第 {t} 时刻 infeasible")
            break


        # 获取求解信息
        # print('the objective value is : {}'.format(model.ObjVal))
        for x in model.getVars():
            if 'Volt_c' in x.VarName:
                match = re.match(r'Volt_c_N(\d+)_H(\d+)',x.VarName)
                PF_v_cs.loc[int(match.group(2)),'Volt_c_N'+match.group(1)] = x.X
                # PF_v_cs.append(x.X)
            elif 'Volt_r' in x.VarName:
                match = re.match(r'Volt_r_N(\d+)_H(\d+)',x.VarName)
                PF_v_rs.loc[int(match.group(2)),'Volt_r_N'+match.group(1)] = x.X
                # PF_v_rs.append(x.X)
            elif 'Volt_diff' in x.VarName:
                match = re.match(r'Volt_diff_N(\d+)_H(\d+)',x.VarName)
                PF_v_diffs.loc[int(match.group(2)),'Volt_diff_N'+match.group(1)] = x.X
                # PF_v_diffs.append(x.X)
            elif 'P_from_Source' in x.VarName:
                match = re.match(r'P_from_Source_N(\d+)_H(\d+)',x.VarName)
                PF_p_ins.loc[int(match.group(2)),'P_from_Source_N'+match.group(1)] = x.X
                # PF_p_ins.append(x.X)
            elif 'P_brake' in x.VarName:
                match = re.match(r'P_brake_N(\d+)_H(\d+)',x.VarName)
                PF_p_bras.loc[int(match.group(2)),'P_brake_N'+match.group(1)] = x.X
                # PF_p_bras.append(x.X)
            elif "aux" in x.VarName:
                pass
            # else:
            #     print('遗漏了变量')

        model.dispose()

        # # stopping train voltage recover
        # for col in PF_v_cs.columns[11:]:
        #     match = re.match('Volt_c_N(\d+)',col)
        #     num_train = int(match.group(1))
        #     for idx in PF_v_cs[col].index:
        #         if pd.isna(PF_v_cs.loc[idx,col]):
        #             match = re.match('H(\d+)',idx)
        #             num_horizon = int(match.group(1))
        #             current_DF = DFs[num_horizon]
        #             if num_train in current_DF['name'].values:
        #                 stoppingTss = current_DF.loc[current_DF['name']==num_train,'stopping'].values[0]
        #                 if stoppingTss > 0 :
        #                     # print(stoppingTss)
        #                     PF_v_cs.loc[idx,col] = PF_v_cs.loc[idx,'Volt_c_N'+str(stoppingTss)]
        #         else: pass
        # for col in PF_v_rs.columns[11:]:
        #     match = re.match('Volt_r_N(\d+)',col)
        #     num_train = int(match.group(1))
        #     for idx in PF_v_rs[col].index:
        #         if pd.isna(PF_v_rs.loc[idx,col]):
        #             match = re.match('H(\d+)',idx)
        #             num_horizon = int(match.group(1))
        #             current_DF = DFs[num_horizon]
        #             if num_train in current_DF['name'].values:
        #                 stoppingTss = current_DF.loc[current_DF['name']==num_train,'stopping'].values[0]
        #                 if stoppingTss > 0 :
        #                     # print(stoppingTss)
        #                     PF_v_rs.loc[idx,col] = PF_v_rs.loc[idx,'Volt_r_N'+str(stoppingTss)]
        #         else: pass
        # for col in PF_v_diffs.columns[11:]:
        #     match = re.match('Volt_diff_N(\d+)',col)
        #     num_train = int(match.group(1))
        #     for idx in PF_v_diffs[col].index:
        #         if pd.isna(PF_v_diffs.loc[idx,col]):
        #             match = re.match('H(\d+)',idx)
        #             num_horizon = int(match.group(1))
        #             current_DF = DFs[num_horizon]
        #             if num_train in current_DF['name'].values:
        #                 stoppingTss = current_DF.loc[current_DF['name']==num_train,'stopping'].values[0]
        #                 if stoppingTss > 0 :
        #                     # print(stoppingTss)
        #                     PF_v_diffs.loc[idx,col] = PF_v_diffs.loc[idx,'Volt_diff_N'+str(stoppingTss)]
        #         else: pass

    # PF_data = {'vs':PF_vs_org.astype('float')
    #            ,'p_ins':PF_p_ins.astype('float')}
    
    PF_data = {'v_cs':PF_v_cs.astype('float'),
               'v_rs':PF_v_rs.astype('float'),
               'v_diffs':PF_v_diffs.astype('float'),
               'p_ins':PF_p_ins.astype('float'),
                "p_brakes":PF_p_bras.astype('float')}

    print('---------- 潮流计算完成，开始分配负载 ----------\n')
    load_avg = PF_data['p_ins'].reset_index(drop=True)
    load_avg.columns = [f'N{i}' for i in range(1, params.nums_Tss+1)]
    load_avg.reset_index(drop=True, inplace=True)
    load_avg['groupBy'] = load_avg.index.values.astype('int')//sec_interval
    load_avg.groupby('groupBy').mean().reset_index(drop=True, inplace=True)
    load_avg.drop(columns=['groupBy'], inplace=True)


    return PF_data, load_avg
    pass