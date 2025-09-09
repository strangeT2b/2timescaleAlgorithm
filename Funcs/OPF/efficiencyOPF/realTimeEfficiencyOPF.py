#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   realTime_OPF_org.py
@Time    :   2025/02/18 22:07:39
@Author  :   勾子瑭
'''

# here put the import lib
import gurobipy as gp
import re
import pandas as pd
import time
import numpy as np

def realTime_OPF_efficiency_cplx(schedule_start, schedule_horizon, realTime_horizon, DFs, DFs_noStop, PV_Price, Ys, SOCs, p_ins, params, in_PV = False, plot=False):
    current_t = schedule_start

    # 改为使用list存储结果
    real_vs_list = []
    real_socs_list = []
    real_p_ins_list = []
    real_p_bs_list = []
    real_p_chs_list = []
    real_p_dischs_list = []
    real_p_gs_list = []
    real_p_brakes_list = []
    real_j_ins_list = []

    # 初始SOC
    soc_0 = SOCs.loc[0,:]
    # 需要track的SOC
    SOCs = SOCs.loc[1:,:]
    SOCs.reset_index(drop=True, inplace = True)
    df_time = pd.DataFrame(columns=['linearModel_time', 'linearSolve_time', 'linearSum_time', 'nonlinearModel_time', 'nonlinearSolve_time'])
    real_vs = pd.DataFrame(columns=['Volt_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_socs = pd.DataFrame(columns=['SOC_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_ins = pd.DataFrame(columns=['P_from_Source_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_bs = pd.DataFrame(columns=['P_Battery_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_chs = pd.DataFrame(columns=['P_Battery_Charge_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_dischs = pd.DataFrame(columns=['P_Battery_DisCharge_N{}'.format(i) for i in range(1,params.nums_Tss+1)])

    real_p_gs = pd.DataFrame(columns=['P_to_Grid_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_brakes = pd.DataFrame(columns=['P_brake_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_j_ins = pd.DataFrame(columns=['J_from_Source_N{}'.format(i) for i in range(1,params.nums_Tss+1)])

    gaps = pd.DataFrame(columns=["linear", "nonlinear"], index=range(schedule_start, schedule_horizon-realTime_horizon))

    out_realTime_horizon = realTime_horizon 
    for current_t in range(schedule_horizon):
        print("schedule_horizon=",schedule_horizon)
        current_t += schedule_start
        
        # 动态调整 realTime horizon
        realTime_horizon = max(min(out_realTime_horizon, schedule_horizon-current_t), 1)

        print('这是第 {} 次求解\n'.format(current_t))
        # DF and Y computing time
        time_DF_read = time.time() # DF and Y computing time

        # realTime_horizon list 
        dfs = DFs[current_t:current_t+realTime_horizon]
        # dfs_pv = PV_Price.loc[current_t:current_t+realTime_horizon-1, 'PV'].values # convert to unit MW
        dfs_pv = PV_Price.loc[current_t:current_t+realTime_horizon-1,:]
        if "PRICE" in dfs_pv.columns:
            dfs_pv.drop(columns=["PRICE"], inplace = True)
        if "time" in dfs_pv.columns:
            dfs_pv.drop(columns=["time"], inplace = True)
        df_price = PV_Price.loc[current_t:current_t+realTime_horizon-1, 'PRICE'].values
        dfs_noStop = DFs_noStop[current_t:current_t+realTime_horizon]
        ys = Ys[current_t:current_t+realTime_horizon]
        time_df_read = time.time() # dfs time

        soc_tracked = SOCs.iloc[current_t:current_t+realTime_horizon+1, :]
        p_in_tracked = p_ins.to_numpy()[current_t:current_t+realTime_horizon+1, :]

        ####### GUROBI #######
        scaleNum = 1000
        nums_noStop = np.zeros(shape=realTime_horizon,dtype='int64')
        params.nums_Tss = dfs[0]['class'].sum()

        P = {}
        if current_t==schedule_start:
            soc_0 = np.ones(params.nums_Tss)*50
        else:
            # soc_0 = [value for key,value in socs_entry.items() if ("SOC" in key)]
            # soc_0 = real_socs.iloc[current_t-1,:].values
            pass

        for t in range(realTime_horizon):
            dfs[t].reset_index(drop=True, inplace = True)
            dfs_noStop[t].reset_index(drop=True, inplace = True)
            nums_noStop[t] = len(dfs_noStop[t].name)
            P[t] = dfs_noStop[t]['P'].values # convert to unit MW

    # 首先进行 linear opf

        # model
        model = gp.Model()
        # var [[[] for i in range(nums_noStop[t])] for t in range(realTime_horizon)]
        v = {}
        soc = {}
        p_b = {}
        p_ch = {}
        p_disch = {}
        p_g = {}
        p_in = {}
        p_brake = {}
        SOC_tracked_error = {}
        p_in_tracked_error = {}
        j_in = {}

        obj_expr = gp.LinExpr()
        obj_L1_expr = gp.LinExpr()
        YV_expr = gp.LinExpr()
        l_expr = gp.LinExpr()
        obj_bra_expr = gp.LinExpr()
        SOC_track_norm = model.addVar(lb=0,ub=400, vtype=gp.GRB.CONTINUOUS, name = '2_norm_soc')
        p_ins_track_norm = model.addVar(lb=0,ub=400, vtype=gp.GRB.CONTINUOUS, name = '2_norm_pin')

        for t in range(realTime_horizon):
            # variables
            for i in dfs_noStop[t].index:
                if dfs_noStop[t].loc[i,'class']==1:
                    soc[t,i] = model.addVar(lb=params.soc_min, ub=params.soc_max, vtype=gp.GRB.CONTINUOUS, name='SOC_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_b[t,i] = model.addVar(lb=params.p_b_min, ub=params.p_b_max, vtype=gp.GRB.CONTINUOUS, name='P_Battery_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_ch[t,i] = model.addVar(lb=params.p_ch_min, ub=params.p_ch_max, vtype=gp.GRB.CONTINUOUS, name='P_Battery_Charge_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_disch[t,i] = model.addVar(lb=params.p_disch_min, ub=params.p_disch_max, vtype=gp.GRB.CONTINUOUS, name='P_Battery_DisCharge_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                    p_g[t,i] = model.addVar(lb=params.p_g_min, ub=params.p_g_max, vtype=gp.GRB.CONTINUOUS, name='P_to_Grid_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_in[t,i] = model.addVar(lb=params.p_in_min, ub=params.p_in_max, vtype=gp.GRB.CONTINUOUS, name='P_from_Source_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    j_in[t,i] = model.addVar(lb=params.j_in_min, ub=params.j_in_max, vtype=gp.GRB.CONTINUOUS, name='J_from_Source_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    v[t,i] = model.addVar(lb=params.v_tss_min, ub=params.v_tss_max, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                    ## objective function expr
                    obj_expr.addTerms(df_price[t]*params.delta_t,p_in[t,i])
                    ## objective L1 expr
                    obj_L1_expr.addTerms(params.delta_t,p_ch[t,i])
                    obj_L1_expr.addTerms(-params.delta_t,p_disch[t,i])
                    ## track_error
                    SOC_tracked_error[t,i] = model.addVar(lb=-200, ub=200, vtype=gp.GRB.CONTINUOUS,name='Track_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_in_tracked_error[t,i] = model.addVar(lb=-200, ub=200, vtype=gp.GRB.CONTINUOUS,name='Track_p_in_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                
                elif dfs_noStop[t].loc[i,'class']==0:
                    v[t,i] = model.addVar(lb=params.v_train_min, ub=params.v_train_max, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                    p_brake[t,i] = model.addVar(lb=params.p_brake_min, ub=params.p_brake_max, vtype=gp.GRB.CONTINUOUS,name='P_brake_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                    ## train braking obj
                    obj_bra_expr.addTerms(1, p_brake[t,i])
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

                    ### track error
                    model.addConstr(SOC_tracked_error[t,i] == soc_tracked.iloc[t,i] - soc[t,i],name='track_error_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    # model.addConstr(p_in_tracked_error[t,i] == p_in_tracked[t,i] - p_in[t,i],name='track_error_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                    ### pv >= p + g
                    l_expr.addTerms(-1, p_b[t,i])
                    l_expr.addTerms(-1, p_g[t,i])
                    l_expr.addConstant(dfs_pv.iloc[t,i])
                    model.addConstr(scaleNum*l_expr == 0, name='PV_Balance_N{}_H{}'.format(i,t))
                    l_expr.clear()

                    ### b + g >= 0
                    model.addConstr(scaleNum*p_b[t,i] + scaleNum*p_g[t,i] >= scaleNum*min(0, dfs_pv.iloc[t,i]), name='BESS_Feasible_{}_H{}'.format(i,t))

                    ### p_in
                    l_expr.addTerms(params.Rs,p_in[t,i])
                    l_expr.addTerms(params.E,v[t,i])
                    model.addConstr(scaleNum*l_expr == scaleNum*params.E*params.E, name='P_IN_N{}_H{}'.format(i,t))
                    l_expr.clear()

                    ### j_in = p_in / rated_volt
                    model.addConstr(scaleNum*j_in[t,i] == scaleNum*p_in[t,i]/params.E)

                    ### power flow of station
                    for j in range(nums_noStop[t]):
                        # print(t,i,j)
                        YV_expr.addTerms(ys[t][i,j],v[t,j])

                    #### stopping train load
                    p_sum = 0 # PV 算重复了
                    if dfs_noStop[t].loc[i,'upStop'] > 0:
                        p_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'upStop'],'P'].values[0]
                    if dfs_noStop[t].loc[i,'downStop'] > 0 and dfs_noStop[t].loc[i,'downStop'] != dfs_noStop[t].loc[i,'upStop']:
                        p_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'downStop'],'P'].values[0]
                    j_sum = p_sum/params.E# change 25.3.8 for PV(之前遗漏了PV)
                    #### volt source current
                    l_expr.addTerms(1, j_in[t,i])
                    #### for TSS is 0
                    l_expr.addConstant(j_sum)
                    #### PV_BESS current
                    l_expr.addTerms(1/params.E, p_g[t,i])

                    model.addConstr(-YV_expr == l_expr, name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    YV_expr.clear()
                    l_expr.clear()

                elif dfs_noStop[t].loc[i,'class'] == 0:
                    ### train braking power balance

                    ### power flow of train
                    for j in range(nums_noStop[t]):
                        YV_expr.addTerms(ys[t][i,j], v[t,j])
                    l_expr.addTerms(-1,p_brake[t,i])
                    model.addConstr(YV_expr == (P[t][i]+p_brake[t,i])/params.E_train, name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    YV_expr.clear()
                    l_expr.clear()


        # objective function
        model.addGenConstrNorm(SOC_track_norm, SOC_tracked_error.values(), 2.0, name = 'SOC_track_norm_constr')
        # model.addGenConstrNorm(p_ins_track_norm, p_in_tracked_error.values(), 2.0, name = 'p_in_track_norm_constr')

        model.setObjective(params.realTime_weight_track*(SOC_track_norm) + params.realTime_weight_brake*obj_bra_expr + params.realTime_weight_p_in*obj_expr + params.realTime_weight_L1*obj_L1_expr, sense=gp.GRB.MINIMIZE)

        model.write("/home/gzt/Codes/2STAGE/Funcs/OPF/efficiencyOPF/log/realTime_cplx_linear.LP")
        time_linearModel = time.time()

        for k, v in params.onlineLinearGpParams.items():
            model.setParam(k, v)
        model
        model.optimize()
        gaps.loc[current_t, "linear"] = model.MIPGap

        if model.status == gp.GRB.INFEASIBLE:
            model.computeIIS()
            model.write("/home/gzt/Codes/2STAGE/Funcs/OPF/efficiencyOPF/log/infeasible_cplxLin.ilp")
            raise Exception("Model is infeasible")

        # model.write('/home/gzt/Codes/2STAGE/Funcs/OPF/efficiencyOPF/log/solveInfo_cplxLin.SOL')
        time_linearSlove = time.time()

        # 只记录了部分解
        vs_entry = {'index': f'H{current_t}'}
        socs_entry = {'index': f'H{current_t}'}
        p_gs_entry = {'index': f'H{current_t}'}
        p_ins_entry = {'index': f'H{current_t}'}
        j_ins_entry = {'index': f'H{current_t}'}
        p_bs_entry = {'index': f'H{current_t}'}
        p_chs_entry = {'index': f'H{current_t}'}
        p_dischs_entry = {'index': f'H{current_t}'}
        p_brakes_entry = {'index': f'H{current_t}'}

        # # 获取求解信息
        linear_solution = {} # 只记录了部分解
        for x in model.getVars():
            if ('Volt' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'Volt_N(\d+)_H(\d+)',x.VarName)
                # real_vs.loc['H'+str(current_t),'Volt_N'+match.group(1)] = x.X
                vs_entry[f'Volt_N{match.group(1)}'] = x.X

                # 记录线性解，非线性热启动
                linear_solution[x.VarName] = x.X

            elif ('SOC' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'SOC_N(\d+)_H(\d+)',x.VarName)
                # real_socs.loc['H'+str(current_t),'SOC_N'+match.group(1)] = x.X
                socs_entry[f'SOC_N{match.group(1)}'] = x.X

            elif ('P_to_Grid' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_to_Grid_N(\d+)_H(\d+)',x.VarName)
                # real_p_gs.loc['H'+str(current_t),'P_to_Grid_N'+match.group(1)] = x.X
                p_gs_entry[f'P_to_Grid_N{match.group(1)}'] = x.X

                # 记录线性解，非线性热启动
                linear_solution[x.VarName] = x.X

            elif ('P_from_Source' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_from_Source_N(\d+)_H(\d+)',x.VarName)
                # real_p_ins.loc['H'+str(current_t),'P_from_Source_N'+match.group(1)] = x.X
                p_ins_entry[f'P_from_Source_N{match.group(1)}'] = x.X

            elif ('J_from_Source' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'J_from_Source_N(\d+)_H(\d+)',x.VarName)
                # real_j_ins.loc['H'+str(current_t),'J_from_Source_N'+match.group(1)] = x.X
                j_ins_entry[f'J_from_Source_N{match.group(1)}'] = x.X

            elif ('P_Battery_N' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_Battery_N(\d+)_H(\d+)',x.VarName)
                # real_p_bs.loc['H'+str(current_t),'P_Battery_N'+match.group(1)] = x.X
                p_bs_entry[f'P_Battery_N{match.group(1)}'] = x.X

                # 记录线性解，非线性热启动
                linear_solution[x.VarName] = x.X

            elif 'P_Battery_Charge' in x.VarName and ('H0' in x.VarName):
                match = re.match(r'P_Battery_Charge_N(\d+)_H(\d+)',x.VarName)
                # real_p_chs.loc['H'+str(current_t),'P_Battery_Charge_N'+match.group(1)] = x.X
                p_chs_entry[f'P_Battery_Charge_N{match.group(1)}'] = x.X

                # 记录线性解，非线性热启动
                linear_solution[x.VarName] = x.X

            elif 'P_Battery_DisCharge' in x.VarName and ('H0' in x.VarName):
                match = re.match(r'P_Battery_DisCharge_N(\d+)_H(\d+)',x.VarName)
                # real_p_dischs.loc['H'+str(current_t),'P_Battery_DisCharge_N'+match.group(1)] = x.X
                p_dischs_entry[f'P_Battery_DisCharge_N{match.group(1)}'] = x.X

                # 记录线性解，非线性热启动
                linear_solution[x.VarName] = x.X

            elif ('P_brake' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_brake_N(\d+)_H(\d+)',x.VarName)
                # real_p_brakes.loc['H'+str(current_t),'P_brake_N'+match.group(1)] = x.X
                p_brakes_entry[f'P_brake_N{match.group(1)}'] = x.X

                # 记录线性解，非线性热启动
                linear_solution[x.VarName] = x.X
            else:
                pass


        time_list = [ time_df_read-time_DF_read
                    , time_linearModel-time_df_read
                    , time_linearSlove-time_linearModel]
        # print('读取 DFs 耗时: {} s\n计算 dfs 和 Y 耗时: {} s\n建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[0],time_list[1],time_list[2],time_list[3]))
        print('建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[1],time_list[2]))
        # 写入 linear 求解用时
        df_time.loc[current_t,['linearModel_time', 'linearSolve_time']] = [time_list[1], time_list[2]]
        df_time.loc[current_t,['linearSum_time']] = df_time.loc[current_t,'linearModel_time'] + df_time.loc[current_t,'linearSolve_time']
        model.dispose()

        # 如果时间充裕，则进行 nonlinear OPF
        time_linearEnd = time.time()
        if time_linearEnd - time_DF_read > 0.8:
            # 添加到list中
            real_vs_list.append(vs_entry)
            real_socs_list.append(socs_entry)
            real_p_gs_list.append(p_gs_entry)
            real_p_ins_list.append(p_ins_entry)
            real_p_bs_list.append(p_bs_entry)
            real_p_chs_list.append(p_chs_entry)
            real_p_dischs_list.append(p_dischs_entry)
            real_p_brakes_list.append(p_brakes_entry)
            # 维护上一个时刻的 SOC
            soc_0 = [socs_entry[f"SOC_N{i}"] for i in range(1,params.nums_Tss+1)]
            continue
        else:
            print(f'进入 nonlinear OPF, 剩余时间{1 - time_linearEnd + time_DF_read} s\n')

        nonlinearTime = 1 - time_linearEnd + time_DF_read
        time_nonlinearStart = time.time()

        # model
        scaleNum = 1000
        model = gp.Model()

        # var [[[] for i in range(nums_noStop[t])] for t in range(realTime_horizon)]
        v = {}
        soc = {}
        p_b = {}
        p_ch = {}
        p_disch = {}
        p_g = {}
        p_in = {}
        p_brake = {}
        SOC_tracked_error = {}

        obj_expr = gp.LinExpr()
        obj_L1_expr = gp.LinExpr()
        q_expr = gp.QuadExpr()
        l_expr = gp.LinExpr()
        tracked_expr = gp.QuadExpr()
        obj_bra_expr = gp.LinExpr()

        SOC_track_norm = model.addVar(lb=0,ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name = '2_norm')    
        for t in range(realTime_horizon):
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
                    obj_expr.addTerms(df_price[t]*params.delta_t, p_in[t,i])
                    ## objective L1 expr
                    obj_L1_expr.addTerms(params.delta_t,p_ch[t,i])
                    obj_L1_expr.addTerms(-params.delta_t,p_disch[t,i])
                    ## track_error
                    SOC_tracked_error[t,i] = model.addVar(lb=-20, ub=20, vtype=gp.GRB.CONTINUOUS,name='Track_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_in_tracked_error[t,i] = model.addVar(lb=-200, ub=200, vtype=gp.GRB.CONTINUOUS,name='Track_p_in_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                
                elif dfs_noStop[t].loc[i,'class']==0:
                    v[t,i] = model.addVar(lb=params.v_train_min, ub=params.v_train_max, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                    p_brake[t,i] = model.addVar(lb=params.p_brake_min, ub=params.p_brake_max, vtype=gp.GRB.CONTINUOUS,name='P_brake_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    obj_bra_expr.addTerms(1,p_brake[t,i])
                    pass

            # constraints
            for i in dfs_noStop[t].index:

                if dfs_noStop[t].loc[i,'class'] == 1:
                    ### 光伏发电和电池充放电效率损失
                    # p_b = p_ch/e + p_disch*e
                    model.addConstr(p_b[t,i] == p_ch[t,i]/params.battery_efficiency + p_disch[t,i]*params.battery_efficiency, name='PB_charge_discharge_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    ### soc
                    if t == 0:
                        model.addConstr(scaleNum*soc[t,i] == scaleNum*soc_0[i] + scaleNum*100*((p_ch[t,i]+p_disch[t,i])*params.delta_t)/(params.battery_max-params.battery_min))
                    elif t >= 1:
                        model.addConstr(scaleNum*soc[t,i] == scaleNum*soc[t-1,i] + scaleNum*100*((p_ch[t,i]+p_disch[t,i])*params.delta_t)/(params.battery_max-params.battery_min))

                    ### pv >= p + g
                    l_expr.addTerms(-1, p_b[t,i])
                    l_expr.addTerms(-1, p_g[t,i])
                    l_expr.addConstant(dfs_pv.iloc[t,i])
                    model.addConstr(scaleNum*l_expr == 0, name='PV_Balance_N{}_H{}'.format(i,t))
                    l_expr.clear()

                    ### b + g >= 0
                    model.addConstr(scaleNum*p_b[t,i] + scaleNum*p_g[t,i] >= scaleNum*min(0, dfs_pv.iloc[t,i]), name='BESS_Feasible_{}_H{}'.format(i,t))

                    ### p_in
                    l_expr.addTerms(params.Rs,p_in[t,i])
                    l_expr.addTerms(params.E,v[t,i])
                    model.addConstr(scaleNum*l_expr == scaleNum*params.E*params.E, name='P_IN_N{}_H{}'.format(i,t))
                    l_expr.clear()

                    ### power flow of station
                    #### q_expr = YVV - V_i^2/params.Rs 
                    q_expr.addTerms(-1/params.Rs,v[t,i],v[t,i])
                    for j in range(nums_noStop[t]):
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

                    # tracked error
                    model.addConstr(SOC_tracked_error[t,i]==soc[t,i]-soc_tracked.iloc[t,i],name='track_error_N{}H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                elif dfs_noStop[t].loc[i,'class'] == 0:
                    ### train braking power balance

                    ### power flow of train
                    for j in range(nums_noStop[t]):
                        q_expr.addTerms(ys[t][i,j],v[t,i],v[t,j])
                    l_expr.addTerms(-1,p_brake[t,i])
                    model.addQConstr((q_expr+l_expr) == P[t][i], name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    q_expr.clear()
                    l_expr.clear()
        model.addGenConstrNorm(SOC_track_norm, SOC_tracked_error.values(), 2.0, name='2_norm')
        # objective function
        model.setObjective(params.realTime_weight_track*SOC_track_norm + params.realTime_weight_brake*obj_bra_expr + params.realTime_weight_p_in*obj_expr + params.realTime_weight_L1*obj_L1_expr, sense=gp.GRB.MINIMIZE)

        time_nonlinearModel = time.time()

        for k, v in params.onlineNonlinearGpParams.items():
            model.setParam(k, v)

        time_sloveLeft = nonlinearTime - (time.time() - time_linearEnd)

        if time_sloveLeft < 0.2:
            # 添加到list中
            real_vs_list.append(vs_entry)
            real_socs_list.append(socs_entry)
            real_p_gs_list.append(p_gs_entry)
            real_p_ins_list.append(p_ins_entry)
            real_p_bs_list.append(p_bs_entry)
            real_p_chs_list.append(p_chs_entry)
            real_p_dischs_list.append(p_dischs_entry)
            real_p_brakes_list.append(p_brakes_entry)
            # 维护上一个时刻的 SOC
            soc_0 = [socs_entry[f"SOC_N{i}"] for i in range(1,params.nums_Tss+1)]
            continue

        model.setParam('TimeLimit', time_sloveLeft)
        model.setParam("MIPGap", 0.1)
        model.write("/home/gzt/Codes/2STAGE/Funcs/OPF/efficiencyOPF/log/realTime_cplx_nonlinear.LP")
        
        #--- 热启动
        # 第二阶段初始化时加载起始点
        for x in model.getVars():
            if x.VarName in linear_solution:
                x.Start = linear_solution[x.VarName]  # 关键热启动语句

        model.optimize()
        time_nonlinearSlove = time.time()
        gaps.loc[current_t, "nonlinear"] = model.MIPGap

        #---
        # 如果超时，则保留 linear 的结果， 直接退出
        if model.Status == gp.GRB.TIME_LIMIT or model.Status == gp.GRB.INFEASIBLE:
            # 添加到list中
            real_vs_list.append(vs_entry)
            real_socs_list.append(socs_entry)
            real_p_gs_list.append(p_gs_entry)
            real_p_ins_list.append(p_ins_entry)
            real_p_bs_list.append(p_bs_entry)
            real_p_chs_list.append(p_chs_entry)
            real_p_dischs_list.append(p_dischs_entry)
            real_p_brakes_list.append(p_brakes_entry)
            # 维护上一个时刻的 SOC
            soc_0 = [socs_entry[f"SOC_N{i}"] for i in range(1,params.nums_Tss+1)]
            continue
        


        # # 获取求解信息
        for x in model.getVars():
            if ('Volt' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'Volt_N(\d+)_H(\d+)',x.VarName)
                # real_vs.loc['H'+str(current_t),'Volt_N'+match.group(1)] = x.X
                vs_entry[f'Volt_N{match.group(1)}'] = x.X


            elif ('SOC' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'SOC_N(\d+)_H(\d+)',x.VarName)
                # real_socs.loc['H'+str(current_t),'SOC_N'+match.group(1)] = x.X
                socs_entry[f'SOC_N{match.group(1)}'] = x.X

            elif ('P_to_Grid' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_to_Grid_N(\d+)_H(\d+)',x.VarName)
                # real_p_gs.loc['H'+str(current_t),'P_to_Grid_N'+match.group(1)] = x.X
                p_gs_entry[f'P_to_Grid_N{match.group(1)}'] = x.X

            elif ('P_from_Source' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_from_Source_N(\d+)_H(\d+)',x.VarName)
                # real_p_ins.loc['H'+str(current_t),'P_from_Source_N'+match.group(1)] = x.X
                p_ins_entry[f'P_from_Source_N{match.group(1)}'] = x.X

            elif ('J_from_Source' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'J_from_Source_N(\d+)_H(\d+)',x.VarName)
                # real_j_ins.loc['H'+str(current_t),'J_from_Source_N'+match.group(1)] = x.X
                j_ins_entry[f'J_from_Source_N{match.group(1)}'] = x.X

            elif ('P_Battery_N' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_Battery_N(\d+)_H(\d+)',x.VarName)
                # real_p_bs.loc['H'+str(current_t),'P_Battery_N'+match.group(1)] = x.X
                p_bs_entry[f'P_Battery_N{match.group(1)}'] = x.X

            elif 'P_Battery_Charge' in x.VarName and ('H0' in x.VarName):
                match = re.match(r'P_Battery_Charge_N(\d+)_H(\d+)',x.VarName)
                # real_p_chs.loc['H'+str(current_t),'P_Battery_Charge_N'+match.group(1)] = x.X
                p_chs_entry[f'P_Battery_Charge_N{match.group(1)}'] = x.X

            elif 'P_Battery_DisCharge' in x.VarName and ('H0' in x.VarName):
                match = re.match(r'P_Battery_DisCharge_N(\d+)_H(\d+)',x.VarName)
                # real_p_dischs.loc['H'+str(current_t),'P_Battery_DisCharge_N'+match.group(1)] = x.X
                p_dischs_entry[f'P_Battery_DisCharge_N{match.group(1)}'] = x.X

            elif ('P_brake' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_brake_N(\d+)_H(\d+)',x.VarName)
                # real_p_brakes.loc['H'+str(current_t),'P_brake_N'+match.group(1)] = x.X
                p_brakes_entry[f'P_brake_N{match.group(1)}'] = x.X

            else:
                pass

        # 添加到list中
        real_vs_list.append(vs_entry)
        real_socs_list.append(socs_entry)
        real_p_gs_list.append(p_gs_entry)
        real_p_ins_list.append(p_ins_entry)
        real_p_bs_list.append(p_bs_entry)
        real_p_chs_list.append(p_chs_entry)
        real_p_dischs_list.append(p_dischs_entry)
        real_p_brakes_list.append(p_brakes_entry)
        # 维护上一个时刻的 SOC
        soc_0 = [socs_entry[f"SOC_N{i}"] for i in range(1,params.nums_Tss+1)]
        
        time_list = [time_nonlinearModel-time_nonlinearStart
                    , time_nonlinearSlove-time_linearModel]
        # print('读取 DFs 耗时: {} s\n计算 dfs 和 Y 耗时: {} s\n建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[0],time_list[1],time_list[2],time_list[3]))
        print('建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[0],time_list[1]))
        df_time.loc[current_t,['nonlinearModel_time', 'nonlinearSolve_time']] = [time_list[0], time_list[1]]
        model.dispose()

    # 转换list为DataFrame
    def list_to_df(data_list):
        if not data_list:
            return pd.DataFrame()
        df = pd.DataFrame(data_list)
        df.set_index('index', inplace=True)
        return df

    real_vs = list_to_df(real_vs_list)
    real_socs = list_to_df(real_socs_list)
    real_p_ins = list_to_df(real_p_ins_list)
    real_p_gs = list_to_df(real_p_gs_list)
    real_p_bs = list_to_df(real_p_bs_list)
    real_p_chs = list_to_df(real_p_chs_list)
    real_p_dischs = list_to_df(real_p_dischs_list)
    real_p_brakes = list_to_df(real_p_brakes_list)
    real_j_ins = list_to_df(real_j_ins_list)
    

    # stopping train voltage recover
    for col in real_vs.columns[11:]:
        match = re.match('Volt_N(\d+)',col)
        num_train = int(match.group(1))
        for idx in real_vs[col].index:
            if pd.isna(real_vs.loc[idx,col]):
                match = re.match('H(\d+)',idx)
                num_horizon = int(match.group(1))
                current_DF = DFs[num_horizon]
                if num_train in current_DF['name'].values:
                    stoppingTss = current_DF.loc[current_DF['name']==num_train,'stopping'].values[0]
                    if stoppingTss > 0 :
                        # print(stoppingTss)
                        real_vs.loc[idx,col] = real_vs.loc[idx,'Volt_N'+str(stoppingTss)]
            else: pass
    # real_socs.reset_index(drop=True, inplace=True)
    # real_socs.index = list(i+1 for i in real_socs.index.values)
    # real_socs.loc[0,:] = soc_0
    return {'vs':real_vs
            ,'p_ins': real_p_ins
            ,'SOCs': real_socs
            ,'p_gs': real_p_gs
            ,'p_bs': real_p_bs
            ,'p_chs':real_p_chs
            ,'p_dischs':real_p_dischs
            ,'p_bras': real_p_brakes
            ,'df_time':df_time}


def realTime_OPF_efficiency_org(schedule_start, schedule_horizon, realTime_horizon, DFs, DFs_noStop, PV_Price, Ys, SOCs, p_ins, params, in_PV = False, plot=False):
    current_t = schedule_start
    # soc_0 = SOCs.loc[0,:]
    # SOCs = SOCs.loc[1:,:]
    # SOCs.reset_index(replace = True, drop = True)
    # SOCs.reset_index(drop=True, inplace = True)
    df_time = pd.DataFrame(columns=['linearModel_time', 'linearSolve_time', 'linearSum_time', 'nonlinearModel_time', 'nonlinearSolve_time'])
    real_vs = pd.DataFrame(columns=['Volt_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_socs = pd.DataFrame(columns=['SOC_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    # real_socs.loc[0,:] = SOCs.loc[0,:]
    real_p_ins = pd.DataFrame(columns=['P_from_Source_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_bs = pd.DataFrame(columns=['P_Battery_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_chs = pd.DataFrame(columns=['P_Battery_Charge_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_dischs = pd.DataFrame(columns=['P_Battery_DisCharge_N{}'.format(i) for i in range(1,params.nums_Tss+1)])

    real_p_gs = pd.DataFrame(columns=['P_to_Grid_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_brakes = pd.DataFrame(columns=['P_brake_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_j_ins = pd.DataFrame(columns=['J_from_Source_N{}'.format(i) for i in range(1,params.nums_Tss+1)])   

    for current_t in range(schedule_horizon-realTime_horizon):
        current_t += schedule_start
        print('这是第 {} 次求解\n'.format(current_t))
        # DF and Y computing time
        time_DF_read = time.time() # DF and Y computing time

        # realTime_horizon list 
        dfs = DFs[current_t:current_t+realTime_horizon]
        # dfs_pv = PV_Price.loc[current_t:current_t+realTime_horizon-1, 'PV'].values # convert to unit MW
        dfs_pv = PV_Price.loc[current_t:current_t+realTime_horizon-1,:]
        if "PRICE" in dfs_pv.columns:
            dfs_pv.drop(columns=["PRICE"], inplace = True)
        if "time" in dfs_pv.columns:
            dfs_pv.drop(columns=["time"], inplace = True)

        df_price = PV_Price.loc[current_t:current_t+realTime_horizon-1, 'PRICE'].values
        dfs_noStop = DFs_noStop[current_t:current_t+realTime_horizon]
        ys = Ys[current_t:current_t+realTime_horizon]
        time_df_read = time.time() # dfs time

        soc_tracked = SOCs.to_numpy()[current_t:current_t+realTime_horizon+1,:]
        p_in_tracked = p_ins.to_numpy()[current_t:current_t+realTime_horizon+1,:]

        ####### GUROBI #######
        nums_noStop = np.zeros(shape=realTime_horizon,dtype='int64')
        params.nums_Tss = dfs[0]['class'].sum()

        P = {}
        if current_t==schedule_start or current_t==0:
            soc_0 = np.ones(params.nums_Tss)*50
        else:
            soc_0 = real_socs.iloc[current_t-1,:].values

        for t in range(realTime_horizon):
            dfs[t].reset_index(drop=True, inplace = True)
            dfs_noStop[t].reset_index(drop=True, inplace = True)
            nums_noStop[t] = len(dfs_noStop[t].name)
            P[t] = dfs_noStop[t]['P'].values # convert to unit MW


        # model
        model = gp.Model()

        # var [[[] for i in range(nums_noStop[t])] for t in range(realTime_horizon)]
        v = {}
        soc = {}
        p_b = {}
        p_ch = {}
        p_disch = {}
        p_g = {}
        p_in = {}
        p_brake = {}
        SOC_tracked_error = {}

        obj_expr = gp.LinExpr()
        obj_L1_expr = gp.LinExpr()
        q_expr = gp.QuadExpr()
        l_expr = gp.LinExpr()
        tracked_expr = gp.QuadExpr()
        obj_bra_expr = gp.LinExpr()

        SOC_track_norm = model.addVar(lb=0,ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name = '2_norm')    
        for t in range(realTime_horizon):
            # variables
            for i in dfs_noStop[t].index:
                if dfs_noStop[t].loc[i,'class']==1:
                    soc[t+1,i] = model.addVar(lb=params.soc_min, ub=params.soc_max, vtype=gp.GRB.CONTINUOUS, name='SOC_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_b[t,i] = model.addVar(lb=params.p_b_min, ub=params.p_b_max, vtype=gp.GRB.CONTINUOUS, name='P_Battery_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_ch[t,i] = model.addVar(lb=0, ub=params.p_b_max, vtype=gp.GRB.CONTINUOUS, name='P_Battery_Charge_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_disch[t,i] = model.addVar(lb=params.p_b_min, ub=0, vtype=gp.GRB.CONTINUOUS, name='P_Battery_DisCharge_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                    p_g[t,i] = model.addVar(lb=params.p_g_min, ub=params.p_g_max, vtype=gp.GRB.CONTINUOUS, name='P_to_Grid_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_in[t,i] = model.addVar(lb=params.p_in_min, ub=params.p_in_max, vtype=gp.GRB.CONTINUOUS, name='P_from_Source_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    v[t,i] = model.addVar(lb=params.v_tss_min, ub=params.v_tss_max, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                    ## objective function expr
                    obj_expr.addTerms(df_price[t]*params.delta_t, p_in[t,i])
                    ## objective L1 expr
                    obj_L1_expr.addTerms(params.delta_t,p_ch[t,i])
                    obj_L1_expr.addTerms(-params.delta_t,p_disch[t,i])
                    ## track_error
                    SOC_tracked_error[t+1,i] = model.addVar(lb=-1000, ub=1000, vtype=gp.GRB.CONTINUOUS,name='Track_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                                
                elif dfs_noStop[t].loc[i,'class']==0:
                    v[t,i] = model.addVar(lb=params.v_train_min, ub=params.v_train_max, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                    p_brake[t,i] = model.addVar(lb=params.p_brake_min, ub=params.p_brake_max, vtype=gp.GRB.CONTINUOUS,name='P_brake_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    obj_bra_expr.addTerms(1,p_brake[t,i])
                    pass

            # constraints
            for i in dfs_noStop[t].index:

                if dfs_noStop[t].loc[i,'class'] == 1:
                    ### 光伏发电和电池充放电效率损失
                    # p_b = p_ch/e + p_disch*e
                    model.addConstr(p_b[t,i] == p_ch[t,i]/params.battery_efficiency + p_disch[t,i]*params.battery_efficiency, name='PB_charge_discharge_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    ### soc
                    if t == 0:
                        model.addConstr(soc[t,i] == SOCs.loc[current_t,i] + 100*((p_ch[t,i]+p_disch[t,i])*params.delta_t)/(params.battery_max-params.battery_min))
                    elif t >= 1:
                        model.addConstr(soc[t+1,i] == soc[t,i] + 100*((p_ch[t,i]+p_disch[t,i])*params.delta_t)/(params.battery_max-params.battery_min))

                    # if t == 0:
                    #     model.addConstr(soc[t,i] == soc_0[i] + ((p_ch[t,i]+p_disch[t,i])*params.delta_t)/(params.battery_max-params.battery_min))
                    # elif t >= 1:
                    #     model.addConstr(soc[t,i] == soc[t-1,i] + ((p_ch[t,i]+p_disch[t,i])*params.delta_t)/(params.battery_max-params.battery_min))

                    ### pv >= p + g
                    l_expr.addTerms(-1, p_b[t,i])
                    l_expr.addTerms(-1, p_g[t,i])
                    l_expr.addConstant(dfs_pv.iloc[t,i])
                    model.addConstr(l_expr == 0, name='PV_Balance_N{}_H{}'.format(i,t))
                    l_expr.clear()

                    ### b + g >= 0
                    model.addConstr(p_b[t,i] + p_g[t,i] >= min(0, dfs_pv.iloc[t,i]), name='BESS_Feasible_{}_H{}'.format(i,t))

                    ### p_in
                    l_expr.addTerms(params.Rs,p_in[t,i])
                    l_expr.addTerms(params.E,v[t,i])
                    model.addConstr(l_expr == params.E*params.E, name='P_IN_N{}_H{}'.format(i,t))
                    l_expr.clear()

                    ### power flow of station
                    #### q_expr = YVV - V_i^2/params.Rs 
                    q_expr.addTerms(-1/params.Rs,v[t,i],v[t,i])
                    for j in range(nums_noStop[t]):
                        q_expr.addTerms(ys[t][i,j],v[t,i],v[t,j])

                    # p_sum = P[t][i] # pv
                    # change 25.3.8 for PV(之前遗漏了PV)
                    # if in_PV:
                    #     p_sum = dfs_pv.iloc[t,i]
                    # else:
                    #     p_sum = 0
                    p_sum = 0 # PV 算重复了
                    if dfs_noStop[t].loc[i,'upStop'] > 0:
                        p_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'upStop'],'P'].values[0]
                    if dfs_noStop[t].loc[i,'downStop'] > 0 and dfs_noStop[t].loc[i,'downStop'] != dfs_noStop[t].loc[i,'upStop']:
                        p_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'downStop'],'P'].values[0]
                    
                    l_expr.addTerms(params.E/params.Rs,v[t,i])
                    l_expr.addTerms(1,p_g[t,i])
                    # print(p_sum)
                    l_expr.addConstant(p_sum)

                    model.addQConstr(-1e-2 * q_expr == 1e-2 * l_expr, name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    q_expr.clear()
                    l_expr.clear()

                    # tracked error
                    model.addConstr(SOC_tracked_error[t+1,i]==soc[t+1,i]-soc_tracked[t+1,i],name='track_error_N{}H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                elif dfs_noStop[t].loc[i,'class'] == 0:
                    ### train braking power balance

                    ### power flow of train
                    for j in range(nums_noStop[t]):
                        q_expr.addTerms(ys[t][i,j],v[t,i],v[t,j])
                    l_expr.addTerms(-1,p_brake[t,i])
                    model.addQConstr(1e-2 * (q_expr+l_expr) == 1e-2 * P[t][i], name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    q_expr.clear()
                    l_expr.clear()
        model.addGenConstrNorm(SOC_track_norm, SOC_tracked_error.values(), 2.0, name='2_norm')
        # objective function
        model.setObjective(params.realTime_weight_track*SOC_track_norm + params.realTime_weight_brake*obj_bra_expr + params.realTime_weight_p_in*obj_expr + params.realTime_weight_L1*obj_L1_expr, sense=gp.GRB.MINIMIZE)

        time_nonlinearModel = time.time()

        model.setParam('OutputFlag',0)
        model.setParam('NonConvex',2)
        model.setParam('PreSolve',2)
        model.setParam('NumericFocus',0)
        model.setParam('FeasibilityTol',1e-6)
        model.setParam('BarHomogeneous', 1)     # 处理数值不稳定
        model.setParam('ScaleFlag', 2)           # 启用自动缩放
        model.setParam('TimeLimit', 20)
        model.params.MIPGap = 0.05
        time_nonlinearModel = time.time()
        # model.resetParams()
        model.optimize()
        time_nonlinearSlove = time.time()
        # model.write('/Users/town/Desktop/STU/Code/Jupyter_Code/Master/Main/2Stage/Funcs/OPF/log/solveInfo_cplxOrg.LP')
        # model.write('/Users/town/Desktop/STU/Code/Jupyter_Code/Master/Main/2Stage/Funcs/OPF/log/solveInfo_cplxOrg.MPS')
        # model.write('/Users/town/Desktop/STU/Code/Jupyter_Code/Master/Main/2Stage/Funcs/OPF/log/solveInfo_cplxOrg.SOL')

        # 如果超时，则保留 linear 的结果， 直接退出
        if model.Status == gp.GRB.TIME_LIMIT:
            continue
        # print_vs = list()
        # # 获取求解信息
        for x in model.getVars():
            if ('Volt' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'Volt_N(\d+)_H(\d+)',x.VarName)
                real_vs.loc['H'+str(current_t),'Volt_N'+match.group(1)] = x.X
                # print_vs.append(x.X)
            elif ('SOC' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'SOC_N(\d+)_H(\d+)',x.VarName)
                real_socs.loc['H'+str(current_t),'SOC_N'+match.group(1)] = x.X
                # socs.append(x.X)  
            elif ('P_to_Grid' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_to_Grid_N(\d+)_H(\d+)',x.VarName)
                real_p_gs.loc['H'+str(current_t),'P_to_Grid_N'+match.group(1)] = x.X
                # p_gs.append(x.X)
            elif ('P_from_Source' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_from_Source_N(\d+)_H(\d+)',x.VarName)
                real_p_ins.loc['H'+str(current_t),'P_from_Source_N'+match.group(1)] = x.X
                # p_ins.append(x.X)
            elif ('P_Battery_N' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_Battery_N(\d+)_H(\d+)',x.VarName)
                real_p_bs.loc['H'+str(current_t),'P_Battery_N'+match.group(1)] = x.X
                # p_bs.append(x.X)
            elif ('P_Battery_Charge' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_Battery_Charge_N(\d+)_H(\d+)',x.VarName)
                real_p_chs.loc['H'+str(current_t),'P_Battery_Charge_N'+match.group(1)] = x.X
            elif 'P_Battery_DisCharge' in x.VarName and ('H0' in x.VarName):
                match = re.match(r'P_Battery_DisCharge_N(\d+)_H(\d+)',x.VarName)
                real_p_dischs.loc['H'+str(current_t),'P_Battery_DisCharge_N'+match.group(1)] = x.X
            elif ('P_brake' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_brake_N(\d+)_H(\d+)',x.VarName)
                real_p_brakes.loc['H'+str(current_t),'P_brake_N'+match.group(1)] = x.X
                # p_bras.append(x.X)
            else:
                pass

        time_list = [time_nonlinearModel-time_DF_read
                    , time_nonlinearSlove-time_nonlinearModel]
        # print('读取 DFs 耗时: {} s\n计算 dfs 和 Y 耗时: {} s\n建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[0],time_list[1],time_list[2],time_list[3]))
        print('建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[0],time_list[1]))
        df_time.loc[current_t,['nonlinearModel_time', 'nonlinearSolve_time']] = [time_list[0], time_list[1]]
        model.dispose()
    # stopping train voltage recover
        for col in real_vs.columns[11:]:
            match = re.match('Volt_N(\d+)',col)
            num_train = int(match.group(1))
            for idx in real_vs[col].index:
                if pd.isna(real_vs.loc[idx,col]):
                    match = re.match('H(\d+)',idx)
                    num_horizon = int(match.group(1))
                    current_DF = DFs[num_horizon]
                    if num_train in current_DF['name'].values:
                        stoppingTss = current_DF.loc[current_DF['name']==num_train,'stopping'].values[0]
                        if stoppingTss > 0 :
                            # print(stoppingTss)
                            real_vs.loc[idx,col] = real_vs.loc[idx,'Volt_N'+str(stoppingTss)]
                else: pass
    # real_socs.reset_index(drop=True, inplace=True)
    # real_socs.index = list(i+1 for i in real_socs.index.values)
    # real_socs.loc[0,:] = soc_0
    return {'vs':real_vs
            ,'p_ins': real_p_ins
            ,'SOCs': real_socs
            ,'p_gs': real_p_gs
            ,'p_bs': real_p_bs
            ,'p_chs':real_p_chs
            ,'p_dischs':real_p_dischs
            ,'p_bras': real_p_brakes
            ,'df_time':df_time}



def realTime_OPF_efficiency_lin(schedule_start, schedule_horizon, realTime_horizon, DFs, DFs_noStop, PV_Price, Ys, SOCs, p_ins, params, in_PV = False, plot=False):

    current_t = schedule_start
    soc_0 = SOCs.loc[0,:]
    SOCs = SOCs.loc[1:,:]
    df_time = pd.DataFrame(columns=['model_time', 'solve_time', 'sum_time'])
    real_vs = pd.DataFrame(columns=['Volt_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_ins = pd.DataFrame(columns=['P_from_Source_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_j_ins = pd.DataFrame(columns=['J_from_Source_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_socs = pd.DataFrame(columns=['SOC_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_bs = pd.DataFrame(columns=['P_Battery_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_chs = pd.DataFrame(columns=['P_Battery_Charge_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_dischs = pd.DataFrame(columns=['P_Battery_DisCharge_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_gs = pd.DataFrame(columns=['P_to_Grid_N{}'.format(i) for i in range(1,params.nums_Tss+1)])
    real_p_brakes = pd.DataFrame(columns=['P_brake_N{}'.format(i) for i in range(1,params.nums_Tss+1)])

    for current_t in range(schedule_horizon-realTime_horizon):
        schedule_start = 0
        current_t += schedule_start
        print('这是第 {} 次求解\n'.format(current_t))
        # DF and Y computing time
        time_DF_read = time.time() # DF and Y computing time

        # realTime_horizon list 
        dfs = DFs[current_t:current_t+realTime_horizon]
        df_pv = PV_Price.loc[current_t:current_t+realTime_horizon-1,'PV'].values
        # dfs_pv = PV_Price.loc[current_t:current_t+realTime_horizon-1, 'PV'].values # convert to unit MW
        dfs_pv = PV_Price.loc[current_t:current_t+realTime_horizon-1,:]
        if "PRICE" in dfs_pv.columns:
            dfs_pv.drop(columns=["PRICE"], inplace = True)
        if "time" in dfs_pv.columns:
            dfs_pv.drop(columns=["time"], inplace = True)

        df_price = PV_Price.loc[current_t:current_t+realTime_horizon-1, 'PRICE'].values
        dfs_noStop = DFs_noStop[current_t:current_t+realTime_horizon]
        ys = Ys[current_t:current_t+realTime_horizon]
        time_df_read = time.time() # dfs time

        # SOCs = pd.read_csv('/Users/town/Desktop/STU/Code/Jupyter_Code/Master/Main/TwoStage/soc_10h.csv')
        soc_tracked = SOCs.to_numpy()[current_t:current_t+realTime_horizon+1,:]

        ####### GUROBI #######
        nums_noStop = np.zeros(shape=realTime_horizon,dtype='int64')
        params.nums_Tss = dfs[0]['class'].sum()

        P = {}
        if current_t==schedule_start:
            soc_0 = np.ones(params.nums_Tss)*50
        else:
            soc_0 = real_socs.iloc[current_t-1,:].values

        for t in range(realTime_horizon):
            dfs[t].reset_index(drop=True, inplace = True)
            dfs_noStop[t].reset_index(drop=True, inplace = True)
            nums_noStop[t] = len(dfs_noStop[t].name)
            P[t] = dfs_noStop[t]['P'].values # convert to unit MW
        
        print('every t nums_noStop is {}'.format(nums_noStop))

        # model
        model = gp.Model()
        v = {}
        soc = {}
        p_b = {}
        p_ch = {}
        p_disch = {}
        p_g = {}
        p_in = {}
        p_brake = {}
        SOC_tracked_error = {}
        j_in = {}

        obj_expr = gp.LinExpr()
        YV_expr = gp.LinExpr()
        l_expr = gp.LinExpr()
        obj_bra_expr = gp.LinExpr()
        SOC_track_norm = model.addVar(lb=0,ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name = '2_norm')

        for t in range(realTime_horizon):
            # variables
            for i in dfs_noStop[t].index:
                if dfs_noStop[t].loc[i,'class']==1:
                    soc[t,i] = model.addVar(lb=params.soc_min, ub=params.soc_max, vtype=gp.GRB.CONTINUOUS, name='SOC_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_b[t,i] = model.addVar(lb=params.p_b_min, ub=params.p_b_max, vtype=gp.GRB.CONTINUOUS, name='P_Battery_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_ch[t,i] = model.addVar(lb=params.p_ch_min, ub=params.p_ch_max, vtype=gp.GRB.CONTINUOUS, name='P_Battery_Charge_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_disch[t,i] = model.addVar(lb=params.p_disch_min, ub=params.p_disch_max, vtype=gp.GRB.CONTINUOUS, name='P_Battery_DisCharge_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                    p_g[t,i] = model.addVar(lb=params.p_g_min, ub=params.p_g_max, vtype=gp.GRB.CONTINUOUS, name='P_to_Grid_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    p_in[t,i] = model.addVar(lb=params.p_in_min, ub=params.p_in_max, vtype=gp.GRB.CONTINUOUS, name='P_from_Source_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    j_in[t,i] = model.addVar(lb=params.j_in_min, ub=params.j_in_max, vtype=gp.GRB.CONTINUOUS, name='J_from_Source_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    v[t,i] = model.addVar(lb=params.v_tss_min, ub=params.v_tss_max, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                    ## objective function expr
                    obj_expr.addTerms(df_price[t]*params.delta_t,p_in[t,i])
                    ## track_error
                    SOC_tracked_error[t,i] = model.addVar(lb=-1000, ub=1000, vtype=gp.GRB.CONTINUOUS,name='Track_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                
                
                elif dfs_noStop[t].loc[i,'class']==0:
                    v[t,i] = model.addVar(lb=params.v_train_min, ub=params.v_train_max, name='Volt_N{}_H{}'.format(str(dfs_noStop[t].loc[i,'name']),t))
                    p_brake[t,i] = model.addVar(lb=params.p_brake_min, ub=params.p_brake_max, vtype=gp.GRB.CONTINUOUS,name='P_brake_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                    ## train braking obj
                    obj_bra_expr.addTerms(1,p_brake[t,i])
                    pass

            # constraints
            for i in dfs_noStop[t].index:

                if dfs_noStop[t].loc[i,'class'] == 1:
                    ### 光伏发电和电池充放电效率损失
                    # p_b = p_ch/e + p_disch*e
                    model.addConstr(p_b[t,i] == p_ch[t,i]/params.battery_efficiency + p_disch[t,i]*params.battery_efficiency, name='PB_charge_discharge_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    ### soc
                    if t == 0:
                        model.addConstr(soc[t,i] == soc_0[i] + 100*((p_ch[t,i]+p_disch[t,i])*params.delta_t)/(params.battery_max-params.battery_min), name='SOC_change_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    elif t >= 1:
                        model.addConstr(soc[t,i] == soc[t-1,i] + 100*((p_ch[t,i]+p_disch[t,i])*params.delta_t)/(params.battery_max-params.battery_min), name='SOC_change_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))


                    ### track error
                    model.addConstr(SOC_tracked_error[t,i] == soc_tracked[t,i] - soc[t,i],name='track_error_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))

                    ### pv >= p + g
                    l_expr.addTerms(-1, p_b[t,i])
                    l_expr.addTerms(-1, p_g[t,i])
                    l_expr.addConstant(dfs_pv.iloc[t,i])
                    model.addConstr(l_expr == 0, name='PV_Balance_N{}_H{}'.format(i,t))
                    l_expr.clear()

                    ### b + g >= 0
                    model.addConstr(p_b[t,i] + p_g[t,i] >= min(0, dfs_pv.iloc[t,i]), name='BESS_Feasible_{}_H{}'.format(i,t))

                    ### p_in
                    l_expr.addTerms(params.Rs,p_in[t,i])
                    l_expr.addTerms(params.E,v[t,i])
                    model.addConstr(l_expr == params.E*params.E, name='P_IN_N{}_H{}'.format(i,t))
                    l_expr.clear()

                    ### j_in = p_in / rated_volt
                    model.addConstr(j_in[t,i] == p_in[t,i]/params.E)

                    ### power flow of station
                    for j in range(nums_noStop[t]):
                        YV_expr.addTerms(ys[t][i,j],v[t,j])

                    j_sum = dfs_noStop[t].loc[i,'I_rated']
                    if in_PV:
                        p_sum = dfs_pv.iloc[t,i]
                    else:
                        p_sum = 0
                    p_sum = 0
                    j_sum += p_sum/params.E# change 25.3.8 for PV(之前遗漏了PV)
                    #### stopping train load
                    if dfs_noStop[t].loc[i,'upStop'] > 0:
                        j_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'upStop'],'I_rated'].values[0]
                    if dfs_noStop[t].loc[i,'downStop'] > 0 and dfs_noStop[t].loc[i,'downStop'] != dfs_noStop[t].loc[i,'upStop']:
                        j_sum -= dfs[t].loc[dfs[t]['name']==dfs_noStop[t].loc[i,'downStop'],'I_rated'].values[0]
                    #### volt source current
                    l_expr.addTerms(1, j_in[t,i])
                    #### for TSS is 0
                    l_expr.addConstant(j_sum)
                    #### PV_BESS current
                    l_expr.addTerms(1/params.E, p_g[t,i])

                    model.addConstr(-YV_expr == l_expr, name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    YV_expr.clear()
                    l_expr.clear()
                    j_sum = 0

                elif dfs_noStop[t].loc[i,'class'] == 0:
                    ### train braking power balance

                    ### power flow of train
                    for j in range(nums_noStop[t]):
                        YV_expr.addTerms(ys[t][i,j], v[t,j])
                    l_expr.addTerms(-1,p_brake[t,i])
                    model.addConstr(YV_expr == (P[t][i]+p_brake[t,i])/params.E_train, name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                    YV_expr.clear()
                    l_expr.clear()


        # objective function
        model.addGenConstrNorm(SOC_track_norm, SOC_tracked_error.values(), 2.0, name = 'SOC_track_norm_constr')
        # tss_idx_t_0 = (dfs[0]['class']==1).index
        # tss_idx_t_end = (dfs[realTime_horizon-1]['class']==1).index
        # model.addGenConstrNorm(SOC_track_norm,
        #                         [SOC_tracked_error[0,i] for i in range(11)] + [SOC_tracked_error[realTime_horizon-1,i] for i in range(11)], 2.0, name = 'SOC_track_norm_constr')
        model.setObjective(params.realTime_weight_track*SOC_track_norm + params.realTime_weight_brake*obj_bra_expr + params.realTime_weight_p_in*obj_expr, sense=gp.GRB.MINIMIZE)

        time_linearModel = time.time()

        model.setParam('OutputFlag',0)
        # model.setParam('PreSolve',2)
        # model.setParam('NumericFocus',0)
        # model.setParam('FeasibilityTol',1e-6)
        model.params.MIPGap = params.grb_gap


        model.optimize()
        time_linearSlove = time.time()

        # # 获取求解信息
        # # print('the objective value is : {}'.format(model.ObjVal))

        for x in model.getVars():
            if ('Volt' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'Volt_N(\d+)_H(\d+)',x.VarName)
                real_vs.loc['H'+str(current_t),'Volt_N'+match.group(1)] = x.X
                # vs_lin.append(x.X)
            elif ('SOC' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'SOC_N(\d+)_H(\d+)',x.VarName)
                real_socs.loc['H'+str(current_t),'SOC_N'+match.group(1)] = x.X
                # socs.append(x.X)  
            elif ('P_to_Grid' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_to_Grid_N(\d+)_H(\d+)',x.VarName)
                real_p_gs.loc['H'+str(current_t),'P_to_Grid_N'+match.group(1)] = x.X
                # p_gs_lin.append(x.X)
            elif ('P_from_Source' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_from_Source_N(\d+)_H(\d+)',x.VarName)
                real_p_ins.loc['H'+str(current_t),'P_from_Source_N'+match.group(1)] = x.X
                # p_ins_rltm_lin.append(x.X)
            elif ('J_from_Source' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'J_from_Source_N(\d+)_H(\d+)',x.VarName)
                real_j_ins.loc['H'+str(current_t),'J_from_Source_N'+match.group(1)] = x.X
                # j_ins_lin.append(x.X)
            elif ('P_Battery_N' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_Battery_N(\d+)_H(\d+)',x.VarName)
                real_p_bs.loc['H'+str(current_t),'P_Battery_N'+match.group(1)] = x.X
                # p_bs_lin.append(x.X)
            elif 'P_Battery_Charge' in x.VarName and ('H0' in x.VarName):
                match = re.match(r'P_Battery_Charge_N(\d+)_H(\d+)',x.VarName)
                real_p_chs.loc['H'+str(current_t),'P_Battery_Charge_N'+match.group(1)] = x.X
            elif 'P_Battery_DisCharge' in x.VarName and ('H0' in x.VarName):
                match = re.match(r'P_Battery_DisCharge_N(\d+)_H(\d+)',x.VarName)
                real_p_dischs.loc['H'+str(current_t),'P_Battery_DisCharge_N'+match.group(1)] = x.X
            elif ('P_brake' in x.VarName) and ('H0' in x.VarName):
                match = re.match(r'P_brake_N(\d+)_H(\d+)',x.VarName)
                real_p_brakes.loc['H'+str(current_t),'P_brake_N'+match.group(1)] = x.X
                # p_brakes_lin.append(x.X)
            else:
                pass
        
        # stopping train voltage recoverß

        # real_p_brake_lin.loc[current_t, :] = p_brakes_lin.iloc[0,:].astype('float')
        # socs_lin.plot(figsize=(20,6))
        # plt.show()

        time_list = [ time_df_read-time_DF_read
                    , time_linearModel-time_df_read
                    , time_linearSlove-time_linearModel]
        # print('读取 DFs 耗时: {} s\n计算 dfs 和 Y 耗时: {} s\n建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[0],time_list[1],time_list[2],time_list[3]))
        print('建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[1],time_list[2]))
        df_time.loc[current_t,['model_time', 'solve_time', 'sum_time']] = [time_list[1], time_list[2], time_list[1]+time_list[2]]

        model.dispose()
    return {'vs':real_vs
            ,'p_ins': real_p_ins
            ,'SOCs': real_socs
            ,'p_gs': real_p_gs
            ,'p_bs': real_p_bs
            ,'p_chs': real_p_chs
            ,'p_dischs': real_p_dischs
            ,'p_bras': real_p_brakes
            ,'df_time':df_time}