#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Main.ipynb
@Time    :   2025/02/19 17:30:05
@Author  :   勾子瑭
@ desc   :   1. 改用了分块存储的 real-time 还有 
             2. 限制充放电功率之后的dayahead 使之趋向更加平滑
             3. 调整电池的充放电和
'''

# here put the import lib
import Funcs
import pandas as pd
pd.set_option('display.float_format',lambda x : '%.6f' % x)
import numpy as np
np.set_printoptions(
    precision=4,    # 小数点后4位
    suppress=True,   # 禁止科学计数法
    threshold=80,    # 超过20个元素显示缩略
    edgeitems=10,     # 缩略时首尾显示3个
    linewidth=200    # 每行120字符
)
import time
import gurobipy as gp
import matplotlib.pyplot as plt
import re
import warnings
from MyProcess import Process
from Funcs.controlParams import Params
import warnings
warnings.filterwarnings('ignore')
from Funcs.PF.power_flow_onlineDisp import power_flow_onlineDisp
from Funcs.OPF.efficiencyOPF.dayAheadEfficiencyOPF import dayAhead_OPF_efficiency_avg
from Funcs.OPF.efficiencyOPF.offlineEfficiencyOPF import offline_OPF_efficiency_org
from Funcs.OPF.efficiencyOPF.offlineEfficiencyOPF_complete import offline_OPF_efficiency_org_complete

from Funcs.OPF.efficiencyOPF.realTimeEfficiencyOPF import realTime_OPF_efficiency_lin, realTime_OPF_efficiency_cplx, realTime_OPF_efficiency_org
from Funcs.utils.SOCInterpolate import SOCInterpolate
from Funcs.PF.power_flow_check import power_flow_checkOnce
from Funcs.utils.dataColumnIndexRename import dataColumnIndexRename
from Funcs.utils.addPVNoise import add_multiplicative_noise, add_normal_noise
# from Funcs.Show.plotTractionTopo{log}y import plot_tracition_topo{log}y
from Funcs.LD.loadDistribute import loadDistribute
import json
import copy
import random

log = "log"
SEED = 15  # 你可以统一设置为15或其他任意整数
# random.seed(SEED)
# np.random.seed(SEED)

par = Params()
par.offline_horizon = int(18*60*60)
par.online_horizon = 15
prcs = Process(par)
# prcs.load_PV_Price()
prcs.PV_Price = prcs.load_PV_Price_different(different=False, seed=15)
prcs.PV_Price = prcs.PV_Price.set_index("time").between_time("5:00", "23:00")
prcs.PV_Price = prcs.PV_Price.resample(f"{len(prcs.PV_Price)//par.offline_horizon}S").mean()
# prcs.PV_Price = prcs.PV_Price.resample(f"{len(prcs.PV_Price)//par.offline_horizon}S").mean()

prcs.params.address_DF = "/home/gzt/Codes/2STAGE/Funcs/TractionDataCreate/dataset/oneDay_biStart_4m42s_noised.csv"
prcs.compute_DFs()
prcs.compute_Ys()
prcs.Ys.__len__()
print('数据加载完成')
# prcs.PV_Price['PV'] = prcs.PV_Price['PV'].apply(lambda x:round(x,6))

par.battery_max_kwh = 500
par.battery_max = par.battery_max_kwh*3.6
par.battery_min = 0


# 尝试更小的充放电功率
par.p_b_max = 10
par.p_b_min = -10
par.p_g_max = 10
par.p_g_min = -10

par.p_ch_max = 3
par.p_ch_min = 0
par.p_disch_max = 0
par.p_disch_min = -3

par.p_brake_max = 5
par.p_brake_min = 0


par.soc_min = 1
par.soc_max = 99

par.__dict__

# parOffline
parOffline = copy.deepcopy(par)
parOffline.soc_min = 0
parOffline.soc_max = 100

# parOnline
parOnline = copy.deepcopy(par)
parOnline.soc_min = 0
parOnline.soc_max = 100

# parRelax
parRelax = copy.deepcopy(par)
parRelax.soc_min = 0
parRelax.soc_max = 100
parRelax.p_in_min = -30
parRelax.p_in_max = 30
parRelax.v_tss_max = 1.85
parRelax.v_tss_min = 1.2

# output 文件
import datetime
strTime = datetime.datetime.today().strftime(format="%Y,%m,%d %H:%M:%S")
output = "/home/gzt/Codes/2STAGE/{log}/output.txt"
import json
# # 写入记录
# with open(output, "w", encoding='utf-8') as f:
#     f.write("simulation time:{strTime}\n")
#     f.write(f"par:\n")
#     json.dump(par.__dict__, f)


# load PV and Price
ax_PV_Price = prcs.PV_Price.plot(figsize=(15,2), title = "PV and Price")
ax_PV_Price.figure.savefig(f'/home/gzt/Codes/2STAGE/{log}/PV_Price.pdf', format="pdf")  # 保存为文件
prcs.PV_Price.to_csv(f"/home/gzt/Codes/2STAGE/{log}/PV_Price.csv")

PV_Price_Noised = copy.deepcopy(prcs.PV_Price)
PV_Price_Noised[list(range(1,prcs.params.nums_Tss+1))] = PV_Price_Noised[list(range(1,prcs.params.nums_Tss+1))].apply(add_multiplicative_noise)
ax_PV_Price_Noised = PV_Price_Noised.plot(figsize=(15,2), title = "PV and Price Noised")
ax_PV_Price_Noised.figure.savefig(f'/home/gzt/Codes/2STAGE/{log}/PV_Price_Noised.pdf', format="pdf")  # 保存为文件
PV_Price_Noised.to_csv(f"/home/gzt/Codes/2STAGE/{log}/PV_Price_Noised.csv")


# ---------- dayAhead dispatch
par.realTime_weight_brake = 0
par.realTime_weight_p_in = 20
par.realTime_weight_L1 = 7
par.realTime_weight_track = 100

sec = 30
prcs.load_avg = pd.read_csv(f'/home/gzt/Codes/2STAGE/Data/load_avg/PF_org_load_avg/PF_18h.csv').iloc[:prcs.params.offline_horizon,:]
prcs.load_avg = loadDistribute(prcs.load_avg, sec, prcs.params)
if "Unnamed: 0" in prcs.load_avg.columns:
    prcs.load_avg = prcs.load_avg.drop(columns=["Unnamed: 0"])
prcs.load_avg.columns = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11']
prcs.load_avg = prcs.load_avg.apply(lambda x:round(x,6))

# dayahead pv
prcs.PV_Price = loadDistribute(prcs.PV_Price, sec, prcs.params)
# prcs.PV_Price = prcs.PV_Price.resample(f"{sec}S").mean()
prcs.PV_Price.to_csv(f"/home/gzt/Codes/2STAGE/{log}/dayAhead_PV_Price.csv")
ax_dayahead_PV_Price = prcs.PV_Price.plot(figsize=(15,2), title = "PV and Price")
ax_dayahead_PV_Price.figure.savefig(f'/home/gzt/Codes/2STAGE/{log}/dayAhead_PV_Price.pdf', format="pdf")  # 保存为文件

prcs.PV_Price.reset_index(drop=True, inplace=True)
PV_Price_Noised.reset_index(drop=True, inplace=True)

dayAhead_data_sec1 = dayAhead_OPF_efficiency_avg(schedule_start=0
                                                , schedule_horizon=len(prcs.load_avg)
                                                , load_avg= prcs.load_avg
                                                , params= par
                                                , in_PV=True
                                                , delta_t = sec
                                                , PV_Price= prcs.PV_Price
                                                , path=f"/home/gzt/Codes/2STAGE/{log}/dayAhead/dayAhead_solve_log.log")

print('SOC 进行插值\n')
SOC_sec1 = SOCInterpolate(dayAhead_data_sec1['SOCs'], sec, prcs.params.offline_horizon, prcs.params.nums_Tss)
p_in_sec1 = SOCInterpolate(dayAhead_data_sec1['p_ins'], sec, prcs.params.offline_horizon, prcs.params.nums_Tss)

# SOC_sec1.plot(figsize = (15,2))
ax_dayAhead_soc = dayAhead_data_sec1['SOCs'].plot(figsize=(15,2), title="dayahead soc curve")
ax_dayAhead_soc.figure.savefig(f'/home/gzt/Codes/2STAGE/{log}/dayAhead_soc.pdf', format="pdf")  # 保存为文件
# with open(output, "a", encoding='utf-8') as f:
#     f.write("\n")
#     f.write(f"par:\n")
#     json.dump(par.__dict__, f)

# 保存调度结果
for key in dayAhead_data_sec1:
    dayAhead_data_sec1[key].to_csv(f"/home/gzt/Codes/2STAGE/{log}/dayAhead/{key}.csv")

# ---------- dayAhead 插值
PV_Price = prcs.PV_Price.drop(columns=["PRICE"])
PV_Price.columns = list(range(1,prcs.params.nums_Tss+1))

prcs.load_avg.columns = list(range(1,prcs.params.nums_Tss+1))
for df in dayAhead_data_sec1.values():
    df.columns = list(range(1,prcs.params.nums_Tss+1))

SOC_sec1.columns = list(range(1,12))
p_bs_sec1 = (SOC_sec1.iloc[1:,:].reset_index(drop=True) - SOC_sec1.iloc[0:len(SOC_sec1)-1,:].reset_index(drop=True))*(par.battery_max-par.battery_min)/100
p_ch_sec1 = p_bs_sec1[p_bs_sec1>0]
p_ch_sec1.fillna(value=0,inplace=True)
p_disch_sec1 = p_bs_sec1[p_bs_sec1<0]
p_disch_sec1.fillna(value=0, inplace=True)
# 加上功率系数
p_bs_sec1[p_bs_sec1<0] = p_bs_sec1[p_bs_sec1<0]*par.battery_efficiency
p_bs_sec1[p_bs_sec1>0] = p_bs_sec1[p_bs_sec1>0]/par.battery_efficiency

p_gs_sec1 = (prcs.PV_Price[p_bs_sec1.columns] - p_bs_sec1)

dayAhead_data_sec1_filled = {}
dayAhead_data_sec1_filled["p_chs"] = p_ch_sec1
dayAhead_data_sec1_filled["p_dischs"] = p_disch_sec1
dayAhead_data_sec1_filled["p_bs"] = p_bs_sec1
dayAhead_data_sec1_filled["p_gs"] = p_gs_sec1
dayAhead_data_sec1_filled["SOCs"] = SOC_sec1
for key in dayAhead_data_sec1_filled:
    dayAhead_data_sec1[key].to_csv(f"/home/gzt/Codes/2STAGE/{log}/dayAhead/filled/{key}.csv")

parOnline.realTime_weight_p_in = 20
parOnline.realTime_weight_L1 = 20
parOnline.realTime_weight_track = 600
parOnline.realTime_weight_brake = 80

parOnline.onlineLinearGpParams = {'OutputFlag': 0,
  'MIPGap': 0.1,
  'TimeLimit': 1}

parOnline.onlineNonlinearGpParams={'OutputFlag': 0,
  'NonConvex': 2,
  'MIPGap': 0.1,
  'TimeLimit': 1}

# ---------- online dispatch
# 开始 track
print("开始进行online tracking\n")
online_data_sec1 = realTime_OPF_efficiency_cplx(schedule_start=0
                                    ,schedule_horizon=parOnline.offline_horizon
                                    ,realTime_horizon=parOnline.online_horizon
                                    ,DFs=prcs.DFs
                                    ,DFs_noStop=prcs.DFs_noStop
                                    ,PV_Price=PV_Price_Noised
                                    ,Ys = prcs.Ys
                                    ,SOCs = SOC_sec1
                                    ,p_ins=p_in_sec1
                                    ,params=parOnline
                                    ,in_PV=True)

ax_online_soc = online_data_sec1['SOCs'].plot(figsize = (15,2), title="online soc curve")
ax_online_soc.figure.savefig(f'/home/gzt/Codes/2STAGE/{log}/online_soc.pdf', format="pdf")  # 保存为文件

# 保存调度结果
for key in online_data_sec1:
    online_data_sec1[key].to_csv(f"/home/gzt/Codes/2STAGE/{log}/online/{key}.csv")
# with open(output, "a", encoding='utf-8') as f:
#     f.write("\n")
#     f.write(f"parOnline:\n")
#     json.dump(parOnline.__dict__, f)

# ----------offline dispatch
par.realTime_weight_p_in = 20
par.realTime_weight_L1 = 20
# par.realTime_weight_track = 800
# par.realTime_weight_track = 600
par.realTime_weight_track = 400
par.realTime_weight_brake = 80

offline_data_org = offline_OPF_efficiency_org(schedule_start=0
                                            , schedule_horizon=par.offline_horizon
                                            , DFs=prcs.DFs
                                            , DFs_noStop=prcs.DFs_noStop
                                            , Ys=prcs.Ys
                                            , params= par
                                            , in_PV=True
                                            , PV_Price= PV_Price_Noised
                                            , plot=False
                                            , path = f"/home/gzt/Codes/2STAGE/{log}/offline/offline_solve_log.log")

offline_soc = offline_data_org['SOCs'].plot(figsize = (15,2), title = "offline soc curve")
offline_soc.figure.savefig(f"/home/gzt/Codes/2STAGE/{log}/offline_soc.pdf", format='pdf')  # 保存为矢量图

# 保存调度结果
for key in offline_data_org:
    offline_data_org[key].to_csv(f"/home/gzt/Codes/2STAGE/{log}/offline/{key}.csv")
# with open(output, "a", encoding='utf-8') as f:
#     f.write("\n")
#     f.write(f"parOffline:\n")
#     json.dump(parOffline.__dict__, f)


# ----------power flow
parRelax.p_g_max = 30
parRelax.p_g_min = -30
parRelax.p_b_max = 30
parRelax.p_b_min = -30
parRelax.p_ch_max = 30
parRelax.p_ch_min = -30
parRelax.p_disch_max = 30
parRelax.p_disch_min = -30
parRelax.p_in_max = 40
parRelax.p_in_min = -40
parRelax.v_tss_max = 2
parRelax.v_tss_min = 1
parRelax.v_train_max = 2
parRelax.v_train_min = 1
parRelax.p_brake_max = 5
parRelax.p_brake_min = 0

# 处理 offline 数据
for key in offline_data_org:
    offline_data_org[key].index = offline_data_org[key].index.astype(str)
    if isinstance(offline_data_org[key].index[0], str) and 'H' in offline_data_org[key].index[0]:
        offline_data_org[key].index = offline_data_org[key].index.map(lambda x : int(x.split("H")[1]))
for key in offline_data_org:
    offline_data_org[key].columns = offline_data_org[key].columns.astype(str)
    if isinstance(offline_data_org[key].columns[0], str) and '_N' in offline_data_org[key].columns[0]:
        offline_data_org[key].columns = offline_data_org[key].columns.map(lambda x : int(x.split("_N")[1]))
offline_data_org['p_bras']['oldIndex'] = offline_data_org['p_bras'].index.astype(int)
joinBy = pd.DataFrame(range(par.offline_horizon), columns=['joinBy'])

filled_p_bras = offline_data_org['p_bras'].join(joinBy, on="oldIndex", how="right")
filled_p_bras.index = range(par.offline_horizon)
filled_p_bras.drop(columns=["oldIndex", "joinBy"], inplace=True)

old_p_bras = offline_data_org['p_bras'].copy()
offline_data_org['p_bras'] = filled_p_bras

# filled_p_bras = offline_data_org['p_bras'].join(joinBy, on="oldIndex", how="right")
# filled_p_bras.index = range(par.offline_horizon)
# filled_p_bras.drop(columns=["oldIndex", "joinBy"], inplace=True)

# online_data_sec1['p_bras']['oldIndex'] = online_data_sec1['p_bras'].index
# joinBy = pd.DataFrame(range(par.offline_horizon), columns=['joinBy'])

# filled_p_bras = online_data_sec1['p_bras'].join(joinBy, on="oldIndex", how="right")
# filled_p_bras.index = range(par.offline_horizon)
# filled_p_bras.drop(columns=["oldIndex", "joinBy"], inplace=True)

# old_p_bras = online_data_sec1['p_bras'].copy()
# online_data_sec1['p_bras'] = filled_p_bras

# filled_p_bras = online_data_sec1['p_bras'].join(joinBy, on="oldIndex", how="right")
# filled_p_bras.index = range(par.offline_horizon)
# filled_p_bras.drop(columns=["oldIndex", "joinBy"], inplace=True)

# 处理 online 数据
for key in online_data_sec1:
    online_data_sec1[key].index = online_data_sec1[key].index.astype(str)
    if isinstance(online_data_sec1[key].index[0], str) and 'H' in online_data_sec1[key].index[0]:
        online_data_sec1[key].index = online_data_sec1[key].index.map(lambda x : int(x.split("H")[1]))
for key in online_data_sec1:
    online_data_sec1[key].columns = online_data_sec1[key].columns.astype(str)
    if isinstance(online_data_sec1[key].columns[0], str) and '_N' in online_data_sec1[key].columns[0]:
        online_data_sec1[key].columns = online_data_sec1[key].columns.map(lambda x : int(x.split("_N")[1]))
online_data_sec1['p_bras']['oldIndex'] = online_data_sec1['p_bras'].index.astype(int)
joinBy = pd.DataFrame(range(par.offline_horizon), columns=['joinBy'])

filled_p_bras = online_data_sec1['p_bras'].join(joinBy, on="oldIndex", how="right")
filled_p_bras.index = range(par.offline_horizon)
filled_p_bras.drop(columns=["oldIndex", "joinBy"], inplace=True)

old_p_bras = online_data_sec1['p_bras'].copy()
online_data_sec1['p_bras'] = filled_p_bras

# filled_p_bras = online_data_sec1['p_bras'].join(joinBy, on="oldIndex", how="right")
# filled_p_bras.index = range(par.offline_horizon)
# filled_p_bras.drop(columns=["oldIndex", "joinBy"], inplace=True)

"""
还需要对比 带入潮流方程求解后的 cost
"""
from Funcs.PF.power_flow_onlineDisp_complete import power_flow_onlineDisp_complete
# 先计算 offline 的潮流
PF_offline_org = power_flow_onlineDisp(online_data = offline_data_org
                            ,params=parRelax
                            ,power_flow_start = 0
                            ,power_flow_horizon= len(offline_data_org['p_gs'])
                            ,DFs = prcs.DFs
                            ,DFs_noStop = prcs.DFs_noStop
                            ,PV_Price = PV_Price_Noised
                            ,in_PV=True
                            ,Ys = prcs.Ys
                            ,plot = False)

PF_offline_org['p_ins'] = PF_offline_org['p_ins'].astype('float').reset_index(drop=True)
PF_offline_org['p_ins'].columns = offline_data_org['p_ins'].columns
# 保存调度结果
for key in PF_offline_org:
    PF_offline_org[key].to_csv(f"/home/gzt/Codes/2STAGE/{log}/offline/powerFlow/{key}.csv")


PF_online_data_sec1 = power_flow_onlineDisp(online_data = online_data_sec1
                            ,params=parRelax
                            ,power_flow_start = 0
                            ,power_flow_horizon= len(online_data_sec1['p_gs'])
                            ,DFs = prcs.DFs
                            ,DFs_noStop = prcs.DFs_noStop
                            ,PV_Price = PV_Price_Noised
                            ,in_PV=True
                            ,Ys = prcs.Ys
                            ,plot = False)
PF_online_data_sec1['p_ins'] = PF_online_data_sec1['p_ins'].astype('float').reset_index(drop=True)
PF_online_data_sec1['p_ins'].columns = offline_data_org['p_ins'].columns
# 保存调度结果
for key in PF_online_data_sec1:
    PF_online_data_sec1[key].to_csv(f"/home/gzt/Codes/2STAGE/{log}/online/powerFlow/{key}.csv")
# with open(output, "a", encoding='utf-8') as f:
#     f.write("\n")
#     f.write(f"parRelax:\n")
#     json.dump(parRelax.__dict__, f)


# ---------- Cost Compute
def cost_compute(p_ins, delta_t):
    pIn_NoNeg = p_ins.astype('float').reset_index(drop=True)
    for col in pIn_NoNeg.columns:
        pIn_NoNeg[col].map(lambda x:max(x,0))
    # pIn_NoNeg = p_ins.sum(axis=1).map(lambda x:max(x,0)).reset_index(drop=True)
    df_cost = delta_t * PV_Price_Noised['PRICE'].reset_index(drop=True) * pIn_NoNeg.sum(axis = 1)
    cost = df_cost.sum().sum()
    return cost, df_cost

cost_org, cost_org_df = cost_compute(p_ins=PF_offline_org['p_ins'].iloc[:par.offline_horizon-par.online_horizon,], delta_t=1)
cost_rT_sec1, cost_rT_df_sec1 = cost_compute(p_ins=PF_online_data_sec1['p_ins'], delta_t=1)

print('实际潮流的 cost 为')
print(f'nonlinear cost = {cost_org}')
print(f'real time cost {sec} = {cost_rT_sec1}')

# with open(output, "a", encoding='utf-8') as f:
#     f.write("\n")
#     f.write(f"实际潮流的 cost 为\n")
#     f.write(f"nonlinear cost = {cost_org}\n")
#     f.write(f"real time cost {sec} = {cost_rT_sec1}\n")


print("\n")
print("开始检查反向潮流")

print("\n")
print("开始检查约束条件")


checkData = dataColumnIndexRename(PF_online_data_sec1)

# 统计一下有多少违反约束的时刻
error_cnt = 0
records = list()
for idx in range(len(checkData['p_ins'].index)):
    print(idx)
    temp, tempDF = power_flow_checkOnce(checkData, checkTime=int(idx), DFs=prcs.DFs, DFs_noStop=prcs.DFs_noStop, Y=prcs.Ys, params=par, feasibleTol=0.00001)
    records.append(tempDF)
    if temp==True:
        continue
    else:
        print('违反约束')
        error_cnt += 1
        continue
recordDF = pd.concat(records)
recordDF.to_csv(f"/home/gzt/Codes/2STAGE/{log}/recordDF.csv", index=False)



### 原始的 power flow 用来对比算法的效果
def power_flow_org(power_flow_start, power_flow_horizon, DFs, DFs_noStop, PV_Price, Ys, params,  in_PV = False, use_brake=False):
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
                if use_brake:
                    p_brake[t,i] = model.addVar(lb=params.p_brake_min, ub=params.p_brake_max, vtype=gp.GRB.CONTINUOUS,name='P_brake_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
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
                if in_PV:
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
                if use_brake:
                    model.addQConstr(q_expr  == P[t][i] + p_brake[t,i], name='PF_N{}_H{}'.format(dfs_noStop[t].loc[i,'name'],t))
                else:
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

        # # stopping train voltage recover
        # for col in PF_vs_org.columns[11:]:
        #     match = re.match('Volt_N(\d+)',col)
        #     num_train = int(match.group(1))
        #     for idx in PF_vs_org[col].index:
        #         if pd.isna(PF_vs_org.loc[idx,col]):
        #             match = re.match('H(\d+)',idx)
        #             num_horizon = int(match.group(1))
        #             current_DF = DFs[num_horizon]
        #             if num_train in current_DF['name'].values:
        #                 stoppingTss = current_DF.loc[current_DF['name']==num_train,'stopping'].values[0]
        #                 if stoppingTss > 0 :
        #                     # print(stoppingTss)
        #                     PF_vs_org.loc[idx,col] = PF_vs_org.loc[idx,'Volt_N'+str(stoppingTss)]
        #         else: pass
        # time_list = [ time_df_read-time_DF_read
        #             , time_model-time_df_read
        #             , time_slove-time_model]
        # print('建模耗时: {} s\n求解耗时: {} s\n'.format(time_list[1],time_list[2]))

    PF_data = {'vs':PF_vs_org.astype('float')
               ,'p_ins':PF_p_ins.astype('float')}
    return PF_data

PF_data = power_flow_org(power_flow_start=0
                        , power_flow_horizon=prcs.params.offline_horizon
                        , DFs=prcs.DFs
                        , DFs_noStop=prcs.DFs_noStop
                        , PV_Price=PV_Price_Noised
                        , Ys=prcs.Ys
                        , params=prcs.params
                        ,  in_PV = True)
for key in PF_data.keys():
    PF_data["key"].to_csv(f"/home/gzt/Codes/2STAGE/{log}/powerFLow/{key}.csv", index = False)