#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   MyProcess.py
@Time    :   2025/02/19 10:36:09
@Author  :   勾子瑭
'''

# here put the import lib
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import re
import warnings
from Funcs import controlParams as Par
from IPython.display import clear_output
from Funcs.admittanceMatrix import admittanceMatrix
from Funcs.admittanceMatrix_split import admittanceMatrix_split

from Funcs.OPF.dayAheadOPF import dayAhead_OPF_avg
from Funcs.OPF.offlineOPF import offline_OPF_org, offline_OPF_lin
from Funcs.OPF.realTimeOPF import realTime_OPF_cplx, realTime_OPF_lin, realTime_OPF_org
from Funcs.PF.power_flow_org import power_flow_org
from Funcs.PF.power_flow_lin import power_flow_lin
from Funcs.PF.power_flow_onlineDisp import power_flow_onlineDisp
import random
import os

class Process():
    def __init__(self, params=Par.Params()):
        self.params = params
        
        self.Ys = list()
        self.Ycs = list()
        self.Yrs = list()
        # np.load(self.params.address_Ys, allow_pickle=True, fix_imports=True)

        self.DFs = list()

        self.DFs_noStop = list()

        self.PV_Price = None

        self.PV_Price_avg = None

        self.SOCs = None

        self.offline_data_lin = None

        self.offline_data_org = None

        self.offline_data_avg = None

        self.online_data_lin = None

        self.online_data_org = None

        self.powerFlow_data_org = None
        
        self.powerFlow_data_lin = None

        self.PF_data = None

        self.load_avg = None
        pass


    def load_PV_Price(self, address_PV_Price=None):
        print('开始加载 PV_Price 数据')
        if address_PV_Price == None:
            PV_Price = pd.read_csv(self.params.address_PV_Price)
        else:
            PV_Price = pd.read_csv(address_PV_Price)
        start_idx = PV_Price[PV_Price['time']==self.params.start_time].index.values[0] if 'time' in PV_Price.columns else 0
        end_idx = start_idx+self.params.offline_horizon

        PV_Price = PV_Price.iloc[start_idx:end_idx,:]
        if PV_Price['PV'].max() > 100:
            PV_Price['PV'] = PV_Price['PV']/1e3 #如果有大于100MW的，肯定是异常值 kw 换成 MW
        PV_Price.reset_index(drop=True,inplace=True)
        for i in range(self.params.nums_Tss):
            PV_Price[i+1] = PV_Price['PV']
        PV_Price.drop(columns=["PV"], inplace=True)
        self.PV_Price = PV_Price
        del PV_Price
        print('PV_Price 加载完成')
        pass


    def load_PV_same(self, address_PV_Price=None, seed = 0):
        """
        用来生成不同的PV
        """
        random.seed(seed)
        if address_PV_Price == None:
            df = pd.read_excel("./Data/PVs/项目B-1#光伏-有功功率-2023年5月1日-2023年7月31日_AfterFilled.xlsx", parse_dates=['time'])
        else:
            df = pd.read_excel(address_PV_Price, parse_dates=['time'])
            
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace = True)
        # 获取所有唯一日期
        unique_dates = df["time"].dt.date.unique()

        # 生成每天的 DataFrame 列表
        daily_PV_origins_list = [df[df["time"].dt.date == day] for day in unique_dates]

        use_list = random.choices(daily_PV_origins_list, k=1)
        use_list = use_list*self.params.nums_Tss
        # use_list = random.choices(daily_PV_origins_list, k=11)
        res = use_list[0]["time"].dt.time
        res.reset_index(drop=True, inplace = True)
        for idx, df in enumerate(use_list):
            df.reset_index(drop=True, inplace = True)
            if "time" in df.columns:
                df.drop(columns = ["time"], inplace = True)
            df.columns = [idx+1]
            if df[idx+1].max()>50:
                df[idx+1] = df[idx+1]/1000
            res = pd.concat([res, df], axis=1)
        PV_Price = pd.read_csv('./Data/PV_Price/12s/PV_Price_12s_2023-04-26.csv')
        # res = pd.concat([res, PV_Price['PRICE']], axis=1)
        return res

    def load_PV_different(self, address_PV_Price=None, seed = 0):
        """
        用来生成不同的PV
        """
        random.seed(seed)
        if address_PV_Price == None:
            df = pd.read_excel("./Data/PVs/项目B-1#光伏-有功功率-2023年5月1日-2023年7月31日_AfterFilled.xlsx", parse_dates=['time'])
        else:
            df = pd.read_excel(address_PV_Price, parse_dates=['time'])
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace = True)
        # 获取所有唯一日期
        unique_dates = df["time"].dt.date.unique()

        # 生成每天的 DataFrame 列表
        daily_PV_origins_list = [df[df["time"].dt.date == day] for day in unique_dates]

        use_list = random.choices(daily_PV_origins_list, k=self.params.nums_Tss)
        # use_list = random.choices(daily_PV_origins_list, k=11)
        res = use_list[0]["time"].dt.time
        res.reset_index(drop=True, inplace = True)
        for idx, df in enumerate(use_list):
            df.reset_index(drop=True, inplace = True)
            if "time" in df.columns:
                df.drop(columns = ["time"], inplace = True)
            # df.columns = [f"{idx+1}"]
            # if df[f"{idx+1}"].max()>50:
            #     df[f"{idx+1}"] = df[f"{idx+1}"]/1000
            df.columns = [idx+1]
            if df[idx+1].max()>50:
                df[idx+1] = df[idx+1]/1000
            res = pd.concat([res, df], axis=1)
        PV_Price = pd.read_csv('./Data/PV_Price/12s/PV_Price_12s_2023-04-26.csv')
        # res = pd.concat([res, PV_Price['PRICE']], axis=1)
        return res


    def load_PV_Price_different(self, address_PV_Price=None, seed = 0, different = False):
        """
        用来生成不同的PV, 并且与电价拼接起来
        """
        if different:
            res = self.load_PV_different(seed = seed)
        else:
            res = self.load_PV_same(seed=seed)

        # 转换为 datetime（时间部分）
        res["time"] = pd.to_datetime(res["time"], format="%H:%M:%S").dt.time
        # 添加固定日期（必须与 end_time 同一天）
        ref_date = pd.to_datetime("2025-06-28").normalize()  # 关键修改：指定目标日期
        res["time"] = res["time"].apply(lambda t: ref_date + pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second))
        # 设为索引
        res = res.set_index("time")
        # 定义完整时间范围（同一天）
        start_time = res.index.min()
        end_time = pd.to_datetime("2025-06-28 23:59:59")
        full_range = pd.date_range(start=start_time, end=end_time, freq="1S")

        # 重新索引并填充
        res_seconds = res.reindex(full_range)
        res_seconds = res_seconds.ffill().bfill()  # 双向填充确保无 NaN

        res_seconds.reset_index(inplace=True)
        res_seconds.rename(columns = {"index":"time"}, inplace=True)
        res_seconds["hour"] = res_seconds["time"].dt.hour

        price = pd.read_csv('./Data/Price/price.csv', parse_dates=["time"])
        price["time"] = pd.to_datetime(price["time"],format = "%H")
        price["time"] = price["time"].dt.hour
        price.drop_duplicates("time",inplace=True)
        price.rename(columns={"time":"hour"}, inplace=True)

        result = pd.merge(
            res_seconds,
            price,
            how="left",
            left_on="hour",
            right_on="hour",  # 或 right_index=True 如果 price 的索引是 hour
            suffixes=("", "_price")  # 空字符串表示保留左列名
        )
        result.drop(columns=["hour"], inplace = True)
        return result


    def compute_DFs(self,save = False):
        # 只是做了一下切割
        DF = pd.read_csv(self.params.address_DF)
        DFs = list()
        DFs_noStop = list()
        idx_split = DF[DF.name==1].index
        all_time_interval = self.params.offline_horizon
        percent = 0 # 进度
        print('开始计算 DFs')
        for i in range(all_time_interval-1):
            if percent != int(100*(i+1)/all_time_interval):
                # 清除之前的输出
                clear_output(wait=True)
                print('DFs 计算完成 {}%'.format(percent))
                percent = int(100*(i+1)/all_time_interval)
            
            DFs.append(DF.loc[idx_split[i]:idx_split[i+1]-1,:])
            p_df = DFs[i]
            p_df.reset_index(drop=True,inplace=True)
            DFs_noStop.append(p_df[p_df['stopping']==0])
        # 补充最后一个时刻的 DF
        DFs.append((DF.loc[idx_split[-1]:,:]).reset_index(drop=True))
        p_df = DFs[-1]
        p_df.reset_index(drop=True,inplace=True)
        DFs_noStop.append(p_df[p_df['stopping']==0])
        clear_output(wait=True)
        print('DFs 计算完成')

        self.DFs = DFs
        del DFs
        self.DFs_noStop = DFs_noStop
        del DFs_noStop
        pass


    def compute_Ys_pool(self, save=False):
        self.Ys = []
        self.Ycs = []
        self.Yrs = []

        from multiprocessing import Pool
        all_time_interval = self.params.offline_horizon
        print('开始并行计算 Ys')
        
        # 并行计算
        with Pool(processes=16) as pool:  # 使用 8 个核心
            results = pool.starmap(
                admittanceMatrix_split,
                [(self.DFs[i], self.DFs_noStop[i], self.params.r_r, self.params.r_c) 
                for i in range(all_time_interval)]
            )
        self.Ys  = [res["Y"]  for res in results]
        self.Ycs = [res["Yc"] for res in results]
        self.Yrs = [res["Yr"] for res in results]  # 直接替换，避免 append 的锁竞争

        if save:
            np.save('admittance_18h_delay4m.npy', np.array(self.Ys, dtype=object))
        print('Ys 计算完成')
        return self.Ys, self.Ycs, self.Yrs

    def compute_Ys(self, save=False):
        self.Ys = []
        self.Ycs = []
        self.Yrs = []

        all_time_interval = self.params.offline_horizon
        percent = 0 # 进度
        print('开始计算 Ys')
        for i in range(all_time_interval):
            if percent != int(100*(i+1)/all_time_interval):
                # 清除之前的输出
                clear_output(wait=True)
                print('Ys 计算完成 {}%'.format(percent))
                percent = int(100*(i+1)/all_time_interval)
            # self.Ys.append(admittanceMatrix(self.DFs[i], self.DFs_noStop[i],r_r=self.params.r_r,r_c=self.params.r_c)["Y"])
            Y_map = admittanceMatrix_split(self.DFs[i], self.DFs_noStop[i],r_r=self.params.r_r,r_c=self.params.r_c)
            self.Ys.append(Y_map["Y"])
            self.Ycs.append(Y_map["Yc"])
            self.Yrs.append(Y_map["Yr"])

        clear_output(wait=True)

        if save == True:
            a = np.array(self.Ys,dtype=object)
            np.save('admittance_18h_delay4m',a,allow_pickle=True,fix_imports=True)
        print('Ys 计算完成')
        return self.Ys, self.Ycs, self.Yrs


    def LoadDistribute(self, filename, start_h, end_h):
        res = pd.DataFrame(columns=['name', 'class', 'upPre', 'upPost', 'downPre', 'downPost', 'preTss',
        'distance_preTss', 'postTss', 'distance_postTss', 'P', 'load', 'location',
        'distance_upPre', 'distance_upPost', 'distance_downPre',
        'distance_downPost', 'stopping', 'upStop', 'downStop', 'E', 'Rs', 'V','I_rated'])
        
        percent = 0
        for h,dfs in enumerate(self.DFs[start_h:end_h]):
            p = dfs
            p['load']=0
            for i in p.index:
                if p.loc[i,'class'] == 1 or p.loc[i,'stopping'] == 1:
                    continue
                P_train = p.loc[i,'P']
                preTss = p.loc[p['name']==p.loc[i,'preTss']].index
                distance_preTss = p.loc[i,'distance_preTss']
                postTss = p.loc[p['name']==p.loc[i,'postTss']].index
                distance_postTss = p.loc[i,'distance_postTss']
                # print(P_train,preTss,distance_preTss,postTss,distance_postTss)
                p.loc[preTss,'load'] += P_train*distance_postTss/(distance_postTss+distance_preTss)
                p.loc[postTss,'load'] += P_train*distance_preTss/(distance_postTss+distance_preTss)

            res = pd.concat([res,p],axis=0, ignore_index=True)

            if percent != int(100*(h-start_h+1)/(end_h-start_h)):
                # 清除之前的输出
                clear_output(wait=True)
                print('Load Distribution 完成 {}%'.format(percent))
                percent = int(100*(h-start_h+1)/(end_h-start_h))

        print('计算完成')
        res.to_csv(f'./Data/DFs/{filename}',index=False)
        print('保存成功')
        pass


    def PV_Price_sec(self, sec_interval, address_src, address_output, how='mean'):
        df = pd.read_csv(address_src)
        df['time'] = pd.to_datetime(df['time'])
        os.mkdir(path = address_output+f'/{sec_interval}')
        df['time'] = pd.to_datetime(df['time'])
        files = [f'PV_Price_{sec_interval}_{i}' for i in df['time'].dt.strftime('%Y-%m-%d').unique()]
        print(f'一共分成了 {len(files)} 个文件，分别是：',files)
        dfs = df.groupby(pd.Grouper(key='time',freq='d'))
        for groupby,d in dfs:
            filename = f'PV_Price_{sec_interval}_'+groupby.strftime('%Y-%m-%d')+'.csv'
            if how == 'mean':
                d = d.groupby(pd.Grouper(key='time', freq=sec_interval)).mean().reset_index()
            elif how == 'max':
                d = d.groupby(pd.Grouper(key='time', freq=sec_interval)).max().reset_index()
            elif how == 'min':
                d = d.groupby(pd.Grouper(key='time', freq=sec_interval)).min().reset_index()
            elif how == 'sum':
                d = d.groupby(pd.Grouper(key='time', freq=sec_interval)).sum().reset_index()
            elif how == 'median':
                d = d.groupby(pd.Grouper(key='time', freq=sec_interval)).median().reset_index()
            else:
                print('how 字段超出了可处理的方式')
                return
            d['time'] = d['time'].dt.strftime('%H:%M:%S')
            d['PV'] = d['PV']*1000
            d.to_csv(f'./Data/PV_Price/{sec_interval}/{filename}',index=False)
        return d
        pass
    
    def make_PV_Price_avg(self,sec_interval,  output, address_output = None, how='mean'):
        PV_Price_avg = pd.DataFrame(columns = ['PV','PRICE'])
        cnt = 0
        t = 0
        for i in range(self.PV_Price.__len__()):
            if cnt == 0:
                PV_Price_avg.loc[PV_Price_avg.__len__(), :] = 0 # 创建一个新的行 
            PV_Price_avg.iloc[PV_Price_avg.__len__()-1,:] += self.PV_Price.loc[i,:]
            cnt += 1
            if cnt==sec_interval:
                cnt = 0
        # 结束之后，取平均
        PV_Price_avg = PV_Price_avg/sec_interval
        if output == True:
            PV_Price_avg.to_csv(address_output, index = False)
        return PV_Price_avg
        pass

    def make_load_avg(self, sec_interval,  address_output, how='mean'):
        load_avg = pd.DataFrame(columns=[f'N{i}' for i in range(self.params.nums_Tss)])
        cnt = 0
        t = 0
        for i in range(self.DFs.__len__()):
            if cnt==0:
                load_avg.loc[load_avg.__len__(),:] = 0 # 创建一个新的行
            load_avg.iloc[load_avg.__len__()-1,:] += self.DFs[i].loc[self.DFs[i]['class']==1,:]['load'].values

            cnt += 1
            if cnt == sec_interval:
                cnt =0
        # 结束之后，整体取平均
        load_avg = load_avg/sec_interval
        load_avg.to_csv(address_output, index = False)
        return load_avg
        pass

    def dayAhead_OPF_avg(self, schedule_start):
        self.offline_data_avg = dayAhead_OPF_avg(schedule_start=schedule_start
                                            ,schedule_horizon=self.load_avg.__len__()
                                            ,load_avg=self.load_avg
                                            ,PV_Price=self.PV_Price_avg
                                            ,params = self.params
                                            ,plot=False
                                            ,in_PV=True
                                            )
        return self.offline_data_avg

    # def dayAhead_OPF_lin(self, schedule_start):
    #     self.offline_data_lin = dayAhead_OPF_lin(schedule_start=schedule_start
    #                                         ,schedule_horizon=self.params.offline_horizon
    #                                         ,DFs=self.DFs
    #                                         ,DFs_noStop=self.DFs_noStop
    #                                         ,PV_Price=self.PV_Price
    #                                         ,Ys=self.Ys
    #                                         ,params = self.params
    #                                         ,plot=False
    #                                         )
    #     pass

    def dayAhead_OPF_org(self, schedule_start, in_PV):
        self.offline_data_org = dayAhead_OPF_org(schedule_start=schedule_start
                                            ,schedule_horizon=self.params.offline_horizon
                                            ,DFs=self.DFs
                                            ,DFs_noStop=self.DFs_noStop
                                            ,PV_Price=self.PV_Price
                                            ,in_PV=in_PV
                                            ,Ys=self.Ys
                                            ,params = self.params
                                            ,plot=False
                                            )
        return self.offline_data_org


    def offline_OPF_org(self, schedule_start, in_PV):
        self.offline_data_org = offline_OPF_org(schedule_start=schedule_start
                                            ,schedule_horizon=self.params.offline_horizon
                                            ,DFs=self.DFs
                                            ,DFs_noStop=self.DFs_noStop
                                            ,PV_Price=self.PV_Price
                                            ,in_PV = in_PV
                                            ,Ys=self.Ys
                                            ,params = self.params
                                            ,plot=False
                                            )
        return self.offline_data_org

    def offline_OPF_lin(self, schedule_start, in_PV):
        self.offline_data_lin = offline_OPF_lin(schedule_start=schedule_start
                                            ,schedule_horizon=self.params.offline_horizon
                                            ,DFs=self.DFs
                                            ,DFs_noStop=self.DFs_noStop
                                            ,PV_Price=self.PV_Price
                                            ,in_PV=in_PV
                                            ,Ys=self.Ys
                                            ,params = self.params
                                            ,plot=False
                                            )
        return self.offline_data_lin
    

    def online_OPF_org(self, schedule_start, in_PV):
        self.online_data_org = realTime_OPF_org(schedule_start=schedule_start
                                            ,schedule_horizon=self.params.offline_horizon
                                            ,realTime_horizon = self.params.online_horizon
                                            ,DFs=self.DFs
                                            ,DFs_noStop=self.DFs_noStop
                                            ,PV_Price=self.PV_Price
                                            ,in_PV=in_PV
                                            ,Ys=self.Ys
                                            ,SOCs = self.offline_data_org['SOCs']
                                            ,params = self.params
                                            ,plot=False
                                            )
        return self.online_data_org

    def online_OPF_lin(self, schedule_start, in_PV):
        self.online_data_lin = realTime_OPF_lin(schedule_start=schedule_start
                                            ,schedule_horizon=self.params.offline_horizon
                                            ,realTime_horizon = self.params.online_horizon
                                            ,DFs=self.DFs
                                            ,DFs_noStop=self.DFs_noStop
                                            ,PV_Price=self.PV_Price
                                            ,in_PV=in_PV
                                            ,Ys=self.Ys
                                            ,SOCs = self.offline_data_lin['SOCs']
                                            ,params = self.params
                                            ,plot=False
                                            )
        return self.online_data_lin

    
    def PowerFLow_org(self, schedule_start, in_PV):
        self.powerFlow_data_org = power_flow_org(power_flow_start = schedule_start
                                    ,power_flow_horizon= self.params.offline_horizon
                                    ,DFs = self.DFs
                                    ,DFs_noStop = self.DFs_noStop
                                    ,PV_Price = self.PV_Price
                                    ,in_PV=in_PV
                                    ,params=self.params
                                    ,Ys = self.Ys)
        return self.powerFlow_data_org
    
    def PowerFLow_lin(self, schedule_start, in_PV):
        self.powerFlow_data_lin = power_flow_lin(power_flow_start = schedule_start
                                    ,power_flow_horizon= self.params.offline_horizon
                                    ,DFs = self.DFs
                                    ,DFs_noStop = self.DFs_noStop
                                    ,PV_Price = self.PV_Price
                                    ,in_PV=in_PV
                                    ,params=self.params
                                    ,Ys = self.Ys)
        return self.powerFlow_data_lin
    
    def PowerFlow_onlineDisp(self, schedule_start, dispatch_data, in_PV, PV_Price):
        self.PF_data = power_flow_onlineDisp(online_data = dispatch_data
                            ,params=self.params
                            ,power_flow_start = schedule_start
                            ,power_flow_horizon= len(dispatch_data['p_gs'])
                            ,DFs = self.DFs
                            ,DFs_noStop = self.DFs_noStop
                            ,PV_Price = PV_Price
                            ,in_PV=in_PV
                            ,Ys = self.Ys
                            ,plot = False)
        return self.PF_data
    
    def Cost_compute(self, schedule_start, p_ins, delta_t = 1):
        pIn_NoNeg = p_ins.astype('float').reset_index(drop=True)
        for col in pIn_NoNeg.columns:
            pIn_NoNeg[col].map(lambda x:max(x,0))
        # pIn_NoNeg = p_ins.sum(axis=1).map(lambda x:max(x,0)).reset_index(drop=True)
        df_cost = delta_t * self.PV_Price['PRICE'].reset_index(drop=True) * pIn_NoNeg.sum(axis = 1)
        cost = df_cost.sum()
        return cost, df_cost
