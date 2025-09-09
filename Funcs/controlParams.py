#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   controlPrams.py
@Time    :   2025/02/18 21:41:40
@Author  :   勾子瑭
'''

import math
# here put the import lib
class Params():
    def __init__(self):

        self.offline_horizon = 2*60*60
        self.online_horizon = 30

        self.battery_efficiency = 0.9

        self.time_interval = 1
        self.delta_t = 1

        self.nums_Tss = 11

        self.stoppingP = 0.4

        self.start_time = '00:00:00'
        
        self.Rs = 0.02
        self.r_r = 0.02
        self.r_c = 0.03
        self.g = 0.01

        self.E = 1.593
        self.E_train = 1.585
        
        self.p_b_min = -20
        self.p_b_max = 20
        self.p_ch_min = 0
        self.p_ch_max = self.p_b_max
        self.p_disch_min = self.p_b_min
        self.p_disch_max = 0

        self.battery_max_kwh = 222
        self.battery_min = 0
        self.battery_max = self.battery_max_kwh*3.6

        self.soc_min = 0
        self.soc_max = 100

        self.p_g_min = -15
        self.p_g_max = 15

        self.p_in_min = 0
        self.p_in_max = 30

        self.j_in_min = 0
        self.j_in_max = 20

        self.v_tss_min = 1.2
        self.v_tss_max = self.E
        
        self.v_c_max = 1.85
        self.v_c_min = 1.2

        self.v_r_max = 0.5
        self.v_r_min = 0

        self.v_train_min = 1.2
        self.v_train_max = 1.75

        self.p_brake_min = 0
        self.p_brake_max = 0

        self.grb_gap = 0.05

        self.realTime_weight_p_in = 5
        self.realTime_weight_track = 60
        self.realTime_weight_brake = 0
        self.realTime_weight_L1 = 7

        # self.address_DF = './Data/DFs/2Hour_delay4m_I_LD.csv'
        # self.address_DF = './Data/DFs/22Hour_delay4m_I_LD.csv'
        self.address_DF = '/home/gzt/Codes/2STAGE/Funcs/TractionDataCreate/dataset/oneDay_biStart_4m42s.csv'

        # self.address_PV_Price = './Data/PV_Price/12s/PV_Price_12s_2023-04-26.csv'
        self.address_PV_Price = './Data/PV_Price/PV_Price_Second.csv'

        self.address_Ys = './Data/Ys/admittance_18h_delay4m.npy'

        self.address_PVs_folder = "./Data/PVs"
        
        self.offlineLinearGpParams = {"OutputFlag": 1
                                     ,"MIPGap": 0.01
                                     ,"TimeLimit": 4800}

        self.offlineNonlinearGpParams = {"OutputFlag": 1
                                        ,"MIPGap": 0.05
                                        ,"TimeLimit": 4800}
        
        self.onlineLinearGpParams = {"OutputFlag": 0
                                    ,"MIPGap": 0.05
                                    ,"TimeLimit": 0.8}
        
        self.onlineNonlinearGpParams = {"OutputFlag": 0
                                        ,"NonConvex": 2
                                        ,"MIPGap": 0.1
                                        ,"TimeLimit": 1}


        # self.offlineLinearGpParams = {"OutputFlag": 1
        #                             ,"ScaleFlag": 3
        #                             ,"PreSolve": 1
        #                             ,"OptimalityTol": 1e-4
        #                             ,"MIPGap": 0.01
        #                             ,"TimeLimit": 4800
        #                             ,"Threads": 10
        #                             ,"FeasibilityTol": 1e-4
        #                             ,'NumericFocus': 3          # 最高数值稳定性检查
        #                             ,"BarConvTol": 1e-4}

        # self.offlineNonlinearGpParams = {"OutputFlag": 1
        #                                 ,"ScaleFlag": 3
        #                                 ,"PreSolve": 1
        #                                 ,"NonConvex": 2
        #                                 ,"OptimalityTol": 1e-4
        #                                 ,'NumericFocus': 3          # 最高数值稳定性检查
        #                                 ,'BarCorrectors': 50       # 增加屏障法校正器
        #                                 ,'BarHomogeneous': 1       # 启用同质算法
        #                                 ,'Method': 2               # 强制使用内点法
        #                                 ,'Quad': 1                 # 优先处理二次项
        #                                 ,"MIPGap": 0.05
        #                                 ,"TimeLimit": 4800
        #                                 ,"Threads": 10
        #                                 ,"FeasibilityTol": 1e-4
        #                                 ,"BarConvTol": 1e-4}
        
        # self.onlineLinearGpParams = {"OutputFlag": 0
        #                             ,"ScaleFlag": 3
        #                             ,"PreSolve": 2
        #                             ,"OptimalityTol": 0.001
        #                             ,"MIPGap": 0.01
        #                             ,"TimeLimit": 0.8
        #                             ,"Threads": 10
        #                             ,"FeasibilityTol": 0.001
        #                             ,"NumericFocus": 3
        #                             ,"BarConvTol": 0.001}
        
        # self.onlineNonlinearGpParams = {"OutputFlag": 0
        #                                 ,"ScaleFlag": 3
        #                                 ,"PreSolve": 2
        #                                 ,"NonConvex": 2
        #                                 ,"OptimalityTol": 0.001
        #                                 ,"MIPGap": 0.1
        #                                 ,"TimeLimit": 1
        #                                 ,"Threads": 10
        #                                 ,'BarCorrectors': 50       # 增加屏障法校正器
        #                                 ,'BarHomogeneous': 1       # 启用同质算法
        #                                 ,"FeasibilityTol": 0.001
        #                                 ,"NumericFocus": 3
        #                                 ,"BarConvTol": 0.001}
        pass

    def tightByPercent(self, percent):
        self.E = self.E
        self.E_train = self.E_train
        
        p_b_diff = (self.p_b_max-self.p_b_min)
        self.p_b_min += p_b_diff*percent
        self.p_b_min = round(self.p_b_min, 4)
        self.p_b_max -= p_b_diff*percent
        self.p_b_max = round(self.p_b_max, 4)
        p_ch_diff = (self.p_ch_max-self.p_ch_min)
        self.p_ch_min = 1e-4
        self.p_ch_max -= p_ch_diff*percent
        self.p_ch_max = round(self.p_ch_max, 4)
        p_disch_diff = (self.p_disch_max-self.p_disch_min)
        self.p_disch_min += p_disch_diff*percent
        self.p_disch_min = round(self.p_disch_min, 4)
        self.p_disch_max = -1e-4

        self.battery_min = self.battery_min
        self.battery_max = self.battery_max

        soc_diff = (self.soc_max-self.soc_min)
        self.soc_min += soc_diff*percent
        self.soc_min = round(self.soc_min)
        self.soc_max -= soc_diff*percent
        self.soc_max = round(self.soc_max)


        p_g_diff = (self.p_g_max-self.p_g_min)
        self.p_g_min += p_g_diff*percent
        self.p_g_min = round(self.p_g_min, 4)
        self.p_g_max -= p_g_diff*percent
        self.p_g_max = round(self.p_g_max)

        p_in_diff = (self.p_in_max-self.p_in_min)
        self.p_in_min = 1e-4
        self.p_in_max -= p_in_diff*percent
        self.p_in_max = round(self.p_in_max)

        j_in_diff = (self.j_in_max-self.j_in_min)
        self.j_in_min = 1e-4
        self.j_in_max -= j_in_diff*percent
        self.j_in_max = round(self.j_in_max)

        v_tss_diff = (self.v_tss_max-self.v_tss_min)
        self.v_tss_min += v_tss_diff*percent
        self.v_tss_min = round(self.v_tss_min, 4)
        self.v_tss_max = 1.590

        v_train_diff = (self.v_train_max-self.v_train_min)
        self.v_train_min += v_train_diff*percent
        self.v_train_min = round(self.v_train_min, 4)
        self.v_train_max -= v_train_diff*percent
        self.v_train_max = round(self.v_train_max, 4)

        p_brake_diff = (self.p_brake_max-self.p_brake_min)
        self.p_brake_min += p_brake_diff*percent
        self.p_brake_max -= p_brake_diff*percent
