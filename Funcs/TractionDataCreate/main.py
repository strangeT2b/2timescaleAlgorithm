
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

from Funcs.train import Train
from Funcs.tss import Tss
from Funcs.topology import topology
from Funcs.plot_animation import plot_dataframe_animation
from Funcs.trainSystem import TrainSystem
from Funcs.tractionSystem import TractionSystem

# 示例用法
import time
tic = time.time()
tss_num = 11
tss_locations = pd.read_csv("/home/gzt/Codes/2STAGE/Funcs/TractionDataCreate/dataset/tss_information/tss_information.csv")["location"].to_list()[:tss_num]
# tss_locations = [0,2000,4
# 000,6000,8000,10000,12000,14000,16000,18000,20000]
upTrain_locations = []
downTrain_locations = []

tss_stopping_times_up = pd.read_csv("/home/gzt/Codes/2STAGE/Funcs/TractionDataCreate/dataset/tss_information/stopping_second_up.csv").astype(int)['stopping_second_up'].to_list()[:tss_num]
tss_stopping_times_down = pd.read_csv("/home/gzt/Codes/2STAGE/Funcs/TractionDataCreate/dataset/tss_information/stopping_second_down.csv").astype(int)['stopping_second_down'].to_list()[:tss_num]
tss_list = TractionSystem(tss_locations, stopping_times_up=tss_stopping_times_up, stopping_times_down=tss_stopping_times_down)
start_interval = 15
intial_wait_time = list(range(0,(len(upTrain_locations)+len(downTrain_locations))*30, start_interval))
upTrain_list, downTrain_list = TrainSystem(upTrain_locations, downTrain_locations,num_tss=len(tss_list), stop_time=35, rdm_wait=False, max_wait_time=60, initial_wait_time = intial_wait_time, traction_system = tss_list)
df = topology(tss_list, upTrain_list, downTrain_list)


duration = 1 # 以 1 s 为单位
iter_num =24*60*60-1 # 迭代次数, 也就是运行多少秒

iter_dfs = list()
# iter_dfs.append(df)


for i in range(iter_num+1):
    iter_df = topology(tss_list, upTrain_list, downTrain_list)
    df = pd.concat([df,iter_df], axis=0)
    iter_dfs.append(iter_df)
    print('进度为 {} %\n'.format(i/iter_num*100))
    # print(upTrain_list)
    # print(downTrain_list)

    # 注意上行与下行的列表编号是反向的的
    if len(upTrain_list)>0 and upTrain_list[-1].location >= max(tss_locations):
        # downTrain_list.insert(0, upTrain_list[-1])
        if upTrain_list[-1].timer == upTrain_list[-1].stop_time:
            upTrain_list.pop(-1)
        # downTrain_list[0].location = max(tss_locations) - abs(max(tss_locations)-downTrain_list[0].location)
    if len(downTrain_list)>0 and downTrain_list[-1].location <= min(tss_locations):
        # upTrain_list.insert(0, downTrain_list[-1])
        if downTrain_list[-1].timer == downTrain_list[-1].stop_time:
            downTrain_list.pop(-1)
        # upTrain_list[0].location = min(tss_locations) + abs(min(tss_locations)-upTrain_list[0].location)
    for j in upTrain_list:
        j.train_update(iter_dfs[i])
        j.simulation(duration = 1, min_tss_location=min(tss_locations),max_tss_location=max(tss_locations))
    for j in downTrain_list:
        j.train_update(iter_dfs[i])
        j.simulation(duration = 1, min_tss_location=min(tss_locations),max_tss_location=max(tss_locations))
    
    # print(iter_dfs[i])
    # print('上行有 {} 辆车, 下行有 {} 辆车\n'.format(len(upTrain_list), len(downTrain_list)))
    # upTrain_list[1].show_train()
    # 上行路线发车
    # 跟铁科院的数据一致， 每隔 4分54秒收发站发车， 并且每座牵引站的停靠时间不同
    # if i%294 == 0 and len(downTrain_list)<10 and len(upTrain_list) <10 and i!= 0:
    up_sec = int(np.random.normal(loc=294, scale=7))
    if i%294 == 0:
        next_name = max(x.name for x in upTrain_list+downTrain_list)+1 if upTrain_list+downTrain_list else tss_num+1
        for name in range(12,next_name):
            if name not in list(x.name for x in upTrain_list+downTrain_list):
                next_name = name
                break
        print("插入上行列车: {}".format(next_name))
        upTrain_list.insert(0, Train(next_name, min(tss_locations), direction=1, stop_time=35, initial_wait_time=0, traction_system=tss_list))
        up_sec = int(np.random.normal(loc=294, scale=7))
        # print(upTrain_list)
    # print(len(upTrain_list))
    # print('iter_num = {}'.format(i))
    
    # 下行路线发车
    # 跟铁科院的数据一致， 每隔 4分54秒收发站发车， 并且每座牵引站的停靠时间不同
    # if i%294 == 0 and len(downTrain_list)<10 and len(upTrain_list) <10 and i!= 0:
    down_sec = int(np.random.normal(loc=294, scale=7))
    if i%294 == 0:
        next_name = max(x.name for x in upTrain_list+downTrain_list)+1 if upTrain_list+downTrain_list else tss_num+1
        for name in range(12,next_name):
            if name not in list(x.name for x in upTrain_list+downTrain_list):
                next_name = name
                break
        print("插入下行列车: {}".format(next_name))
        downTrain_list.insert(0, Train(next_name, max(tss_locations), direction=-1, stop_time=35, initial_wait_time=0, traction_system=tss_list))
        down_sec = int(np.random.normal(loc=294, scale=7))
        # print(downTrain_list)

    


df.loc[df['P']==0, 'P']=0.4
df.loc[df['class']==1, 'P']=0
df['E'] = 0
df.loc[df['class']==1, 'E'] = 1593
df['Rs'] = 0
df.loc[df['class']==1, 'Rs'] = 0.02
df['V'] = 1500
# df = df.loc[:,['name', 'class','upPre','upPost','downPre','downPost','preTss','distance_preTss','postTss','distance_postTss','E','Rs','V','P', 'stopping', 'upStop', 'downStop', 'location']]
for i in ['distance_preTss','distance_postTss','distance_upPre','distance_upPost','distance_downPre','distance_downPost']:
    df.loc[df[i]!=-1,i] = df.loc[df[i]!=-1,i]/1000
# df['P'] = df['P']*1000000
df["E"] = df["E"] / 1000  # 转换为 kv

one_length = len(iter_df)
print(df)
print('一次采样共 {} 行\n'.format(one_length))

df.to_csv('/home/gzt/Codes/2STAGE/Funcs/TractionDataCreate/dataset/oneDay_biStart_4m42s_noised_dropLocation.csv',index=False)

df.drop(columns=['location'],inplace=True)
df.to_csv('/home/gzt/Codes/2STAGE/Funcs/TractionDataCreate/dataset/oneDay_biStart_4m42s_noised.csv',index=False)

toc = time.time()
print('耗时: {}s\n'.format(toc-tic))

traction_length = max(tss_locations)-min(tss_locations)
plot_dataframe_animation(iter_dfs
                         , min_location=min(tss_locations)
                         , max_location=max(tss_locations))




