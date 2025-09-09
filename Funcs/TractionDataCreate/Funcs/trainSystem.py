import pandas as pd
import numpy as np
import random
from Funcs.train import Train

def TrainSystem(up_locations, down_locations, num_tss=None,stop_time=10, rdm_wait = False, max_wait_time = 30, initial_wait_time = None, traction_system=None):
    '''
    输入 上行,下行的位置, 以及牵引站的数量(用于编号), 所有的列车从牵引站的后一个正整数开始变好
    返回 上下行路线的 两个list, 其中元素为 class train 
    '''
    num_up = len(up_locations)
    num_down = len(down_locations)
    up_locations.sort()
    down_locations.sort(reverse = True) # 下行的 locations 要反向排序, 以保证所有列车顺时针是递增的 例如4,5,6,...
    up_names = np.arange(num_tss+1, num_tss+1+num_up)
    down_names = np.arange(num_tss+1+num_up, num_tss+1+num_up+num_down)
    trainUp_list = list()
    trainDown_list = list()
    # 选择随机出发
    
    if rdm_wait==True:
        for i in range(num_up):
            trainUp_list.append(Train(up_names[i], up_locations[i], direction=1, stop_time=stop_time, initial_wait_time=random.randint(0,max_wait_time), traction_system = traction_system))
        for i in range(num_down):
            trainDown_list.append(Train(down_names[i], down_locations[i], direction=-1, stop_time=stop_time, initial_wait_time=random.randint(0,max_wait_time), traction_system = traction_system))
    # 同时出发
    elif initial_wait_time != None:
        for i in range(num_up):
            trainUp_list.append(Train(up_names[i], up_locations[i], direction=1, stop_time=stop_time, initial_wait_time=initial_wait_time[i], traction_system = traction_system))
        for i in range(num_down):
            trainDown_list.append(Train(down_names[i], down_locations[i], direction=-1, stop_time=stop_time, initial_wait_time=initial_wait_time[num_up+i], traction_system = traction_system))
    else:
        for i in range(num_up):
            trainUp_list.append(Train(up_names[i], up_locations[i], direction=1, stop_time=stop_time, traction_system = traction_system))
        for i in range(num_down):
            trainDown_list.append(Train(down_names[i], down_locations[i], direction=-1, stop_time=stop_time))
    # print('上行列车的个数是{}\n上行列车的编号分别是{}:\n 上行列车的位置分别是:{}\n'.format(num_up, up_names, up_locations))
    # print('下行列车的个数是{}\n下行列车的编号分别是{}:\n 下行列车的位置分别是:{}\n'.format(num_down, down_names, down_locations))

    return trainUp_list, trainDown_list    