import numpy as np
from Funcs.tss import Tss

def TractionSystem(locations, stopping_times_up, stopping_times_down, names = None):
    if len(locations) != len(stopping_times_up) or len(stopping_times_up)!= len(stopping_times_down):
        raise ValueError("位置列表长度与停靠时间列表不符")
    
    num_tss = len(locations)
    locations.sort()
    tss_list = list()
    if names == None:
        names = np.arange(1, num_tss+1) # 如果没有输入 names 参数的话, 直接用 1,2,... 来为牵引站编号
    # print('牵引站的个数是{}\n牵引站的编号分别是{}:\n 牵引站的位置分别是:{}\n'.format(num_tss, names, locations))
    for i in range(num_tss):
        preTss = i-1 
        postTss = i+1 if i < num_tss-1 else -1
        tss_list.append(Tss(names[i], locations[i], stopping_time_up=stopping_times_up[i], stopping_time_down=stopping_times_down[i]))
    return tss_list