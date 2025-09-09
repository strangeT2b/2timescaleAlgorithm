#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   plot_horizon_volt.py
@Time    :   2025/02/18 22:13:15
@Author  :   
'''

# here put the import lib

import matplotlib.pyplot as plt

def plot_once_p_in(p_ins, p_gs, DFs, h, num_tss):
    '''
    用于画出某一时刻所有点的功率，并且按照 location 进行排序， 标记出 TSS 和 Train
    还需要把 光伏的电画上去
    '''
    DF_copy = DFs[h]
    p_in_p = p_ins.iloc[h,:].astype('float')
    DF_copy.loc[:num_tss-1,'P'] = p_in_p.values
    p_all = DF_copy[['name','class', 'P','upPre','upPost','downPre','downPost','location']]
    # p_all.loc[:,'name'] = p_all.loc[:,'name'].astype('str')
    def modify_name(name):
        # 提取数字部分
        num = int(name) 
        if num <= num_tss:
            return 'Tss_{}'.format(name)
        else:
            return 'Train_{}'.format(name)


    # 应用函数修改 'name' 列
    p_all.loc[:,'name'] = p_all.loc[:,'name'].apply(modify_name)
    p_all = p_all.sort_values('location')

    p_all.reset_index(drop=True,inplace=True)
    
    tss = p_all[p_all['class']==1]
    p_in_tss = p_all[p_all['class']==1]

    train_up = p_all.loc[(p_all['class']==0) & (p_all['downPre']==0) & (p_all['downPost']==0),:]
    train_up = train_up[train_up.isna() == False]
    p_train_up = p_all.iloc[train_up.index.values,:]
    train_down = p_all[(p_all['class']==0) & (p_all['upPre']==0) & (p_all['upPost']==0)]
    train_down = train_down[train_down.isna() == False]
    p_train_down = p_all.iloc[train_down.index.values,:]
    
    # print('牵引站有: {}\n上行车辆有：{} \n下行车辆有{}'.format(p_in_tss['name'].values,train_up['name'].values,train_down['name'].values))

    plt.figure(figsize=(15,3))
    plt.plot(p_all['name'], p_all['P'])

    plt.scatter(p_in_tss['name'], p_in_tss['P']
                ,marker='o'
                ,color = 'blue')

    plt.scatter(p_train_up['name'], p_train_up['P']
    ,marker = 'o'
    ,color = 'darkorange')
    plt.scatter(p_train_down['name'], p_train_down['P']
    ,marker = 'o'
    ,color = 'deeppink')
    plt.xlim([-1,len(p_all['name'])+1])
    plt.xlabel('Node')
    plt.ylabel('Power/MW')
    plt.legend(['P','Tss','upTrain','downTrain'],loc='upper right')
    plt.title('Power of Nodes')
    plt.grid()
    plt.xticks()
    plt.tight_layout()
    plt.show()


#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   plot_horizon_volt.py
@Time    :   2025/02/18 22:13:15
@Author  :   勾子瑭
'''

# here put the import lib

import matplotlib.pyplot as plt

def plot_once_volt(vs, DFs, h, num_tss):
    '''
    用于画出某一时刻所有点的电压，并且按照 location 进行排序， 标记出 TSS 和 Train
    '''
    vs_p = vs.copy()
    # 更新索引 换成 tss 和 train
    def transform_column_name(col_name):
        if col_name.startswith('Volt_N'):
            # 提取数字部分
            num = int(col_name.split('_N')[1])
            if num <= num_tss:
                return col_name.replace('Volt', 'TSS')
            else:
                return col_name.replace('Volt', 'Train')
        return col_name  # 如果不是 Volt_N 格式，保持不变
    
    # 更新列名
    
    vs_p.columns = [transform_column_name(col) for col in vs_p.columns]
    df_p = vs_p.iloc[h,DFs[h].sort_values('location').index.values].astype('float')

    tss = DFs[h][(DFs[h]['class']==1)]
    vs_tss = vs_p.iloc[h,tss.index.values].astype('float')
    vs_tss = vs_tss[vs_tss.isna()==False]

    train_up = DFs[h][(DFs[h]['class']==0) & (DFs[h]['downPre']==0) & (DFs[h]['downPost']==0)]
    vs_up = vs_p.iloc[h,train_up.index.values].astype('float')
    vs_up = vs_up[vs_up.isna()==False]
    train_up = train_up[train_up.isna() == False]

    train_down = DFs[h][(DFs[h]['class']==0) & (DFs[h]['upPre']==0) & (DFs[h]['upPost']==0)]
    vs_down = vs_p.iloc[h,train_down.index.values].astype('float')
    vs_down = vs_down[vs_down.isna()==False]
    train_down = train_down[train_down.isna() == False]

    # print('牵引站有: {}\n上行列车有: {}\n下行列车有: {}'.format(tss['name'].values, train_up['name'].values, train_down['name'].values))

    plt.figure(figsize=(15,3))
    df_p = df_p[df_p.isna() == False]
    plt.plot(df_p.index, df_p.values)

    plt.scatter(vs_tss.index, vs_tss.values
                ,marker='o'
                ,color = 'blue')

    plt.scatter(vs_up.index, vs_up.values
    ,marker = 'o'
    ,color = 'darkorange')
    plt.scatter(vs_down.index, vs_down.values
    ,marker = 'o'
    ,color = 'deeppink')
    plt.xlim([-1,len(df_p)+1])
    plt.xlabel('Node')
    plt.ylabel('Voltage/kv')
    plt.ticklabel_format(axis='y', style='plain', useOffset=False)
    plt.legend(['volt','Tss','upTrain','downTrain'],loc='upper right')
    plt.title('Voltage of Nodes')
    plt.grid(True)
    plt.xticks()
    plt.tight_layout()
    plt.show()


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_dataframe_animation(dataframes, min_location, max_location):
    # 设置画布
    fig, ax = plt.subplots()
    fig.set_figheight(3)
    fig.set_figwidth(60)
    
    # 更新函数
    def update(frame):
        ax.clear()  # 清除当前的图形
        df = dataframes[frame]  # 当前帧的数据
        h = (frame+1)//60//60
        m = (frame+1)//60%60
        s = (frame+1)%60%60
        # 设置标题
        ax.set_title(f"{h} h {m} min {s} s")
        
        # 设置坐标轴范围
        ax.set_xlim(min_location-max_location*0.1, max_location*1.1)  # x轴范围，根据需要调整
        y_up = 2.6
        y_tss = 2
        y_down = 1.4
        ax.set_ylim(y_down-2, y_up+2) 

        ax.hlines(y=y_up,xmin=min_location,xmax=max_location,colors='royalblue',linestyles='--')
        ax.hlines(y=y_down,xmin=min_location,xmax=max_location,colors='royalblue',linestyles='--')

        # 创建空图例字典来存储每个类别的标签是否已绘制
        legend_dict = {'Tss': False, 'downTrain': False, 'upTrain': False}
        
        # 绘制数据
        for _, row in df.iterrows():
            x = row['location']
            if row['class'] == 1:
                y = y_tss  # 对于 class=1，y固定为2
                if not legend_dict['Tss']:  # 仅绘制一次图例
                    ax.scatter(x, y, color='royalblue', label='Tss', marker='s',linewidths=10)
                    ax.scatter(x, y_up, color='royalblue', marker='|',linewidths=3)
                    ax.scatter(x, y_down, color='royalblue', marker='|',linewidths=3)
                    legend_dict['Tss'] = True
                else:
                    ax.scatter(x, y, color='royalblue', marker='s',linewidths=10)
                    ax.scatter(x, y_up, color='royalblue', marker='|',linewidths=3)
                    ax.scatter(x, y_down, color='royalblue', marker='|',linewidths=3)
            elif row['class'] == 0:
                if (row['upPre'] == 0 and row['upPost'] == 0) and (row['downPre'] != 0 or row['downPost'] != 0):
                    y = y_down  # 对于 class=0，upPre和upPost都为0且downPre或downPost不为0，y固定为1
                    if not legend_dict['downTrain']:  # 仅绘制一次图例
                        ax.scatter(x, y, color='coral', label='downTrain', marker='<', linewidths=5)
                        legend_dict['downTrain'] = True
                    else:
                        ax.scatter(x, y, color='coral', marker='<', linewidths=5)
                elif row['downPre'] == 0 and row['downPost'] == 0 and (row['upPre'] != 0 or row['upPost'] != 0):
                    y = y_up  # 对于 class=0，downPre和downPost都为0且upPre或upPost不为0，y固定为3
                    if not legend_dict['upTrain']:  # 仅绘制一次图例
                        ax.scatter(x, y, color='orchid', label='upTrain', marker='>', linewidths=5)
                        legend_dict['upTrain'] = True
                    else:
                        ax.scatter(x, y, color='orchid', marker='>', linewidths=5)

        # 设置图例
        ax.legend()

    # 创建动画，并将其分配给变量ani
    ani = FuncAnimation(fig, update, frames=len(dataframes), repeat=False)

    # 显示动画
    plt.show()



import matplotlib.pyplot as plt
def plot_tracition_topology(df, min_location=0, max_location=20000):
    """绘制单个时刻的轨道拓扑结构
    
    参数:
        df (pd.DataFrame): 包含拓扑数据的DataFrame
        min_location (float): 最小位置坐标
        max_location (float): 最大位置坐标
    """
    # 设置画布
    fig, ax = plt.subplots(figsize=(15,3))
    
    # 预定义y轴位置和样式
    y_up = 2.6    # 上行轨道位置
    y_tss = 2.0   # TSS位置
    y_down = 1.4  # 下行轨道位置
    
    colors = {
        'Tss': 'royalblue',
        'downTrain': 'orchid',
        'upTrain': 'coral'
    }
    
    markers = {
        'Tss': 's',     # 正方形
        'downTrain': '<',  # 左三角
        'upTrain': '>'     # 右三角
    }
    
    # 设置坐标轴范围
    ax.set_xlim(min_location - max_location*0.1, max_location*1.1)
    ax.set_ylim(y_down - 0.5, y_up + 0.5)  # 缩小y轴范围
    
    # 绘制基准线
    ax.hlines(y=y_up, xmin=min_location, xmax=max_location, 
             colors=colors['Tss'], linestyles='--', linewidth=0.5)
    ax.hlines(y=y_down, xmin=min_location, xmax=max_location, 
             colors=colors['Tss'], linestyles='--', linewidth=0.5)
    
    # 绘制数据点
    for _, row in df.iterrows():
        x = row['location']
        
        if row['class'] == 1:  # TSS
            ax.scatter(x, y_tss, color=colors['Tss'], 
                      marker=markers['Tss'], s=100, label='Tss')
            # 上下标记线
            ax.scatter(x, y_up, color=colors['Tss'], marker='|', s=30)
            ax.scatter(x, y_down, color=colors['Tss'], marker='|', s=30)
            
        elif row['class'] == 0:  # 列车
            if (row['upPre'] == 0 and row['upPost'] == 0) and \
               (row['downPre'] != 0 or row['downPost'] != 0):
                # 下行列车
                ax.scatter(x, y_down, color=colors['downTrain'], 
                          marker=markers['downTrain'], s=80, label='downTrain')
            
            elif (row['downPre'] == 0 and row['downPost'] == 0) and \
                 (row['upPre'] != 0 or row['upPost'] != 0):
                # 上行列车
                ax.scatter(x, y_up, color=colors['upTrain'], 
                          marker=markers['upTrain'], s=80, label='upTrain')
    
    # 处理图例 - 避免重复
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper right')
    
    # 添加网格和标题
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_title("Railway Topology at Current Moment", fontsize=12)
    
    plt.tight_layout()
    plt.show()

    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_detail(datas, dfs, dfs_noStop, h, num_tss=11):
    df = dfs[h]
    vs = datas['vs']
    p_ins = datas['p_ins']
    plot_once_volt(vs, dfs, h, num_tss)
    plot_once_p_in(p_ins, dfs, h, num_tss)
    plot_tracition_topology(df)
    pass