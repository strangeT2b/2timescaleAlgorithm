#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   plot_horizon_volt.py
@Time    :   2025/02/18 22:13:15
@Author  :   勾子瑭
'''

# here put the import lib

import matplotlib.pyplot as plt

def plot_once_p_in(p_ins, DFs, h, num_tss):
    '''
    用于画出某一时刻所有点的功率，并且按照 location 进行排序， 标记出 TSS 和 Train
    '''
    DF_copy = DFs[h]
    p_in_p = p_ins.iloc[h,:].astype('float')
    DF_copy.loc[:num_tss-1,'P'] = p_in_p.values
    p_all = DF_copy[['name','class', 'P','upPre','upPost','downPre','downPost','location']]
    # p_all.loc[:,'name'] = p_all.loc[:,'name'].astype('str')
    def modify_name(name):
        # 提取数字部分
        num = int(name)  # 假设 'Volt_N' 后面的数字为目标
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
    
    print('牵引站有: {}\n上行车辆有：{} \n下行车辆有{}'.format(p_in_tss['name'].values,train_up['name'].values,train_down['name'].values))

    plt.figure(figsize=(22,7))
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