#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   diff_p_ins.py
@Time    :   2025/02/18 22:11:50
@Author  :   勾子瑭
'''

# here put the import lib
import numpy as np
import matplotlib.pyplot as plt

def diff_p_ins(online_p_ins, offline_p_ins,simulation_horizon,horizon):
    '''
    得各个牵引站输出功率之和,并且输出online,offline,diff 并且画出 bar 图
    '''
    # print('offline 各牵引网出力:\n')
    offline_sum_p_ins = offline_p_ins.iloc[:simulation_horizon-horizon,:].astype('float').sum(axis=0)
    # display(offline_sum_p_ins)
    # print('online 各牵引网出力:\n')
    online_sum_p_ins = online_p_ins.iloc[:simulation_horizon-horizon,:].astype('float').sum(axis=0)
    # display(online_sum_p_ins)
    diff_sum_p_ins = online_p_ins.iloc[:simulation_horizon-horizon,:].astype('float').sum(axis=0) - offline_p_ins.iloc[:simulation_horizon-horizon,:].astype('float').sum(axis=0)
    # for i in diff_sum_p_ins.index:
    #     print('online 对于 offline 的偏差为:{}%\n'.format((diff_sum_p_ins[i]/offline_sum_p_ins[i])*100))

    # 确保两个Series的索引相同
    if not online_sum_p_ins.index.equals(offline_sum_p_ins.index):
        raise ValueError("两个Series的索引必须相同")
    
    # 计算差异
    difference = online_sum_p_ins - offline_sum_p_ins
    
    # 设置条形图的宽度
    bar_width = 0.35
    
    # 设置索引位置
    index = np.arange(len(online_sum_p_ins))
    
    # 创建图形和轴
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(5)
    
    # 绘制第一个Series的条形图
    ax.bar(index - bar_width/2, online_sum_p_ins, bar_width, label='online_sum_p_ins')
    
    # 绘制第二个Series的条形图
    ax.bar(index + bar_width/2, offline_sum_p_ins, bar_width, label='offline_sum_p_ins')
    
    # 绘制差异的条形图
    ax.bar(index, difference, bar_width, color='grey', alpha=0.8, label='Difference')
    
    # 设置x轴刻度标签
    ax.set_xticks(index)
    # xlabels = list()
    # for i in online_sum_p_ins.index.values:
    #     xlabels.append('P_sum_'+str(i[14:]))
    # ax.set_xticklabels(xlabels)
    
    # 添加图例
    ax.legend()
    
    # 设置标题和标签
    ax.set_title('Comparison of P_ins of on/offline_sum_p_ins with Difference')
    ax.set_xlabel('Tss')
    ax.set_ylabel('P_ins')
    
    # 显示图形
    plt.show()

    return online_sum_p_ins,offline_sum_p_ins,diff_sum_p_ins