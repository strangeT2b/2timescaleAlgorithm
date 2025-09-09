#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   plot_horizon_volt.py
@Time    :   2025/02/18 22:13:15
@Author  :   勾子瑭
'''

# here put the import lib
"""
用来绘制一段时间内，所有节点的电压变化
"""
import matplotlib.pyplot as plt

def plot_horizon_volt(vs,start,end,num_tss=11):

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
    vs.columns = [transform_column_name(col) for col in vs.columns]

    fig,axes = plt.subplots(2,1,figsize=(20,10))
    vs.iloc[start:end,:num_tss].plot(ax=axes[0]
                                ,xlim=[-50,end-start+300]
                                ,ylim=[1.5,1.68]
                                # ,xticks = np.linspace(start,end,12)
                                ,title='voltage of Tss'
                                ,xlabel='time/s'
                                ,ylabel='voltage/kv')
    vs.iloc[start:end,num_tss:].plot(ax=axes[1]
                                ,xlim=[-50,end-start+300]
                                ,ylim=[1.5,1.68]
                                # ,xticks = np.linspace(start,end,12)
                                ,title='voltage of Train'
                                ,xlabel='time/s'
                                ,ylabel='voltage/kv')

    plt.tight_layout()
    plt.show()