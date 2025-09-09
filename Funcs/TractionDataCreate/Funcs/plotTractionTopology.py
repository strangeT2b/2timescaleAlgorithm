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
    ani.save('traction_topology_animation.mp4', writer='ffmpeg', fps=30, dpi=300)
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
    fig, ax = plt.subplots(figsize=(60, 3))
    
    # 预定义y轴位置和样式
    y_up = 2.6    # 上行轨道位置
    y_tss = 2.0   # TSS位置
    y_down = 1.4  # 下行轨道位置
    
    colors = {
        'Tss': 'royalblue',
        'downTrain': 'coral',
        'upTrain': 'orchid'
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

    