import pandas as pd
import numpy as np
import random

def topology(tss_list, trainUp_list, trainDown_list):
    # 收集列表中的信息
    tss_names = list()
    tss_locations = list()
    up_names = list()
    up_locations = list()
    down_names = list()
    down_locations = list()
    tss_Ps = list()
    up_Ps = list()
    down_Ps = list()
    for i,j in enumerate(tss_list):
        tss_locations.append(j.location)
        tss_names.append(j.name)
        tss_Ps.append(j.P)
    for i,j in enumerate(trainUp_list):
        up_locations.append(j.location)
        up_names.append(j.name)
        up_Ps.append(j.P)
    for i,j in enumerate(trainDown_list):
        down_locations.append(j.location)
        down_names.append(j.name)
        down_Ps.append(j.P)
    
    #, 'preTss', 'distance_preTss', 'postTss', 'distance_postTss'
    up_df = pd.DataFrame(columns=['name', 'location', 'class', 'upPre', 'upPost', 'distance_preTss', 'distance_postTss', 'P','stopping', 'upStop'])
    down_df = pd.DataFrame(columns=['name', 'location', 'class', 'downPre', 'downPost', 'distance_preTss', 'distance_postTss', 'P','stopping', 'downStop'])

    # # 将 上行、下行 分别与牵引站做成一个 df
    up_df['name'] = tss_names+up_names
    up_df['location'] =tss_locations+up_locations
    up_df['P'] = tss_Ps + up_Ps
    up_tss_idx = np.arange(len(tss_names)) # 先标记 tss 的类型
    up_df['class'] =0 # 0 代表机车
    up_df.loc[up_tss_idx, 'class'] = 1 # 1 代表牵引站


    down_df['name'] = tss_names+down_names
    down_df['location'] = tss_locations+down_locations
    down_df['P'] = tss_Ps + down_Ps
    down_tss_idx = np.arange(len(tss_names)) # 先标记 tss 的类型
    down_df['class'] =0 # 0 代表机车
    down_df.loc[down_tss_idx, 'class'] = 1 # 1 

    # 定义一个牵引站 阶梯函数: 节点 ——> 上一牵引站, 下一牵引站
    bins = tss_locations # 利用 牵引站的位置 作为分箱水平线
    up_piecewise = np.digitize(up_df['location'], bins=bins, right=True) #左闭右开, 当 <= x < , 好像无所谓?? 因为 当机车 到达牵引站时, 功率为零, 不参与计算
    upPreTss_name = [tss_names[i-1] if i-1 >= 0 else 0 for i in up_piecewise]
    upPostTss_name = [tss_names[i] if i < len(bins) else 0 for i in up_piecewise]

    down_piecewise = np.digitize(down_df['location'], bins=bins, right=True) #左开右闭, 当 < x <= , 好像无所谓?? 因为 当机车 到达牵引站时, 功率为零, 不参与计算
    downPreTss_name = [tss_names[i-1] if i-1 >= 0 else 0 for i in down_piecewise]
    downPostTss_name = [tss_names[i] if i < len(bins) else 0 for i in down_piecewise]

    # 加上两列牵引站的相对位置信息
    up_df['preTss'] = upPreTss_name
    up_df['postTss'] = upPostTss_name
    down_df['preTss'] = downPreTss_name
    down_df['postTss'] = downPostTss_name

    # 按照 location 排列, 以方便获得节点的相对位置
    up_df.sort_values(by = ['location', 'class'], inplace=True)
    down_df.sort_values(by = ['location', 'class'], inplace=True)
    # 将 index 初始化
    up_df.reset_index(drop=True, inplace=True) # 重置索引, 用来寻找相邻的点
    down_df.reset_index(drop=True, inplace=True)

    # 利用 df 的信息得到每个 Node 的相邻节点信息, 以及与相邻牵引站的距离信息
    for i in up_df.index:
 
        if up_df.loc[i,'class']==1: # TSS
            if up_df.loc[i,'location'] in up_locations:
                upstop = up_df.loc[(up_df['class']==0)&(up_df['location']==up_df.loc[i,'location']), 'name']
                # print(upstop.values)
                up_df.loc[i, 'upStop'] = upstop.values[0]
        elif up_df.loc[i,'class']==0: # Train
            if up_df.loc[i,'location'] in tss_locations:
                stoptss =up_df.loc[(up_df['class']==1)&(up_df['location']==up_df.loc[i,'location']), 'name']
                up_df.loc[i, 'stopping'] = stoptss.values[0]

        # 获得拓扑信息
        if i==0: # 如果是第一个元素的话, 那么他的上一个节点是空的
            up_df.loc[i,'upPre'] = 0 # 0 代表没有节点
            if up_df.loc[i,'location'] == up_df.loc[i+1,'location']: # 如果下一个节点的location与此节点相等, 那么是重合节点, 需要再向后移动
                if i+1 == up_df.__len__()-1: # 如果重合节点是最后一个元素的话, 那么他的下一个节点是空的
                    up_df.loc[i,'upPost'] = 0
                else:  # 如果重合节点不是最后一个节点, 那么只需要在往下移动一个节点
                    up_df.loc[i,'upPost'] = up_df.loc[i+2,'name'] # 下一个节点
            else: # 如果没有重合的话
                up_df.loc[i,'upPost'] = up_df.loc[i+1,'name'] # 下一个节点
            # 计算前面牵引站的距离
            pre_location = up_df.loc[up_df['name'] == up_df.loc[i,'preTss'],'location'].values 
            up_df.loc[i,'distance_preTss'] = up_df.loc[i,'location'] - pre_location if up_df.loc[i,'preTss'] != 0 else -1
            # 计算后面牵引站的距离
            # print(post_location.values) 用于检查, 必须把 values 取出来才可以用于切片
            post_location = (up_df.loc[up_df['name'] == up_df.loc[i,'postTss'],'location']).values 
            up_df.loc[i,'distance_postTss'] = post_location - up_df.loc[i,'location'] if up_df.loc[i,'postTss'] != 0 else -1 
        elif i == up_df.__len__()-1: # 如果是最后一个元素的话, 那么他的下一个节点是空的
            up_df.loc[i,'upPost'] = 0
            if up_df.loc[i,'location'] == up_df.loc[i-1,'location']: # 如果上一个节点的location与此节点相等, 那么是重合节点, 需要再向前移动
                if i-1==0: # 如果重合节点是第一个节点的话, 那么其实此节点没有上一节点
                    up_df.loc[i,'upPre'] = 0 # 0 代表没有节点
                else: # 如果重合节点不是第一个节点, 那么只需要在往上移动一个节点
                    up_df.loc[i,'upPre'] = up_df.loc[i-2,'name'] # 上一个节点
            else: # 如果没有重合的话
                up_df.loc[i,'upPre'] = up_df.loc[i-1,'name'] # 上一个节点
            # 计算前面牵引站的距离
            pre_location = up_df.loc[up_df['name'] == up_df.loc[i,'preTss'],'location'].values
            up_df.loc[i,'distance_preTss'] = up_df.loc[i,'location'] - pre_location if up_df.loc[i,'preTss'] != 0 else -1
            # 计算后面牵引站的距离
            post_location = up_df.loc[up_df['name'] == up_df.loc[i,'postTss'],'location'].values
            up_df.loc[i,'distance_postTss'] = post_location - up_df.loc[i,'location'] if up_df.loc[i,'postTss'] != 0 else -1
        else:
            if up_df.loc[i,'location'] == up_df.loc[i-1,'location']: # 如果上一个节点的location与此节点相等, 那么是重合节点, 需要再向前移动
                if i-1==0: # 如果重合节点是第一个节点的话, 那么其实此节点没有上一节点
                    up_df.loc[i,'upPre'] = 0 # 0 代表没有节点
                else: # 如果重合节点不是第一个节点, 那么只需要在往上移动一个节点
                    up_df.loc[i,'upPre'] = up_df.loc[i-2,'name'] # 上一个节点
            else: # 如果没有重合的话
                up_df.loc[i,'upPre'] = up_df.loc[i-1,'name'] # 上一个节点
            
            if up_df.loc[i,'location'] == up_df.loc[i+1,'location']: # 如果下一个节点的location与此节点相等, 那么是重合节点, 需要再向后移动
                if i+1 == up_df.__len__()-1: # 如果重合节点是最后一个元素的话, 那么他的下一个节点是空的
                    up_df.loc[i,'upPost'] = 0
                else:  # 如果重合节点不是最后一个节点, 那么只需要在往下移动一个节点
                    up_df.loc[i,'upPost'] = up_df.loc[i+2,'name'] # 下一个节点
            else: # 如果没有重合的话
                up_df.loc[i,'upPost'] = up_df.loc[i+1,'name'] # 下一个节点

            # 计算前面牵引站的距离
            pre_location = up_df.loc[up_df['name'] == up_df.loc[i,'preTss'],'location'].values
            up_df.loc[i,'distance_preTss'] = up_df.loc[i,'location'] - pre_location if up_df.loc[i,'preTss'] != 0 else -1
            # 计算后面牵引站的距离
            post_location = up_df.loc[up_df['name'] == up_df.loc[i,'postTss'],'location'].values
            up_df.loc[i,'distance_postTss'] = post_location - up_df.loc[i,'location'] if up_df.loc[i,'postTss'] != 0 else -1
    # print(up_df)
        
    # 相同的 处理下行的数据    
    for i in down_df.index:
        # 获取下行路线的停靠数据
        if down_df.loc[i,'class']==1: # TSS
            if down_df.loc[i,'location'] in down_locations:
                downstop = down_df.loc[(down_df['class']==0)&(down_df['location']==down_df.loc[i,'location']), 'name']
                down_df.loc[i, 'downStop'] = downstop.values[0]
        elif down_df.loc[i,'class']==0: # Train
            if down_df.loc[i,'location'] in tss_locations:
                stoptss =down_df.loc[(down_df['class']==1)&(down_df['location']==down_df.loc[i,'location']), 'name']
                down_df.loc[i, 'stopping'] = stoptss.values[0]

        # 获得拓扑信息
        if i==0: # 如果是第一个元素的话, 那么他的上一个节点是空的
            down_df.loc[i,'downPre'] = 0 # 0 代表没有节点
            if down_df.loc[i,'location'] == down_df.loc[i+1,'location']: # 如果下一个节点的location与此节点相等, 那么是重合节点, 需要再向后移动
                if i+1 == down_df.__len__()-1: # 如果重合节点是最后一个元素的话, 那么他的下一个节点是空的
                    down_df.loc[i,'downPost'] = 0
                else:  # 如果重合节点不是最后一个节点, 那么只需要在往下移动一个节点
                    down_df.loc[i,'downPost'] = down_df.loc[i+2,'name'] # 下一个节点
            else: # 如果没有重合的话
                down_df.loc[i,'downPost'] = down_df.loc[i+1,'name'] # 下一个节点
            # 计算前面牵引站的距离
            pre_location = down_df.loc[down_df['name'] == down_df.loc[i,'preTss'],'location'].values
            down_df.loc[i,'distance_preTss'] = down_df.loc[i,'location'] - pre_location if down_df.loc[i,'preTss'] != 0 else -1
            # 计算后面牵引站的距离
            post_location = down_df.loc[down_df['name'] == down_df.loc[i,'postTss'],'location'].values 
            down_df.loc[i,'distance_postTss'] = post_location - down_df.loc[i,'location'] if down_df.loc[i,'postTss'] != 0 else -1
        elif i == down_df.__len__()-1: # 如果是最后一个元素的话, 那么他的下一个节点是空的
            if down_df.loc[i,'location'] == down_df.loc[i-1,'location']: # 如果上一个节点的location与此节点相等, 那么是重合节点, 需要再向前移动
                if i-1==0: # 如果重合节点是第一个节点的话, 那么其实此节点没有上一节点
                    down_df.loc[i,'downPre'] = 0 # 0 代表没有节点
                else: # 如果重合节点不是第一个节点, 那么只需要在往上移动一个节点
                    down_df.loc[i,'downPre'] = down_df.loc[i-2,'name'] # 上一个节点
            else: # 如果没有重合的话
                down_df.loc[i,'downPre'] = down_df.loc[i-1,'name'] # 上一个节点
            down_df.loc[i,'downPost'] = 0
            # 计算前面牵引站的距离
            pre_location = down_df.loc[down_df['name'] == down_df.loc[i,'preTss'],'location'].values  
            down_df.loc[i,'distance_preTss'] = down_df.loc[i,'location'] - pre_location if down_df.loc[i,'preTss'] != 0 else -1
            # 计算后面牵引站的距离
            post_location = down_df.loc[down_df['name'] == down_df.loc[i,'postTss'],'location'].values 
            down_df.loc[i,'distance_postTss'] = post_location - down_df.loc[i,'location'] if down_df.loc[i,'postTss'] != 0 else -1
        else:
            if down_df.loc[i,'location'] == down_df.loc[i-1,'location']: # 如果上一个节点的location与此节点相等, 那么是重合节点, 需要再向前移动
                if i-1==0: # 如果重合节点是第一个节点的话, 那么其实此节点没有上一节点
                    down_df.loc[i,'downPre'] = 0 # 0 代表没有节点
                else: # 如果重合节点不是第一个节点, 那么只需要在往上移动一个节点
                    down_df.loc[i,'downPre'] = down_df.loc[i-2,'name'] # 上一个节点
            else: # 如果没有重合的话
                down_df.loc[i,'downPre'] = down_df.loc[i-1,'name'] # 上一个节点

            if down_df.loc[i,'location'] == down_df.loc[i+1,'location']: # 如果下一个节点的location与此节点相等, 那么是重合节点, 需要再向后移动
                if i+1 == down_df.__len__()-1: # 如果重合节点是最后一个元素的话, 那么他的下一个节点是空的
                    down_df.loc[i,'downPost'] = 0
                else:  # 如果重合节点不是最后一个节点, 那么只需要在往下移动一个节点
                    down_df.loc[i,'downPost'] = down_df.loc[i+2,'name'] # 下一个节点
            else: # 如果没有重合的话
                down_df.loc[i,'downPost'] = down_df.loc[i+1,'name'] # 下一个节点

            # 计算前面牵引站的距离
            pre_location = down_df.loc[down_df['name'] == down_df.loc[i,'preTss'],'location'].values 
            down_df.loc[i,'distance_preTss'] = down_df.loc[i,'location'] - pre_location if down_df.loc[i,'preTss'] != 0 else -1
            # 计算后面牵引站的距离  
            post_location = down_df.loc[down_df['name'] == down_df.loc[i,'postTss'],'location'].values 
            down_df.loc[i,'distance_postTss'] = post_location - down_df.loc[i,'location'] if down_df.loc[i,'postTss'] != 0 else -1
    # print(down_df)
                
    # 合并 up_df, down_df 纵向拼接
    df = pd.concat([up_df[['name', 'location', 'class', 'preTss', 'postTss', 'distance_preTss', 'distance_postTss','P','stopping']]
                    , down_df[['name', 'location', 'class', 'preTss', 'postTss', 'distance_preTss', 'distance_postTss','P','stopping']]]
                    , axis=0)
    # print(df)

    df = pd.merge(left=df, right=up_df.loc[:, ['name', 'upPre', 'upPost', 'upStop']], left_on='name', right_on='name', how='outer')
    df = pd.merge(left=df, right=down_df.loc[:, ['name', 'downPre', 'downPost', 'downStop']], left_on='name', right_on='name', how='outer')
    # 填充 upPre, upPost, downPre, downPost 的缺失值
    df.fillna(value=0, inplace=True)
    # 消除多余的牵引站的信息
    df.drop_duplicates(subset='name', inplace=True, keep='first')
    # print(df)

    # 需要把 第一个牵引站的前牵引站 name 设为 0, 距离设为 -1, 把其他牵引站的先后相邻关系修正
    df_tss = df.loc[df['class']==1,:].sort_values(by='location')
    df_tss.reset_index(inplace=True, drop=True)
    # 修正其他的牵引站 preTss 数据 和 postTss 数据
    for i in range(len(df_tss)-1):
        df_tss.loc[i+1,['preTss', 'distance_preTss']] = df_tss.loc[i,'name'], abs(df_tss.loc[i+1,'location']-df_tss.loc[i, 'location'])
        df_tss.loc[i,['postTss', 'distance_postTss']] = df_tss.loc[i+1,'name'], abs(df_tss.loc[i+1,'location']-df_tss.loc[i, 'location'])
    # 修正 第一个牵引站的 preTss 数据
    first_name = df_tss.loc[0,'name']
    df_tss.loc[df_tss['name']==first_name, ['preTss', 'distance_preTss']] = [0,-1]
    # 修正 最后牵引站的后牵引站 name 设为 0, 距离设为 -1 
    last_name = df_tss.loc[len(df_tss)-1,'name']
    df_tss.loc[df_tss['name']==last_name, ['postTss', 'distance_postTss']] = [0,-1]

    # display(df)
    df.drop_duplicates(subset='name', inplace=True, keep='first')
    # display(df)
    df.sort_values(by='name', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[['name', 'class', 'upPre', 'upPost', 'downPre', 'downPost', 'preTss', 'distance_preTss', 'postTss', 'distance_postTss','P','stopping', 'upStop', 'downStop', 'location']]

    # display(df)
    df.loc[df['class']==1,['preTss', 'distance_preTss', 'postTss', 'distance_postTss']] = df_tss.loc[:,['preTss', 'distance_preTss', 'postTss', 'distance_postTss']]

    # display(df)
    '''
    由于 digitize 的分箱方法, 是左开右闭的, 
    对于牵引站而言, 牵引站无法获取正确的 下一牵引站数据, 在上面 已经用 df_tss 修正了
    对于列车而言, 因此当一辆列车与牵引站重合时, 他的下一个牵引站依然显示是此牵引站并且距离为零
    下面的代码对此进行修正
    '''
    # display(df_tss)
    df_train = df[df['class']==0]
    df_train.reset_index(drop=True, inplace=True)
    # display(df_train)
    for i in range(len(df_train)):
        if df_train.loc[i, 'distance_postTss']==0:
            before_postTss = df_train.loc[i,'postTss']
            # print('列车 {} 与牵引站 {} 重合'.format(df_train.loc[i,'name'], before_postTss))
            df_train.loc[i,'postTss'] = df_tss.loc[df_tss['name']==before_postTss, 'postTss'].values
            df_train.loc[i,'distance_postTss'] = df_tss.loc[df_tss['name']==before_postTss, 'distance_postTss'].values

    # display(df_train)
    df.loc[df['class']==0,['preTss', 'distance_preTss', 'postTss', 'distance_postTss']] = df_train.loc[:,['preTss', 'distance_preTss', 'postTss', 'distance_postTss']].values
    
    # print(df)
    df.fillna(value=0,inplace=True)
    
    # 修正前后相邻信息
    for i in df.index:
        if  df.loc[(df['name'] == df.loc[i,'upPre']), 'stopping'].values.size > 0:
            if df.loc[(df['name'] == df.loc[i,'upPre']), 'stopping'].values != 0:
                df.loc[i,'upPre'] = df.loc[(df['name'] == df.loc[i,'upPre']), 'stopping'].values
        if  df.loc[(df['name'] == df.loc[i,'upPost']), 'stopping'].values.size > 0:
            if  df.loc[(df['name'] == df.loc[i,'upPost']), 'stopping'].values != 0:
                df.loc[i,'upPost'] = df.loc[(df['name'] == df.loc[i,'upPost']), 'stopping'].values 
        if df.loc[(df['name'] == df.loc[i,'downPre']), 'stopping'].values.size > 0:
            if df.loc[(df['name'] == df.loc[i,'downPre']), 'stopping'].values != 0:
                df.loc[i,'downPre'] = df.loc[(df['name'] == df.loc[i,'downPre']), 'stopping'].values 
        if df.loc[(df['name'] == df.loc[i,'downPost']), 'stopping'].values.size > 0:
            if df.loc[(df['name'] == df.loc[i,'downPost']), 'stopping'].values != 0:
                df.loc[i,'downPost'] = df.loc[(df['name'] == df.loc[i,'downPost']), 'stopping'].values

######
    
    # 新增相邻节点的距离信息
    for i in df.index:
        if df.loc[i,'class'] == 1:
            if df.loc[i,'upPre'] < 1:
                df.loc[i,'distance_upPre'] = -1
            else:
                df.loc[i,'distance_upPre'] = abs(df.loc[i,'location']-df.loc[df['name']==df.loc[i,'upPre'],'location']).values

            if df.loc[i,'upPost'] < 1:
                df.loc[i,'distance_upPost'] = -1
            else:
                df.loc[i,'distance_upPost'] = abs(df.loc[i,'location']-df.loc[df['name']==df.loc[i,'upPost'],'location']).values

            if df.loc[i,'downPre'] < 1:
                df.loc[i,'distance_downPre'] = -1
            else:
                df.loc[i,'distance_downPre'] = abs(df.loc[i,'location']-df.loc[df['name']==df.loc[i,'downPre'],'location']).values

            if df.loc[i,'downPost'] < 1:
                df.loc[i,'distance_downPost'] = -1
            else:
                df.loc[i,'distance_downPost'] = abs(df.loc[i,'location']-df.loc[df['name']==df.loc[i,'downPost'],'location']).values
                
        else:
            if df.loc[i,'upPre']==0 and df.loc[i,'upPost']==0:
                if df.loc[i,'downPre'] < 1:
                    df.loc[i,'distance_downPre'] = -1
                else:
                    df.loc[i,'distance_downPre'] = abs(df.loc[i,'location']-df.loc[df['name']==df.loc[i,'downPre'],'location']).values

                if df.loc[i,'downPost'] < 1:
                    df.loc[i,'distance_downPost'] = -1
                else:
                    df.loc[i,'distance_downPost'] = abs(df.loc[i,'location']-df.loc[df['name']==df.loc[i,'downPost'],'location']).values
            else:
                if df.loc[i,'upPre'] < 1:
                    df.loc[i,'distance_upPre'] = -1
                else:
                    df.loc[i,'distance_upPre'] = abs(df.loc[i,'location']-df.loc[df['name']==df.loc[i,'upPre'],'location']).values

                if df.loc[i,'upPost'] < 1:
                    df.loc[i,'distance_upPost'] = -1
                else:
                    df.loc[i,'distance_upPost'] = abs(df.loc[i,'location']-df.loc[df['name']==df.loc[i,'upPost'],'location']).values
    df = df[['name', 'class', 'upPre', 'upPost', 'downPre', 'downPost', 'preTss', 'distance_preTss', 'postTss', 'distance_postTss'
             ,'P', 'location','distance_upPre','distance_upPost','distance_downPre','distance_downPost','stopping', 'upStop', 'downStop']]
    df.fillna(-1, inplace=True)

    # 规定每一列的数值类型
    for i in df.columns:
        if i in ['name', 'class', 'upPre', 'upPost', 'downPre', 'downPost', 'preTss',  'postTss','stopping', 'upStop', 'downStop']:
            df[i] = df[i].astype('int')
        else:
            df[i] = df[i].astype('float') 

    return df