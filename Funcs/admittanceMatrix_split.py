#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   admittanceMatrix.py
@Time    :   2025/02/19 10:49:01
@Author  :   勾子瑭
@Desc.   :   与 admittanceMatrix 不同，admittanceMatrix_split 分别计算接触网和铁轨的导纳
'''

# here put the import lib
import numpy as np

def admittanceMatrix_split(Dataframe, DF_noStop, r_r = 0.017,r_c = 0.02):
    DF = Dataframe.reset_index(inplace = True, drop = True)
    DF_noStop.reset_index(inplace = True, drop = True)
    numsNode = len(DF_noStop.index)
    Y = np.zeros(shape=(numsNode,numsNode))
    Y_c = np.zeros(shape=(numsNode,numsNode))
    Y_r = np.zeros(shape=(numsNode,numsNode))
    DF_inv = 1/(DF_noStop.loc[:,['distance_upPre','distance_upPost','distance_downPre','distance_downPost']])
    keys = {
        'upPre':'distance_upPre'
        ,'upPost':'distance_upPost'
        ,'downPre':'distance_downPre'
        ,'downPost':'distance_downPost'
    }
    
    for i in range(numsNode):
        is_error = 0
        neigh = DF_noStop.loc[i,['upPre','upPost','downPre','downPost']]
        dist_neigh_inv = DF_inv.loc[i,['distance_upPre','distance_upPost','distance_downPre','distance_downPost']]

        # 由于改变了停靠列车的相邻关系数据，所以可能会出现 Tss 前后没有机车导致上行下行的数据出现重复， 导致 导纳矩阵 计算不正确，因此需要筛掉重复的数据
        if len(neigh)==4:
            if (neigh[['upPre','upPost']].values == neigh[['downPre','downPost']].values).all():
            #     neigh[['downPre','downPost']] = [0,0]
            #     is_error = 1
            # if neigh['upPost'] == neigh['downPost']:
            #     neigh['downPost'] == 0
            #     is_error = 1
            # if neigh['upPre'] == neigh['downPre']:
            #     neigh['downPre'] == 0
            #     is_error = 1
                pass
        for (j,k) in enumerate(neigh):
            if k==0:
                dist_neigh_inv.iloc[j] = -1
        
        for j in ['upPre','upPost','downPre','downPost']:
            if neigh[j] !=0:
                k = keys[j]
                Y[i,DF_noStop[DF_noStop['name']==neigh[j]].index.values[0]] += dist_neigh_inv.loc[k]/(r_c+r_r)
                Y_c[i,DF_noStop[DF_noStop['name']==neigh[j]].index.values[0]] += dist_neigh_inv.loc[k]/(r_c)
                Y_r[i,DF_noStop[DF_noStop['name']==neigh[j]].index.values[0]] += dist_neigh_inv.loc[k]/(r_r)

        Y[i,i] += -sum(Y[i])
        Y_c[i,i] += -sum(Y_c[i])
        Y_r[i,i] += -sum(Y_r[i])

    return {"Y":Y
            ,"Yc":Y_c
            ,"Yr":Y_r}