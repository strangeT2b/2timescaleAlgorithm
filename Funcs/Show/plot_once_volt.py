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
