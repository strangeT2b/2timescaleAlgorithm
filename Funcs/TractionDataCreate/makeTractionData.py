
import pandas as pd
import numpy as np
import random

class Tss:
    def __init__(self, name, location, preTss=None, postTss=None, P=0, stopping_time=0):
        self.name = name # 需要统一编号
        self.location = location
        self.preTss = preTss
        self.PostTss = postTss
        self.P = P
        self.stopping_time = stopping_time

class Train:
    # 初始化 Train, 所有的时间单位都用 km, h 为单位
    def __init__(self, name, location, direction = 1, speed = 0, acceleration=1, deceleration=1, max_speed=15
                 , timer=0, stop_time=10, state='Initial', distance_next_tss=2000, initial_wait_time=0, P=0
                 , constantP_time=30, traction_system=None):
        self.name = name # 编号
        self.location = location # 位置, 可以用来判断两端的牵引站, 从而判断他们的位置关系
        self.direction = direction # 规定列车的运行方向, 从而规定了列车的 上行 or 下行
        self.speed = speed # 速度
        self.acceleration = acceleration # 加速度
        self.deceleration = deceleration # 减速的加速度
        self.max_speed = max_speed # 最大的速度, 用来判断是否达到最大的速度
        self.timer = timer # 一个计时器, 用来判断还需要靠岸多久才出发
        self.stop_time = stop_time # 列车靠岸累计多久会发车
        self.state = state # 用来表示机车现在的状态, Accelerating, Decelerating, constantSpeed, Stopping, 其中 constantSpeed 被 constantP 和 Coasting 代替
        self.time_to_max_speed = (self.max_speed-self.speed) / self.acceleration if self.acceleration != 0 else 0 # 当前速度增到最大速度所需要的时间
        self.distance_to_max_speed = self.speed*self.time_to_max_speed + 0.5 * self.acceleration * self.time_to_max_speed ** 2 # 当前速度增到最大速度所需要的距离
        self.time_to_stop = self.speed / self.deceleration if self.deceleration != 0 else 0 # 速度减少到 0 所需要的时间
        self.distance_to_stop = 0.5 * self.deceleration * self.time_to_stop ** 2 # 速度减少到零所需要的距离
        self.distance_next_tss = distance_next_tss # 距离下一牵引站的距离
        self.initial_wait_time = initial_wait_time # 初始等待时间
        self.P = P # 列车的功率
        self.constantP_time = constantP_time # 规定恒功率的时间
        self.constantP_timer = self.constantP_time # 恒功率计时器, 结束后进入惰性状态
        self.stoppping_time_dict = {int(tss.location) : tss.stopping_time for tss in traction_system} # 用来存储每个牵引站的停靠时间


    '''
    经过调查, 地铁的时速大概在 40 ~ 60 km/h, 合大概 11 ~ 16.7 m/s, 
    然而地铁的加速度大概在 地铁电车加速度一般在3-4 km/h/s的范围内。如上海地铁一号线AC01型电车加速度是3.2, 日本的某些电车加速度甚至可以达到4以上, 相当于 0.8 ~ 1.11 m/s/s
    减速度应当比地铁加速度略微大一点(可能是用于动能回收)
    这样计算, 加速跑出去的里程大概在 155 m 以下, 而减速的里程应当比 155m 更小, 对于一站地的举例来说, 加速时与减速时的过程是很小的, 
    因此可以断定 在没有特殊情况时 地铁时一定可以加速到最大速度然后匀速行驶的, 也就是说 地铁在两站地之间 一定会经历 加速 -》 匀速 ——》 减速 的过程
    所以不考虑 加速过程中 没有加到最大速度就需要减速靠岸的情况
    这个思路可以简化代码
    '''

    def show_train(self):
        print('列车的编号为:{}\n'.format(self.name))
        print('列车当前的状态是:{}\n'.format(self.state))
        print('列车的位置为:{}\n'.format(self.location))
        print('列车的运行方向为:{}\n'.format(self.direction))
        print('列车的速度为:{} m/s\n'.format(self.speed))
        print('列车的距离下一牵引站: {} m'.format(self.distance_next_tss))
        print('列车的加速度为:{} m/s/s\n'.format(self.acceleration))
        print('列车的减速度为:{} m/s/s\n'.format(self.deceleration))
        print('列车的最大速度为:{} m/s\n'.format(self.max_speed))
        print('列车的计时器为:{}\n'.format(self.timer))
        print('列车需要 {} s 加到最大速度\n'.format(self.time_to_max_speed))
        print('列车需要 {} m 加到最大速度\n'.format(self.distance_to_max_speed))
        print('列车需要 {} s 减速到零\n'.format(self.time_to_stop))
        print('列车需要 {} m 减速到零\n'.format(self.distance_to_stop))


    def train_update(self, DF):
        train_df = DF.loc[DF['class']==0, :] # 取出列车的 df
        idx_self = train_df['name']==self.name
        # print(train_df.loc[train_df['name']==self.name, 'upPre'].values)
        if train_df.loc[train_df['name']==self.name, 'upPre'].values==0 and train_df.loc[train_df['name']==self.name, 'upPost'].values==0: # 下行车辆
            self.distance_next_tss = DF.loc[DF['name']==self.name, 'distance_preTss'].values[0]
        elif train_df.loc[train_df['name']==self.name, 'downPre'].values==0 and train_df.loc[train_df['name']==self.name, 'downPost'].values==0: # 上行车辆
            self.distance_next_tss = DF.loc[DF['name']==self.name, 'distance_postTss'].values[0]

        self.time_to_max_speed = (self.max_speed-self.speed) / self.acceleration if self.acceleration != 0 else 0 # 当前速度增到最大速度所需要的时间
        self.distance_to_max_speed = self.speed*self.time_to_max_speed + 0.5 * self.acceleration * self.time_to_max_speed ** 2 # 当前速度增到最大速度所需要的距离
        self.time_to_stop = self.speed / self.deceleration if self.deceleration != 0 else 0 # 速度减少到 0 所需要的时间
        self.distance_to_stop = 0.5 * self.deceleration * self.time_to_stop ** 2 # 速度减少到零所需要的距离
        pass


    def accelerate(self, duration, min_tss_location, max_tss_location):
        # 计算加速所需时间和距离

        self.state = 'Accelerating'
        self.timer = 0 # 靠岸时间清零
        self.P = 5 # 加速功率为 5 MW

        # 如果在 duration 中无法加到最大速度
        if duration <= self.time_to_max_speed:
            # 加速达不到最大速度的情况
            self.location += (self.speed*duration + (0.5 * self.acceleration * duration ** 2))*self.direction # 加上 direction, positive 表示 上行, negative 表示 下行
            self.speed = self.speed + self.acceleration * duration

            # 更新状态
            self.time_to_max_speed = (self.max_speed-self.speed) / self.acceleration if self.acceleration != 0 else 0 # 当前速度增到最大速度所需要的时间
            self.distance_to_max_speed = self.speed*self.time_to_max_speed + 0.5 * self.acceleration * self.time_to_max_speed ** 2 # 当前速度增到最大速度所需要的距离
            self.time_to_stop = self.speed / self.deceleration if self.deceleration != 0 else 0 # 速度减少到 0 所需要的时间
            self.distance_to_stop = 0.5 * self.deceleration * self.time_to_stop ** 2 # 速度减少到零所需要的距离

        # 如果在 duration 达到最大速度后匀速
        else:
            self.location += (self.distance_to_max_speed)*self.direction
            self.speed = self.max_speed
            time_extend = duration - self.time_to_max_speed # 达到最大速度之后剩余的匀速时间
            #self.location += self.max_speed*time_extend*self.direction # 匀速前进的距离

            # 更新状态
            self.distance_next_tss += -abs((self.distance_to_max_speed)*self.direction) # 很重要的, 更新与下一个牵引站之间的距离
            self.time_to_max_speed = (self.max_speed-self.speed) / self.acceleration if self.acceleration != 0 else 0 # 当前速度增到最大速度所需要的时间
            self.distance_to_max_speed = self.speed*self.time_to_max_speed + 0.5 * self.acceleration * self.time_to_max_speed ** 2 # 当前速度增到最大速度所需要的距离
            self.time_to_stop = self.speed / self.deceleration if self.deceleration != 0 else 0 # 速度减少到 0 所需要的时间
            self.distance_to_stop = 0.5 * self.deceleration * self.time_to_stop ** 2 # 速度减少到零所需要的距离
            self.constantP_timer = self.constantP_time # 即将进入恒功率匀速状态, 重制计时器
            return self.constantSpeed(duration = time_extend, min_tss_location = min_tss_location, max_tss_location=max_tss_location)


    def constantSpeed(self, duration, min_tss_location, max_tss_location):
        # 恒功率倒计时
        if self.constantP_timer > 0:
            self.state = 'ConstantP'
            self.P = 1.5 # 恒功率 1.5 MW
            self.constantP_timer -= duration # 倒计时
        else:
            self.state = 'Coasting'
            self.P = 0.4 # 惰行, 0.4 MW

        # 如果 途中会进入减速
        if self.distance_next_tss < duration * self.max_speed+self.distance_to_stop: 
            self.location += (self.distance_next_tss - self.distance_to_stop)*self.direction # 行进到减速点
            
            # 更新状态
            self.distance_next_tss += -abs((self.distance_next_tss - self.distance_to_stop)*self.direction) # 很重要的, 更新与下一个牵引站之间的距离
            self.time_to_max_speed = (self.max_speed-self.speed) / self.acceleration if self.acceleration != 0 else 0 # 当前速度增到最大速度所需要的时间
            self.distance_to_max_speed = self.speed*self.time_to_max_speed + 0.5 * self.acceleration * self.time_to_max_speed ** 2 # 当前速度增到最大速度所需要的距离
            self.time_to_stop = self.speed / self.deceleration if self.deceleration != 0 else 0 # 速度减少到 0 所需要的时间
            self.distance_to_stop = 0.5 * self.deceleration * self.time_to_stop ** 2 # 速度减少到零所需要的距离

            extend_time = duration - (self.distance_next_tss - self.distance_to_stop)/self.speed # 计算剩下进入减速的时间
            return self.decelerate(duration = extend_time, min_tss_location = min_tss_location, max_tss_location=max_tss_location)

        # 如果 途中不会进入减速, 继续匀速行驶
        else: 
            self.location += self.max_speed*duration*self.direction

            # 更新状态
            self.time_to_max_speed = (self.max_speed-self.speed) / self.acceleration if self.acceleration != 0 else 0 # 当前速度增到最大速度所需要的时间
            self.distance_to_max_speed = self.speed*self.time_to_max_speed + 0.5 * self.acceleration * self.time_to_max_speed ** 2 # 当前速度增到最大速度所需要的距离
            self.time_to_stop = self.speed / self.deceleration if self.deceleration != 0 else 0 # 速度减少到 0 所需要的时间
            self.distance_to_stop = 0.5 * self.deceleration * self.time_to_stop ** 2 # 速度减少到零所需要的距离
        pass


    def decelerate(self, duration, min_tss_location, max_tss_location):
        # 计算减速所需时间和距离
        self.state = 'Decelerating' # 更改 减速 状态
        self.P = -5 # 制动 -5MW

        # 如果 减速达不到 stop 的状态
        if duration <= self.time_to_stop: 
            # 减速不到停车状态的情况
            self.location += (self.speed * duration - 0.5 * self.deceleration * duration ** 2)*self.direction
            self.speed = self.speed - self.deceleration * duration

            # 更新状态
            self.time_to_max_speed = (self.max_speed-self.speed) / self.acceleration if self.acceleration != 0 else 0 # 当前速度增到最大速度所需要的时间
            self.distance_to_max_speed = self.speed*self.time_to_max_speed + 0.5 * self.acceleration * self.time_to_max_speed ** 2 # 当前速度增到最大速度所需要的距离
            self.time_to_stop = self.speed / self.deceleration if self.deceleration != 0 else 0 # 速度减少到 0 所需要的时间
            self.distance_to_stop = 0.5 * self.deceleration * self.time_to_stop ** 2 # 速度减少到零所需要的距离

        # 完全停车的情况
        else:
            self.location += (self.distance_to_stop)*self.direction
            self.speed = 0

            # 更新状态
            self.time_to_max_speed = (self.max_speed-self.speed) / self.acceleration if self.acceleration != 0 else 0 # 当前速度增到最大速度所需要的时间
            self.distance_to_max_speed = self.speed*self.time_to_max_speed + 0.5 * self.acceleration * self.time_to_max_speed ** 2 # 当前速度增到最大速度所需要的距离
            self.time_to_stop = self.speed / self.deceleration if self.deceleration != 0 else 0 # 速度减少到 0 所需要的时间
            self.distance_to_stop = 0.5 * self.deceleration * self.time_to_stop ** 2 # 速度减少到零所需要的距离
            
            self.stop_time = self.stoppping_time_dict[int(self.location)] # 获取当前牵引站的停靠时间

            extend_time = duration-self.time_to_stop # 记录靠岸的时间
            return self.stop(duration=extend_time, min_tss_location=min_tss_location, max_tss_location=max_tss_location)
        

    def stop(self, duration, min_tss_location, max_tss_location):
        self.state = 'Stopping'
        self.P = 0.4
        if duration+self.timer > self.stop_time: # 如果超过了需要等待的时间, 进入加速状态
            if self.location == min_tss_location or self.location == max_tss_location: 
                # 如果到达线路的边缘, 进入反向路线
                self.direction = -self.direction
                return self.accelerate(abs(duration+self.timer-self.stop_time), min_tss_location, max_tss_location)
            else: # 如果没有到达边缘, 继续前行
                return self.accelerate(abs(duration+self.timer-self.stop_time), min_tss_location, max_tss_location)

        elif duration+self.timer <= self.stop_time: # 如果不超过需要等待的时间, 保持 stop 状态
            self.timer += duration # 更新计时器 timer
            pass
            # return self.stop(duration, min_tss_location, max_tss_location)


    def initial(self, duration, min_tss_location, max_tss_location):
        self.speed = 0
        self.P = 0.4
        if self.location == min_tss_location:
            # 如果是第一个牵引站, 那么他的方向一定是上行
            self.direction = 1
        elif self.location == max_tss_location: 
            # 如果是最后一个牵引站, 那么他的方向一定是下行
            self.direction = -1
        # 倒计时结束, 出发
        if self.initial_wait_time <= 0:
            extend_time = abs(self.initial_wait_time)
            return self.accelerate(duration = extend_time, min_tss_location = min_tss_location, max_tss_location=max_tss_location)
        # 倒计时还没有结束, 继续等待
        else:
            self.initial_wait_time -= duration
            pass


    def simulation(self, duration, min_tss_location, max_tss_location): # 需要规定最近与最远的牵引站location, 方便掉头操作
        if self.state == 'Initial':
            return self.initial(duration, min_tss_location = min_tss_location, max_tss_location=max_tss_location)
        elif self.state == 'Stopping': # 如果停留时间
            return self.stop(duration, min_tss_location = min_tss_location, max_tss_location=max_tss_location)
        elif self.state == 'Accelerating':
            return self.accelerate(duration, min_tss_location = min_tss_location, max_tss_location=max_tss_location)
        elif self.state == 'ConstantP' or self.state == 'Coasting':
            return self.constantSpeed(duration, min_tss_location = min_tss_location, max_tss_location=max_tss_location)
        elif self.state == 'Decelerating':
            return self.decelerate(duration, min_tss_location = min_tss_location, max_tss_location=max_tss_location)
        pass


def TractionSystem(locations, names = None):
    num_tss = len(locations)
    locations.sort()
    tss_list = list()
    if names == None:
        names = np.arange(1, num_tss+1) # 如果没有输入 names 参数的话, 直接用 1,2,... 来为牵引站编号
    # print('牵引站的个数是{}\n牵引站的编号分别是{}:\n 牵引站的位置分别是:{}\n'.format(num_tss, names, locations))
    for i in range(num_tss):
        preTss = i-1 
        postTss = i+1 if i < num_tss-1 else -1
        tss_list.append(Tss(names[i], locations[i]))
    return tss_list


def TrainSystem(up_locations, down_locations, num_tss=None,stop_time=10, rdm_wait = False, max_wait_time = 30, initial_wait_time = None):
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
            trainUp_list.append(Train(up_names[i], up_locations[i], direction=1, stop_time=stop_time, initial_wait_time=random.randint(0,max_wait_time)))
        for i in range(num_down):
            trainDown_list.append(Train(down_names[i], down_locations[i], direction=-1, stop_time=stop_time, initial_wait_time=random.randint(0,max_wait_time)))
    # 同时出发
    elif initial_wait_time != None:
        for i in range(num_up):
            trainUp_list.append(Train(up_names[i], up_locations[i], direction=1, stop_time=stop_time, initial_wait_time=initial_wait_time[i]))
        for i in range(num_down):
            trainDown_list.append(Train(down_names[i], down_locations[i], direction=-1, stop_time=stop_time, initial_wait_time=initial_wait_time[num_up+i]))
    else:
        for i in range(num_up):
            trainUp_list.append(Train(up_names[i], up_locations[i], direction=1, stop_time=stop_time))
        for i in range(num_down):
            trainDown_list.append(Train(down_names[i], down_locations[i], direction=-1, stop_time=stop_time))
    # print('上行列车的个数是{}\n上行列车的编号分别是{}:\n 上行列车的位置分别是:{}\n'.format(num_up, up_names, up_locations))
    # print('下行列车的个数是{}\n下行列车的编号分别是{}:\n 下行列车的位置分别是:{}\n'.format(num_down, down_names, down_locations))

    return trainUp_list, trainDown_list    

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
                up_df.loc[i, 'upStop'] = upstop.values
        elif up_df.loc[i,'class']==0: # Train
            if up_df.loc[i,'location'] in tss_locations:
                stoptss =up_df.loc[(up_df['class']==1)&(up_df['location']==up_df.loc[i,'location']), 'name']
                up_df.loc[i, 'stopping'] = stoptss.values

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
                down_df.loc[i, 'downStop'] = downstop.values
        elif down_df.loc[i,'class']==0: # Train
            if down_df.loc[i,'location'] in tss_locations:
                stoptss =down_df.loc[(down_df['class']==1)&(down_df['location']==down_df.loc[i,'location']), 'name']
                down_df.loc[i, 'stopping'] = stoptss.values

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

# 绘制动画
import pandas as pd
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

# 示例用法
import time
tic = time.time()
tss_locations = [0,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000]
upTrain_locations = [0,2000,4000,8000,10000,12000,14000,16000,18000,20000]
downTrain_locations = [2000,4000,6000,8000,10000,12000,14000,16000]
tss_list = TractionSystem(tss_locations)
start_interval = 15
intial_wait_time = list(range(0,(len(upTrain_locations)+len(downTrain_locations))*30, start_interval))
upTrain_list, downTrain_list = TrainSystem(upTrain_locations, downTrain_locations,num_tss=len(tss_list), stop_time=35, rdm_wait=False, max_wait_time=60,initial_wait_time = intial_wait_time)
df = topology(tss_list, upTrain_list, downTrain_list)

duration = 1 # 以 1 s 为单位
iter_num =60*60-1 # 迭代次数, 也就是运行多少秒

iter_dfs = list()
iter_dfs.append(df)


for i in range(iter_num):
    # 注意上行与下行的列表编号是反向的的
    if upTrain_list[-1].location >= max(tss_locations):
        downTrain_list.insert(0, upTrain_list[-1])
        upTrain_list.pop(-1)
        downTrain_list[0].location = max(tss_locations) - abs(max(tss_locations)-downTrain_list[0].location)
    if downTrain_list[-1].location <= min(tss_locations):
        upTrain_list.insert(0, downTrain_list[-1])
        downTrain_list.pop(-1)
        upTrain_list[0].location = min(tss_locations) + abs(min(tss_locations)-upTrain_list[0].location)
    for j in upTrain_list+downTrain_list:
        j.train_update(iter_dfs[i])
        j.simulation(duration = 1, min_tss_location=min(tss_locations),max_tss_location=max(tss_locations))
    
    # print(iter_dfs[i])
    # print('上行有 {} 辆车, 下行有 {} 辆车\n'.format(len(upTrain_list), len(downTrain_list)))
    # upTrain_list[1].show_train()
    
    iter_df = topology(tss_list, upTrain_list, downTrain_list)
    df = pd.concat([df,iter_df], axis=0)
    iter_dfs.append(iter_df)
    print('进度为 {} %\n'.format(i/iter_num*100))

df.loc[df['P']==0, 'P']=0.4
df.loc[df['class']==1, 'P']=0
df['E'] = 0
df.loc[df['class']==1, 'E'] = 1593
df['Rs'] = 0
df.loc[df['class']==1, 'Rs'] = 0.02
df['V'] = 1500
# df = df.loc[:,['name', 'class','upPre','upPost','downPre','downPost','preTss','distance_preTss','postTss','distance_postTss','E','Rs','V','P', 'stopping', 'upStop', 'downStop', 'location']]
for i in ['distance_preTss','distance_postTss','distance_upPre','distance_upPost','distance_downPre','distance_downPost']:
    df.loc[df[i]!=-1,i] = df.loc[df[i]!=-1,i]/1000
df['P'] = df['P']*1000000

one_length = len(iter_df)
print(df)
print('一次采样共 {} 行\n'.format(one_length))

df.to_csv('./dataset/oneHour_delay30.csv',index=False)

df.drop(columns=['location'],inplace=True)
df.to_csv('./dataset/oneHour_delay30_dropLocation.csv',index=False)

toc = time.time()
print('耗时: {}s\n'.format(toc-tic))

traction_length = max(tss_locations)-min(tss_locations)
plot_dataframe_animation(iter_dfs
                         , min_location=min(tss_locations)
                         , max_location=max(tss_locations))




