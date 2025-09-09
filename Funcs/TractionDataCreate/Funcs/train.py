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
        self.state = state # 用来表示机车现在的状态, Accelerating, Decelerating, constantSpeed, Stopping, 其中 constantSpeed 被 constantP 和 Coasting 代替
        self.time_to_max_speed = (self.max_speed-self.speed) / self.acceleration if self.acceleration != 0 else 0 # 当前速度增到最大速度所需要的时间
        self.distance_to_max_speed = self.speed*self.time_to_max_speed + 0.5 * self.acceleration * self.time_to_max_speed ** 2 # 当前速度增到最大速度所需要的距离
        self.time_to_stop = self.speed / self.deceleration if self.deceleration != 0 else 0 # 速度减少到 0 所需要的时间
        self.distance_to_stop = 0.5 * self.deceleration * self.time_to_stop ** 2 # 速度减少到零所需要的距离
        self.distance_next_tss = distance_next_tss # 距离下一牵引站的距离

        self.P = P # 列车的功率
        self.constantP_time = constantP_time # 规定恒功率的时间
        self.constantP_timer = self.constantP_time # 恒功率计时器, 结束后进入惰性状态
        if self.direction == 1:
            self.stoppping_time_dict = {int(tss.location) : tss.stopping_time_up for tss in traction_system} # 用来存储每个牵引站的停靠时间
        elif self.direction == -1:
            self.stoppping_time_dict = {int(tss.location) : tss.stopping_time_down for tss in traction_system} # 用来存储每个牵引站的停靠时间
        self.initial_wait_time = self.stoppping_time_dict[int(self.location)]  # 初始等待时间
        self.stop_time = self.stoppping_time_dict[int(self.location)] # 初始等待时间 # 列车靠岸累计多久会发车
    
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