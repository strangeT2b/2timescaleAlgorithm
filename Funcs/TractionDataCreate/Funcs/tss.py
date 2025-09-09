class Tss:
    def __init__(self, name, location, preTss=None, postTss=None, P=0, stopping_time_up=0, stopping_time_down=0):
        self.name = name # 需要统一编号
        self.location = location
        self.preTss = preTss
        self.PostTss = postTss
        self.P = P
        self.stopping_time_up = stopping_time_up
        self.stopping_time_down = stopping_time_down

    