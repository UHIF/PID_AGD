class PID:
    def __init__(self, P=0.2, I=0.0, D=0.0):
        self.kp = P
        self.ki = I
        self.kd = D
        self.uPrevious = 0
        self.uCurent = 0
        self.setValue = 0
        self.lastErr = 0
        self.preLastErr = 0
        self.errSum = 0
        self.errSumLimit = 10

    # 位置式PID
    def control(self, curValue):
        err = self.setValue - curValue
        dErr = err - self.lastErr
        self.preLastErr = self.lastErr
        self.lastErr = err
        self.errSum += err
        outPID = self.kp * err + (self.ki * self.errSum) + (self.kd * dErr)
        return outPID
