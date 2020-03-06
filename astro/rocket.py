import numpy as np


class GravityTurn:
    def __init__(self, mu=np.double(2.986e5), stages=2):
        self.mu: np.double = mu
        self.stage = []
        for i in range(stages):
            self.stage.append(RocketStage())
        self.m_star = None

    def def_payload(self, m):
        self.m_star: np.double = m

    def def_stage(self, isp, epsilon, m0, stage):
        self.stage[stage].def_stage(isp, epsilon, m0)

    def solve(self):
        pass


class RocketStage:
    def __init__(self):
        self.isp = None
        self.epsilon = None
        self.mass = None

    def def_stage(self, isp, epsilon, m0):
        self.isp: np.double = isp
        self.epsilon: np.double = epsilon
        self.mass: np.double = m0  # Initial mass (Including Fuel)
