import time
import numpy as np
import pandas as pd
from numpy.random import default_rng
from vrp import VRP
from sa import SA
from aco import ACO
from visual import Visual


class Experiments:
    def __init__(self, sa_list, sa_sensor_list, aco_list, aco_sensor_list):
        self.sa_1 = sa_list[0]
        self.sa_2 = sa_list[1]
        self.sa_3 = sa_list[2]
        self.sa_4 = sa_list[3]
        self.sa_5 = sa_list[4]
        self.sa_sensor_1 = sa_sensor_list[0]
        self.sa_sensor_2 = sa_sensor_list[1]
        self.sa_sensor_3 = sa_sensor_list[2]
        self.sa_sensor_4 = sa_sensor_list[3]
        self.sa_sensor_5 = sa_sensor_list[4]
        self.aco_1 = aco_list[0]
        self.aco_2 = aco_list[1]
        self.aco_3 = aco_list[2]
        self.aco_4 = aco_list[3]
        self.aco_5 = aco_list[4]
        self.aco_sensor_1 = aco_sensor_list[0]
        self.aco_sensor_2 = aco_sensor_list[1]
        self.aco_sensor_3 = aco_sensor_list[2]
        self.aco_sensor_4 = aco_sensor_list[3]
        self.aco_sensor_5 = aco_sensor_list[4]

    def experiment_1(self, ):
        times = []
        bests = []
        for i in range(0, 100):
            pass

