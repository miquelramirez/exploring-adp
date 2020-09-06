"""
    Household power consumption and generation model
"""

import json
import scipy.interpolate
import numpy as np

from datetime import datetime
from datetime import timedelta

class Model(object):

    def __init__(self):
        self.gen = np.zeros(1)
        self.gen_times = np.zeros(1)
        self.dem_avg = np.zeros(1)
        self.dem_10p = np.zeros(1)
        self.dem_90p = np.zeros(1)
        self.dem_times = np.zeros(1)
        self.gen_interp = None
        self.dem_avg_interp = None
        self.dem_10p_interp = None
        self.dem_90p_interp = None


    def generation_data(self, start: datetime, res: timedelta, n):
        """
        Calculates array with interpolated  average generation data for the start time,
        resolution and number of points
        :param start:
        :param res:
        :param n:
        :return:
        """
        g = np.zeros(n)
        t = start
        for i in range(n):
            #print(t.hour + t.minute/60.0)
            g[i] = self.gen_interp(t.hour + t.minute/60.0)
            t += res
            t.replace(hour=t.hour % 24)
        return g

    def consumption_average_data(self, start: datetime, res: timedelta, n):
        """
        Calculates array with interpolated  average demand data for the start time,
        resolution and number of points
        :param start:
        :param res:
        :param n:
        :return:
        """
        D = np.zeros(n)
        t = start
        for i in range(n):
            D[i] = self.dem_avg_interp(t.hour + t.minute/60.0)
            t += res
            t.replace(hour=t.hour % 24)

        return D

    def consumption_10p_data(self, start: datetime, res: timedelta, n):
        """
        Calculates array with interpolated  average demand data for the start time,
        resolution and number of points
        :param start:
        :param res:
        :param n:
        :return:
        """
        D10p = np.zeros(n)
        t = start
        for i in range(n):
            D10p[i] = self.dem_10p_interp(t.hour + t.minute/60.0)
            t += res
            t.replace(hour=t.hour % 24)

        return D10p

    def consumption_90p_data(self, start: datetime, res: timedelta, n):
        """
        Calculates array with interpolated  average demand data for the start time,
        resolution and number of points
        :param start:
        :param res:
        :param n:
        :return:
        """
        D90p = np.zeros(n)
        t = start
        for i in range(n):
            D90p[i] = self.dem_avg_interp(t.hour + t.minute/60.0)
            t += res
            t.replace(hour=t.hour % 24)
        return D90p

    def sample_demand(self, start: datetime, res: timedelta, n):
        """
        Monte Carlo simulation to obtain scenario

        :param start:
        :param res:
        :param n:
        :return:
        """
        D = np.zeros(n)
        t = start
        for i in range(n):
            hour = t.hour + t.minute/60.0
            mu = self.dem_avg_interp(hour)
            sigma = 0.5 * (self.dem_90p_interp(hour) - self.dem_10p_interp(hour))
            D[i] = np.clip(np.random.normal(loc=mu, scale=sigma), self.dem_10p_interp(hour), self.dem_90p_interp(hour))
            t += res
            t.replace(hour=t.hour % 24)
        return D

    def sample_generation(self, start: datetime, res: timedelta, n):
        """
        Monte Carlo simulation to obtain scenario

        :param start:
        :param res:
        :param n:
        :return:
        """
        h = np.zeros(n)
        t = start
        for i in range(n):
            hour = t.hour + t.minute/60.0
            mu = self.gen_interp(hour)
            sigma = 0.25 * self.gen_interp(hour)
            h[i] = np.clip(np.random.normal(loc=mu, scale=sigma), 0.0, 3.0*mu)
            t += res
            t.replace(hour=t.hour % 24)
        return h

    @classmethod
    def load(cls, path_to_data):

        with open(path_to_data) as instream:
            data = json.load(instream)

            m = Model()

            m.gen_times = np.zeros(len(data['generation']))
            m.gen = np.zeros(len(data['generation']))

            for k, entry in enumerate(data['generation']):
                m.gen_times[k] = entry['hour']
                m.gen[k] = entry['power']

            m.dem_times = np.zeros(len(data['consumption']))
            m.dem_avg = np.zeros(len(data['consumption']))
            m.dem_10p = np.zeros(len(data['consumption']))
            m.dem_90p = np.zeros(len(data['consumption']))

            for k, entry in enumerate(data['consumption']):
                m.dem_times[k] = entry['hour']
                m.dem_avg[k] = entry['avg']
                m.dem_10p[k] = entry['pct_10']
                m.dem_90p[k] = entry['pct_90']

            m.gen_interp = scipy.interpolate.interp1d(m.gen_times, m.gen)
            m.dem_avg_interp = scipy.interpolate.interp1d(m.dem_times, m.dem_avg)
            m.dem_10p_interp = scipy.interpolate.interp1d(m.dem_times, m.dem_10p)
            m.dem_90p_interp = scipy.interpolate.interp1d(m.dem_times, m.dem_90p)

            return m

class Household(object):

    def __init__(self, R_0=0.0):
        self.t = 0
        self.R_0 = R_0
        self.D = 0.0
        self.R = 0.0
        self.C = 0.0
        self.h = 0.0

    def init(self, model, current_time, time_res):
        self.D = model.sample_demand(current_time, time_res, 1)[0]
        self.h = model.sample_generation(current_time, time_res, 1)[0]
        self.C = 0.0
        self.R = self.R_0
        self.t = 0

    def tick(self, model, current_time, time_res, p, x_gb, x_sb, x_sd, x_bd, x_gd):
        """
        Ticks the simulation with
        :param x_gb:
        :param x_sb:
        :param x_sd:
        :param x_bd:
        :param x_gd:
        :return:
        """
        self.R = self.R + x_gb + x_sb - x_bd
        self.C += x_gb * p + x_gd * p
        self.D = model.sample_demand(current_time, time_res, 1)[0]
        self.h = model.sample_generation(current_time, time_res, 1)[0]
        self.t += 1