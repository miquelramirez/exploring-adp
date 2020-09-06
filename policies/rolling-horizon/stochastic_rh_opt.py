"""
    Example on an application of Stochastic Rolling Horizon
"""
import cvxpy as cp
import numpy as np
import json
import os
import scipy.interpolate

from datetime import datetime, timedelta
from argparse import ArgumentParser

from model import Model

def process_cmd_line():
    parser = ArgumentParser(description="Stochastic Rolling Horizon Optimization")
    parser.add_argument("--start-time", dest='start_time', type=str, help="Start time hh:mm (24 hrs)", default="12:00")
    parser.add_argument("--duration", dest='duration', type=int, help="Duration in hours", default=4)
    parser.add_argument("--resolution", dest='resolution', type=int, help="Resolution in minutes", default=30)
    parser.add_argument("--data", dest='data_path', type=str, help="Data file")
    parser.add_argument("--scenarios", dest='scenarios', type=int, default=20, help="Number of scenarios")
    parser.add_argument("--seed", dest='seed', type=int, default=1337, help="RNG seed")
    options = parser.parse_args()

    options.start_time = datetime.strptime(options.start_time, "%H:%M")
    options.duration = timedelta(hours=int(options.duration))
    options.end_time = options.start_time + options.duration
    options.resolution = timedelta(minutes=int(options.resolution))
    options.T = int(np.ceil(options.duration / options.resolution))

    if not os.path.exists(options.data_path):
        raise SystemExit("Could not open data file '{}'".format(options.data_path))

    model = Model.load(options.data_path, options)

    return options, model

class Scenario(object):

    def __init__(self, index, T, p):
        # Price is fixed
        self.index = index
        self.T = T
        self.p = np.array([p for t in range(T)])
        self.h = np.zeros(1)
        self.D = np.zeros(1)

        self.R = cp.Variable(T, name='R_{}'.format(index))
        self.x_gb = cp.Variable(T, name='x_gb_{}'.format(index))
        self.x_sb = cp.Variable(T, name='x_sb_{}'.format(index))
        self.x_sd = cp.Variable(T, name='x_sd_{}'.format(index))
        self.x_bd = cp.Variable(T, name='x_bd_{}'.format(index))
        self.x_gd = cp.Variable(T, name='x_gd_{}'.format(index))

    def constraints(self, R0=0.1):
        C = [
                self.R[0] == R0,
                # all demand needs to be met
                self.D <= self.x_gd + self.x_sd + self.x_bd,
                # battery outflow cannot be greater than battery level
                self.x_bd <= self.R,
                self.R <= 5.0,
                # solar outflow cannot be greater than generation
                self.h >= self.x_sd + self.x_sb,
                self.x_gb >= 0.0,
                self.x_sb >= 0.0,
                self.x_bd >= 0.0,
                self.x_sd >= 0.0,
                self.x_gd >= 0.0
            ] + [self.R[t + 1] == self.R[t] + self.x_gb[t] + self.x_sb[t] - self.x_bd[t]
                 for t in range(self.T - 1)]
        return C

    def objective(self):
        return self.p[1:] * self.x_gb[1:] + self.p[1:] * self.x_gd[1:]

def compute_controls(model: Model, R0, D0, h0, K, T, current_time, time_res, price):

    # Generate scenarios
    # Price of kWh
    p = price
    W = []
    for k in range(K):
        w = Scenario(k, T, p)
        w.h = model.sample_generation(current_time, time_res, T)
        w.D = model.sample_demand(current_time, time_res, T)
        #print(w.p)
        #print(w.h)
        #print(w.D)
        w.D[0] = D0
        w.h[0] = h0
        W += [w]

    # Inputs
    R = cp.Variable(2, name='R')
    x_gb = cp.Variable(name='x_gb*')
    x_sb = cp.Variable(name='x_sb*')
    x_sd = cp.Variable(name='x_sd*')
    x_bd = cp.Variable(name='x_bd*')
    x_gd = cp.Variable(name='x_gd*')

    p_w = 1.0 / len(W)

    f = x_gb * p + x_gd * p + cp.sum([p_w * w.objective() for w in W])
    #print(f)
    obj = cp.Minimize(f)

    C = [
         x_gb >= 0.0,
         x_sb >= 0.0,
         x_sd >= 0.0,
         x_bd >= 0.0,
         x_gd >= 0.0]
    for w in W:
        C += w.constraints(R0=R0)

    # Non-anticipativity constraints
    for w in W:
        C += [x_gb == w.x_gb[0],
              x_sb == w.x_sb[0],
              x_sd == w.x_sd[0],
              x_bd == w.x_bd[0],
              x_gd == w.x_gd[0]]
    #print(C)
    P = cp.Problem(obj, C)
    Q = P.solve(solver=cp.CPLEX)

    print("Status", P.status)
    if P.status not in ("infeasible", "unbounded"):
        pass
        #print("Power plan")
        #for w in W:
        #    print("h_{}".format(w.index), w.h[0])
        #    print("D_{}".format(w.index), w.D[0])
        #    print("x_bd_{}".format(w.index), w.x_bd[0].value)
        #    print("x_gd_{}".format(w.index), w.x_gd[0].value)
        #    print("x_sd_{}".format(w.index), w.x_sd[0].value)
        # print("Cost", P.value)
        # print("x_gb", x_gb.value)
        # print("x_sb", x_sb.value)
        # print("x_sd", x_sd.value)
        # print("x_bd", x_bd.value)
        # print("x_gd", x_gd.value)

    return x_gb.value, x_sb.value, x_sd.value, x_bd.value, x_gd.value

def main():

    options, model = process_cmd_line()

    np.random.seed(options.seed)

    print("Stochastic Rolling Horizon")
    print("Problem data:")
    print("\tstart time:", options.start_time)
    print("\tend time:", options.end_time)
    print("\tduration:", options.duration)
    print("\tresolution", options.resolution)
    print("\tHorizon", options.T)


    # Generate scenarios
    # Price of kWh
    p = 0.25278
    W = []
    for k in range(options.scenarios):
        w = Scenario(k, options.T, p)
        w.h = model.sample_generation(options.start_time, options.resolution, options.T)
        w.D = model.sample_demand(options.start_time, options.resolution, options.T)
        #print(w.p)
        #print(w.h)
        #print(w.D)
        W += [w]

    # Inputs
    R = cp.Variable(2, name='R')
    x_gb = cp.Variable(name='x_gb*')
    x_sb = cp.Variable(name='x_sb*')
    x_sd = cp.Variable(name='x_sd*')
    x_bd = cp.Variable(name='x_bd*')
    x_gd = cp.Variable(name='x_gd*')

    p_w = 1.0 / len(W)

    f = x_gb * p + x_gd * p + cp.sum([p_w * w.objective() for w in W])
    #print(f)
    obj = cp.Minimize(f)

    C = [
         x_gb >= 0.0,
         x_sb >= 0.0,
         x_sd >= 0.0,
         x_bd >= 0.0,
         x_gd >= 0.0]
    for w in W:
        C += w.constraints(R0=0.1)

    # Non-anticipativity constraints
    for w in W:
        C += [x_gb == w.x_gb[0],
              x_sb == w.x_sb[0],
              x_sd == w.x_sd[0],
              x_bd == w.x_bd[0],
              x_gd == w.x_gd[0]]
    #print(C)
    P = cp.Problem(obj, C)
    Q = P.solve(solver=cp.CPLEX)

    print("Status", P.status)
    if P.status not in ("infeasible", "unbounded"):
        print("Power plan")
        #for w in W:
        #    print("h_{}".format(w.index), w.h[0])
        #    print("D_{}".format(w.index), w.D[0])
        #    print("x_bd_{}".format(w.index), w.x_bd[0].value)
        #    print("x_gd_{}".format(w.index), w.x_gd[0].value)
        #    print("x_sd_{}".format(w.index), w.x_sd[0].value)
        print("Cost", P.value)
        print("x_gb", x_gb.value)
        print("x_sb", x_sb.value)
        print("x_sd", x_sd.value)
        print("x_bd", x_bd.value)
        print("x_gd", x_gd.value)




if __name__ == '__main__':
    main()