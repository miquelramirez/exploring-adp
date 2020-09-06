"""
    Example on an application of Deterministic Rolling Horizon
"""
import cvxpy as cp
import numpy as np
import os

from datetime import datetime, timedelta
from argparse import ArgumentParser

from model import Model

def process_cmd_line():
    parser = ArgumentParser(description="Deterministic Rolling Horizon Optimization")
    parser.add_argument("--start-time", dest='start_time', type=str, help="Start time hh:mm (24 hrs)", default="12:00")
    parser.add_argument("--duration", dest='duration', type=int, help="Duration in hours", default=4)
    parser.add_argument("--resolution", dest='resolution', type=int, help="Resolution in minutes", default=30)
    parser.add_argument("--data", dest='data_path', type=str, help="Data file")
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

def compute_controls(model: Model, R0, D0, h0, T, current_time, time_res, price):

    # State variables
    R = cp.Variable(T, name='R')
    # Determinized
    # price of power from the grid at time t
    p = np.zeros(T)
    # solar panel production at time t
    h = np.zeros(T)
    # building demand at time t
    D = np.zeros(T)

    # fix values to average from randomly chosen power bill
    # solar panels power: 3.2 kWh
    # price per kWh, 0.25278 AUD
    # demand: 7.7 kWh

    for t in range(T):
        p[t] = price

    h = model.generation_data(current_time, time_res, T)
    h[0] = h0
    D = model.consumption_average_data(current_time, time_res, T)
    D[0] = D0

    # Inputs
    x_gb = cp.Variable(T, name='x_gb')
    x_sb = cp.Variable(T, name='x_sb')
    x_sd = cp.Variable(T, name='x_sd')
    x_bd = cp.Variable(T, name='x_bd')
    x_gd = cp.Variable(T, name='x_gd')

    f = cp.Minimize(x_gb * p + x_gd * p)

    C = [
        R[0] == R0,
        # all demand needs to be met, but we can generate excess power
        D <= x_gd + x_sd + x_bd,
        # battery outflow cannot be greater than battery level
        x_bd <= R,
        R <= 5.0,
        # solar outflow cannot be greater than generation
        h >= x_sd + x_sb,
        x_gb >= 0.0,
        x_sb >= 0.0,
        x_bd >= 0.0,
        x_sd >= 0.0,
        x_gd >= 0.0
    ] + [R[t+1] == R[t] + x_gb[t] + x_sb[t] - x_bd[t] for t in range(T-1)]

    P = cp.Problem(f, C)
    Q = P.solve(solver=cp.CPLEX)

    print("Status", P.status)
    if P.status not in ("infeasible", "unbounded"):
        pass
        # print("Power plan")
        # print("Cost", P.value)
        # print("x_gb", x_gb[0].value)
        # print("x_sb", x_sb[0].value)
        # print("x_sd", x_sd[0].value)
        # print("x_bd", x_bd[0].value)
        # print("x_gd", x_gd[0].value)
        # print("x_gb", [x_gb[t].value for t in range(options.T)])
        # print("x_sb", [x_sb[t].value for t in range(options.T)])
        # print("x_sd", [x_sd[t].value for t in range(options.T)])
        # print("x_bd", [x_bd[t].value for t in range(options.T)])
        # print("x_gd", [x_gd[t].value for t in range(options.T)])

    return x_gb[0].value, x_sb[0].value, x_sd[0].value, x_bd[0].value, x_gd[0].value

def main():

    options, model = process_cmd_line()

    print("Deterministic Rolling Horizon")
    print("Problem data:")
    print("\tstart time:", options.start_time)
    print("\tend time:", options.end_time)
    print("\tduration:", options.duration)
    print("\tresolution", options.resolution)
    print("\tHorizon", options.T)

    # State variables
    R = cp.Variable(options.T, name='R')
    # Determinized
    # price of power from the grid at time t
    p = np.zeros(options.T)
    # solar panel production at time t
    h = np.zeros(options.T)
    # building demand at time t
    D = np.zeros(options.T)

    # fix values to average from randomly chosen power bill
    # solar panels power: 3.2 kWh
    # price per kWh, 0.25278 AUD
    # demand: 7.7 kWh

    for t in range(options.T):
        p[t] = 0.25278

    h = model.generation_data(options.start_time, options.resolution, options.T)
    D = model.consumption_average_data(options.start_time, options.resolution, options.T)

    print("h:", h)
    print("p:", p)
    print("D:", D)

    # Inputs
    x_gb = cp.Variable(options.T, name='x_gb')
    x_sb = cp.Variable(options.T, name='x_sb')
    x_sd = cp.Variable(options.T, name='x_sd')
    x_bd = cp.Variable(options.T, name='x_bd')
    x_gd = cp.Variable(options.T, name='x_gd')

    f = cp.Minimize(x_gb * p + x_gd * p)

    C = [
        R[0] == 0.1,
        # all demand needs to be met, but we can generate excess power
        D <= x_gd + x_sd + x_bd,
        # battery outflow cannot be greater than battery level
        x_bd <= R,
        R <= 5.0,
        # solar outflow cannot be greater than generation
        h >= x_sd + x_sb,
        x_gb >= 0.0,
        x_sb >= 0.0,
        x_bd >= 0.0,
        x_sd >= 0.0,
        x_gd >= 0.0
    ] + [R[t+1] == R[t] + x_gb[t] + x_sb[t] - x_bd[t] for t in range(options.T-1)]

    P = cp.Problem(f, C)
    Q = P.solve(solver=cp.CPLEX)

    print("Status", P.status)
    if P.status not in ("infeasible", "unbounded"):
        print("Power plan")
        print("Cost", P.value)
        print("x_gb", x_gb[0].value)
        print("x_sb", x_sb[0].value)
        print("x_sd", x_sd[0].value)
        print("x_bd", x_bd[0].value)
        print("x_gd", x_gd[0].value)
        # print("x_gb", [x_gb[t].value for t in range(options.T)])
        # print("x_sb", [x_sb[t].value for t in range(options.T)])
        # print("x_sd", [x_sd[t].value for t in range(options.T)])
        # print("x_bd", [x_bd[t].value for t in range(options.T)])
        # print("x_gd", [x_gd[t].value for t in range(options.T)])




if __name__ == '__main__':
    main()