
"""
Here specify input

scale - if a full scale ship set to 1 if a ship model is scale specify that scale
and then specify all data for the model in that scale

input_file_name - the name of the file is always 'input.csv' and is always output from the 
level 1 calculation. It is necessary for Thrust Alocation alghorithm. 
Include that file in the main directory

m - mass 
I - Moment of inertia (Izz)
M_A - added mass
D_L - linear damping matrix
D_NL - nonlinear damping matrix
thrusters_models - match the maximum revolutions and the model 
to achive the same maximum power as in level 1 or redo level one with the 
matching power. This is IMPORTANT!. Remeber to include that many items
in the list as many thrusters there are in level 1

NOTE: level 1 and level 3 must match exactly

wind_file - always names 'wind.csv' and in the main directory. As result of 
wind tunnel model tests or as similar ship or CFD. The provided coefficient 
must be nondimentional and calculated as given in manual in docs folder

AF_wind - frontal projected area of the above water part of the ship
AL_wind - lateral projected area ------||------
Lpp - Lenght between perpendicurs

wave_file_aqwa - this is a specifically constructed output file from
wave loads simulation in aqwa software - full scale! Could be other 
definistion of loads (RAO and QTF) but then module aqwa_read must be replaced!

controller - Choose 'MPC' or 'PID'
 
PID_gains - columns - Kp, Kp, kd, ki
MPC_horison - a prediction horison for MPC optimization
MPC_time_step - a timestem over an optimization (prediction) in a horison
MPC_boundaries - MPC constraints (1st column: forces, 2nd column: position,
                                  3rd column: speed)

observer_gains - self explanatory (1st row: bias T, 2nd row: K3,
                                  3rd row: K4), columns : X, Y, PSI

lmb - damping coefficient for wave frequency filtering
omega_0 - other coefficient to set the filtering of the wave frequency

duration - time in minutes of simulation after conditions buildup and within  stable weather conditions
time_step - time step in secondsof the simulation taking into account thrusters and hull dynamics

buildup_duration - time in minutes of a buildup of environmental conditions chosen for the most sevier 
conditions for DP number 11. As the wind is usually the highest force, that should be determinand.
The rest of conditons buidup time would be calculated from this parameter.

loop_all - True if a wchole DP plot are to be calculated, False if only chasen pairs of
(angle, DP_num) are to be run

NOTE: On a standard comuter (Apple M! 8GB) a 3 minutes simulation takes 7 minutes in total 
and in addision computer is overload (using multile cores might help)
hence start with one simulation (one conditions), use multipole cores and consider
supercomputer for the whole loop. Other solution is code optimization.
"""

from functions import ship_data_preparation
from thruster_loop import prepare_matrix_thr
from main_sim_ver3 import simulation

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import multiprocessing

scale = 36

input_file_name = 'input.csv' 

m = 239
I = 132

M_A = {
       'X' : [13,0,0],
       'Y' : [0,157,24],
       'PSI' :[0,24,72]
       }

D_L = {
       'X' : [-1.457,0,0],
       'Y' : [0,-1.333,0.033],
       'PSI' :[0,-4.502,-0.061]
       }

D_NL = {
       'X' : [-5.067,0,0],
       'Y' : [0,-145.613,-0.034],
       'PSI' :[0,67.308,-0.067]
       }

thrusters_models = {
    'maximum RPM': [1200,1200,1200,1200], 
    'RPS mean rate': [6,6,6,6],
    'rotation mean rate [deg/s]': [90,90,90,90],
    'quadratic component': [8.63215905e-07, 8.63215905e-07, 2.20439983e-06, 2.1205369703e-06 ], 
    'linear component': [0,0,0,0]    
    }


wind_file = 'wind.csv'

AF_wind = 0.47
AL_wind = 1.39
Lpp = 2.74


wave_file_aqwa = 'ANALYSIS.LIS'

controller = 'MPC'

PID_gains = {
    'X': [80,70,0.3], 
    'Y': [80,100,0.3],
    'PSI': [80,100,0.1]
    }

MPC_horizon = 20
MPC_time_step = 0.5
MPC_boundaries = {
    'X': [40,0.2,0.3], 
    'Y': [70,0.2,0.3],
    'PSI': [70,3,2]
    }

MPC_penalty = {
    'X': 0.1, 
    'Y': 0.5,
    'PSI': 0.1
    }

observer_gains = {
    'Bias': [1000,1000,1000], 
    'K3': [700,1100,1100],
    'K4': [2000,2000,2000]
    }

activate_observer = False

lmb = 0.3
omega_0 = 5

duration = 3
time_step = 0.05

buildup_duration =  3
engage_buildup_calculator = True

loop_all = False
if loop_all:
    angles = range(0, 360, 10)
    DP_numbers = range(1, 12, 1)
    
else:
    angles = [90]
    DP_numbers = [1]

run_simulation = True


"""_________________________________________"""
"""Recalcalculation of some data and imports"""

conditions = [(angle, DP_num) for angle in angles for DP_num in DP_numbers]
    
conditions_parameters = {
    'DP number' : [1,2,3,4,5,6,7,8,9,10,11],
    'wind speed' : [1.5,3.4,5.4,7.9,10.7,13.8,17.1,20.7,24.4,28.4,32.6] ,
    'wave height' : [0.1,0.4,0.8,1.3,2.1,3.1,4.2,5.7,7.4,9.5,12.1],
    'wave period' : [3.5,4.5,5.5,6.5,7.5,8.5,9,10,10.5,11.5,12],
    'current speed' :[0.25,0.5,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75] 
    }

condition_parameters_scaled = {}
for key, value in conditions_parameters.items():
    
    if key == 'wind speed' or key == 'current speed' or key == 'wave period':
        scale_parameter = 1/scale**0.5
    elif key == 'wave height':
        scale_parameter = 1/scale
    else:
        scale_parameter = 1
    
    list_scaled_parameters = [round(parameter * scale_parameter,3) for parameter in value]
    condition_parameters_scaled[key] = list_scaled_parameters

df = pd.read_csv(input_file_name)
ship_data = df.values.T[1]

x_skeg, y_skeg, thrusters, num_thr, ship_name = ship_data_preparation(ship_data).to_thr()
thr_data, name = prepare_matrix_thr(thrusters).thr_data_m()

THR = np.transpose(list(thrusters_models.values()))

df = pd.read_csv(wind_file, sep=';')
wind = df.values.T

df = pd.read_csv(wave_file_aqwa, sep='\t')
wave_coeff = df

PID = np.transpose(list(PID_gains.values()))
observer_param = np.transpose(list(observer_gains.values()))

#---------------------------------------------------------------PROGRAM STARTS HERE---------------------------------------------------------------
def sim():
    if run_simulation:
        
        results = []
        for (angle, DP_number) in conditions:
            
            print('... initiated for angle = ', angle, ' and DP number = ', DP_number)
            table_item = condition_parameters_scaled['DP number'].index(DP_number)
            
            sea_state = [condition_parameters_scaled['wave height'][table_item],
                         condition_parameters_scaled['wave period'][table_item],
                         3.3, 0, 1]
            wind_data = [AF_wind,AL_wind,Lpp,angle,condition_parameters_scaled['wind speed'][table_item],0,1]
            
            current_data = [angle, condition_parameters_scaled['current speed'][table_item],1]
            
            print('Current is ', current_data[1], ' m/s at angle ', current_data[0])
            print('Wind is ', wind_data[4], ' m/s at angle ', wind_data[3])
            print('Sea state is ', sea_state[0], ' m wave height and ', sea_state[1], ' s wave period')
            print('----------------------------------------------------------------------')
            if engage_buildup_calculator:
                buildup_duration_final = buildup_duration * (1 + abs(np.sin(np.radians(angle))) + (condition_parameters_scaled['wind speed'][table_item]/condition_parameters_scaled['wind speed'][-1])**2)
            else:
                buildup_duration_final = buildup_duration

            print('Total time is ', round(duration+buildup_duration_final,2), ' minutes')
            
            condtions_DP_angle = [angle, DP_number]
            try:
            
                dp = simulation(m, I, list(M_A.values()), list(D_L.values()), list(D_NL.values()), THR, num_thr, ship_name, 
                        thr_data, x_skeg, y_skeg, time_step, duration, buildup_duration_final, current_data, 
                        wave_coeff, scale, angle, sea_state, wind, wind_data, 
                        PID[0], PID[1], PID[2],controller, float(MPC_horizon), float(MPC_time_step), 
                        np.transpose(list(MPC_boundaries.values())), list(MPC_penalty.values()), lmb , omega_0, observer_param, activate_observer, condtions_DP_angle).DP_simulation()
           
            except:
                
                print('Something went wrong... check input')
                dp = None
            
            result = [angle, DP_number, dp]
            results.append(result)

        df = pd.DataFrame(results, columns=['Angle', 'DP number', 'Simulation Result'])
        df.to_csv('out/DP_capability_plot_data.csv', index=False)



if __name__ == "__main__":
    sim()