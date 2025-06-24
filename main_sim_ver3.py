import numpy as np
from model import *
from functions import *
from wave_wind_force import*
import time
from scipy import integrate
from scipy.signal import savgol_filter
from PIL import Image
import shutil
import do_mpc
from casadi import *
from casadi.tools import *
import os

# UNITS: force N, moment Nm, revolutions rpm


class simulation:
    def __init__(self, m, I, MA, DL, DNL, THR, num_thr, ship_name, thr_data, x_skeg, y_skeg, 
                 time_step, time_of_sim, buildup_duration, current_ned, wave_coeff, scale, wave_angle, sea_state,
                 wind, wind_data, kp, kd, ki,controller, mpc_horizon, mpc_time_step, MPC_boundaries, 
                 mpc_penalty, lmb , omega_0, observer_param, activate_observer, conditions):
        
        self.m = float(m)
        self.I = float(I)
        self.MA = np.array(MA,dtype='float64')
        self.DL = np.array(DL,dtype='float64')
        self.DNL = np.array(DNL,dtype='float64')
        self.THR = np.array(THR,dtype='float64')
        self.num_thr = int(num_thr)
        self.ship_name = ship_name
        self.x_skeg = np.array(x_skeg,dtype='float64')
        self.y_skeg = np.array(y_skeg,dtype='float64')
        self.time_step = float(time_step)
        self.time_of_simulation = float(time_of_sim)
        self.current_data = np.array(current_ned,dtype='float64')
        self.wave_coeff = wave_coeff
        self.wave_angle = float(wave_angle)
        self.sea_state = np.array(sea_state,dtype='float64')
        self.wind = np.array(wind,dtype='float64')
        self.wind_data = np.array(wind_data,dtype='float64')
        self.kp = np.array(kp,dtype='float64')
        self.kd = np.array(kd,dtype='float64')
        self.ki = np.array(ki,dtype='float64')
        self.thr_data = thr_data
        self.scale = scale
        self.controller = controller
        self.mpc_horizon = mpc_horizon
        self.mpc_time_step = mpc_time_step
        self.mpc_boundaries = MPC_boundaries
        self.mpc_penalty = mpc_penalty
        self.lmb =lmb
        self.omega_0 = omega_0
        self.observer_param = observer_param
        self.buildup_duration = buildup_duration
        self.activate_observer = activate_observer
        self.conditions = conditions
        
        
    def preprocessing(self):
        
        kp = np.diag(self.kp)
        kd = np.diag(self.kd)
        ki = np.diag(self.ki)
        
        hull_model_data = [self.m,self.I,
                           [self.MA[0],self.MA[1],self.MA[2]],
                           [self.DL[0],self.DL[1],self.DL[2]],
                           [self.DNL[0],self.DNL[1],self.DNL[2]]]
        
       
        return kp, kd, ki, hull_model_data,
    
    def increasement(self,bd,t, time_step):
        
        bd = bd*60
        step_increase = 1/(bd/time_step)
        
        increase = []
        for i in range(len(t)):
            
            if t[i]<bd:
                if i == 0:
                    inc = 0
                elif i>0:
                    inc += step_increase
            else:
                inc = 1
               
            increase.append(inc)
       
        return increase
        
        
    def DP_simulation(self):
        
        if os.path.exists('out/'+str(self.ship_name)):
            pass

        else:
            os.mkdir('out/'+str(self.ship_name))

        if os.path.exists('out/'+str(self.ship_name)+'/angle_'+str(self.conditions[0])+'_DP_num_'+str(self.conditions[1])):
            pass

        else:
            os.mkdir('out/'+str(self.ship_name)+'/angle_'+str(self.conditions[0])+'_DP_num_'+str(self.conditions[1]))
            
        time_step = self.time_step
        time_of_simulation = self.time_of_simulation
        
        t = np.arange(0,(time_of_simulation+self.buildup_duration)*60,time_step)
        iterations_of_sim = len(t)
        t0 = 0
        t1 = time_step*iterations_of_sim
        t = np.linspace(t0,t1,iterations_of_sim+1)
        
        # set state space matrix
        y0 = np.zeros(9)
        y = np.zeros((len(t), len(y0)))
        y[0,:] = y0
        
       
        # wind 
        wind_table = wind_matrix(self.wind_data,self.wind)
        
        # handling slow increasement
        
        increase = self.increasement(self.buildup_duration,t, time_step)
        
        # garter data
        kp, kd, ki, hull_model_data = self.preprocessing()
        
        # initiate the model class
        ship_model = model(hull_model_data,self.THR,self.num_thr, self.ship_name, self.thr_data,
                           self.x_skeg,self.y_skeg,np.array([0,0,0]),np.zeros(self.num_thr),np.array([0,0,90,90]),
                           self.time_step,np.zeros(self.num_thr),np.array([0,0,90,90]),
                           self.current_data,self.wave_coeff,self.wave_angle,self.sea_state,wind_table,self.wind_data,
                           [1,1,1],kp,kd,ki,t,0,0,0,0,self.scale, self.activate_observer,self.conditions)
        
        # initiaiting integration
        r = integrate.ode(ship_model.ode_system).set_integrator("dopri5") 
        r.set_initial_value(y0,t0) 

        # simulation loop
        start = time.time()
        
        a_arr = np.zeros(3)
        v_arr = np.zeros(3)
        eta_arr = np.zeros(3)
        speed_ned_arr = np.zeros(3)
        ned_arr = np.zeros(3)
        
        ned=np.zeros(3)
        upsi=0
       
        model_mpc = ship_model.template_model()
        mpc = ship_model.template_mpc(model_mpc,self.mpc_horizon, self.mpc_time_step, self.mpc_boundaries, self.mpc_penalty)
        mpc.x0 = np.zeros(6)
        mpc.set_initial_guess()
        
        print('Number of iterations: ',iterations_of_sim)
        for iterations in range(iterations_of_sim): #iterations_of_sim
            #my_bar.progress((iterations+1)/iterations_of_sim)
            
            actual_time_sim = iterations*time_step
            
            # Update of the force acting on the hull
            ship_model.tau_thrusters(t[iterations+1])
            
            if self.sea_state[-1] == 1:
                ship_model.tau_waves(ned[2],increase[iterations],t[iterations+1])
                if iterations == 0:
                    print('Wave engaged')

            if self.wind_data[-1] == 1:
                ship_model.tau_winds(ned[2],increase[iterations]) 
                if iterations == 0:
                    print('Wind engaged') 

            if self.current_data[-1] == 1:
                ship_model.current(increase[iterations])
                if iterations == 0:
                    print('Current engaged')
            
            if iterations%100==0:
                print('| i = ',iterations)
           
            # Integrtate
            y[iterations+1,:] = r.integrate(t[iterations+1])
            
            # Save data for plots
            a = ship_model.ode_system(t[iterations+1], y[iterations+1])[:3]
            speed_ned = ship_model.ode_system(t[iterations+1], y[iterations+1])[6:]
            
            v = y[iterations+1][:3]
            eta = y[iterations+1][3:6]
            ned = y[iterations+1][6:]
            
            radius = (ned[0]**2 + ned[1]**2)**0.5
            psi = abs(ned[2])
            
            if radius>5/self.scale or psi>3:
                print('Position lost in ', round((iterations*time_step)/60,1), 'minutes at r = ',round(radius,2),' and psi = ',round(psi))
                dp = False
                break
            else:
                dp = True
            # Observer
            ship_model.observer(v, ned, self.lmb,self.omega_0,self.observer_param)
            
            # PID control
            ship_model.rotation(eta[2])
            #ship_model.PID(v,ned)
            
            if self.controller == 'PID':
                ship_model.PID_with_observer()
            
            else:
                ship_model.mpc_(mpc)
            
            ship_model.objective_function_calc(self.mpc_penalty)
            
            
            a_arr = np.vstack((a_arr,a))
            v_arr = np.vstack((v_arr,v))
            eta_arr = np.vstack((eta_arr,eta))
            speed_ned_arr = np.vstack((speed_ned_arr,speed_ned))
            ned_arr = np.vstack((ned_arr,ned))
    
       
        
        end=time.time()
        time_of_calc = end-start
        time_message = convert(time_of_calc)
        
        
            # ----------plots-------------------------------------------------------
        #ship_model.thr_control_plots() # actual revolutions and angle of the thrusters
        ship_model.state_control_plots(a_arr,v_arr,eta_arr) # state variables
        ship_model.NED_plots(ned_arr) # position in NED
        ship_model.bias_plots()
        
        if self.sea_state[-1]!=0:
            ship_model.plot_wave_force()
        
        ship_model.tau_plot() # tau from PID and tau from thrusters due to TA, thrusters models and location matrix
        ship_model.ta_plots() # forces on each thruster TA vs. forces - check if thrusters models work fine
        ship_model.input_thr_plots()
        ship_model.current_plot()
        #if self.wind_data[4] != 0:
            #ship_model.wind_wave_plots()
        
        return dp
        
