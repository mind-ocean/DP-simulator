"""
This script gathers all linked subprograms

In this ver2 we apply updated ventilation losses, mechanical efficiencies
all according to DNV ST0111 2021
"""
from thruster_loop import*
import time
from rudders import*
from rudders_new import*
from result import*
from balance import*
from Inequality_constraints_new import*
from thrust_power_TA import*
from thr_interactions import*
import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class main_TA:
    def __init__(self, num_thr, ship_name):
        self.num_thr = num_thr
        self.ship_name = ship_name

    def solver(self, thr_data,x_skeg,y_skeg,tau_control,ta_0, thr_models_data, time_step, angle):
        
        Thrust = []
        for i in range(self.num_thr):
            
            n = float(thr_models_data[i][1]) * time_step *60
            T_polynomial = [thr_models_data[i][3],thr_models_data[i][4]]
            
            max_rev = thr_models_data[i][0]
            
            a = float(T_polynomial[0])
            b = float(T_polynomial[1])
            
            
            T = a*(max_rev+n)**2+b*(max_rev+n)- a*(max_rev)**2-b*(max_rev)
            
            #print(T)
            choices1={
                'propeller FPP':'propeller',
                'propeller FPP nozzle':'propeller',
                'propeller CPP':'propeller',
                'propeller CPP nozzle':'propeller',
                'propeller CRP':'propeller',
                'Azimuth without nozzle':'azi',
                'azi':'azi',
                'azi nozzle':'azi',
                'azi CRP':'azi',
                'pod':'azi',
                'pod nozzle':'azi',
                'pod CRP':'azi',
                'tunnel':'tunnel',
                'cyclo':'azi',
                }
          
            thr_type=str(choices1.get(thr_data[i][0], 'not in the base'))
            
            if thr_type=='azi':
                
                #if angle<0:
                    #angle = angle+360
                
                Tx = T*np.cos(np.radians(angle[i]))
                Ty = T*np.sin(np.radians(angle[i]))
                
                
            elif thr_type=='tunnel':
                
                Tx = 0
                Ty = T
            
            Thrust.append(Tx)
            Thrust.append(Ty)
        
        d_max = np.absolute(Thrust)
        
        ship_name = self.ship_name
        num_thr = self.num_thr
            
        start = time.time()
        name = ship_name
        e = 0.99
        rudder_angle_step = 5
        rudder_max_angle = 30
        ro_water = 1026
       
        
        prop = propeller_rudder(thr_data, num_thr, rudder_angle_step, rudder_max_angle)
        ax, ay, at = prop.coeff()
       
        # Thrusters/skeg interactions
        interactions = thr_int(thr_data, num_thr, y_skeg, x_skeg)
        thr_thr = interactions.work_thr_thr()
        thr_dead = interactions.dead_thr_thr()
        thr_skeg = interactions.skeg_int()
        
        # Interactions function - points
        function = thr_int_points(
            thr_thr, thr_dead, thr_skeg, thr_data, num_thr, x_skeg, y_skeg)
        points = function.joined_points()
        specific_int, int_gather_all_thr = function.exist_fun()

        multi_loss = multiloss(points)
        points = multi_loss.newnew()

        # resulting points for each thruster
        function_modify = new_points(points, num_thr, int_gather_all_thr)
        thr_int_def = function_modify.all_thr_int_points()
        #thr_int_def = [[[0,0.01],[180,0.01],[314,0.01],[315,1],[316,0.01],[360,0.01]],
                       #[[0,0.01],[180,0.01],[251,0.01],[252,1],[253,0.01],[360,0.01]],[],[]] # !!!
        
        b_=tau_control
        #b_=np.array([100,100,100])
        b_=np.array(b_)
    
        
        # the rest of the losses
        beta_misc = 0.9

        # maximum thrust - nominal and effective
        thrust = thrust_power(
            thr_data, beta_misc, num_thr)
        T_nom, T_eff = thrust.thrust_val()
    
        # weight for P matrix and radius of thr - ONLY WORKING THRUSTERS WITH NO ZERO LOSSES
        weight, radius = thrust.weight(e)

        # Power coefficients
        P, q = thrust.P_matrix(weight)
        
        # constraints for the propellers
        rud_simple=rudders_new(rudder_max_angle,T_eff,thr_data,rudder_angle_step)
        G_prop, h_prop=rud_simple.con_prop()
        
        # Inequality constraints
        ineq = groups(thr_int_def, thr_data, T_eff, num_thr, e, at)
        G, h, loss = ineq.G_h_all()
        
        # joining the constraints for the porpellers and other thrusters
        join = concatenate_Gh(G_prop, h_prop, G, h, T_eff, thr_data)
        G_new, h_new, op, pp = join.combain()
        
        # Combinantions of all possible alternatives of G and h, loss and ax,ay
        combinations = clear_and_comb(G_new, h_new, loss)
        G_comb = combinations.sym_all()
        h_comb = combinations.h_prep()
        loss_comb = combinations.loss_prep()
        
        tunnel = tunnel_thr(thr_data, num_thr, G_comb, h_comb, T_eff)
        G_new, h_new, div = tunnel.new_G_new_h()
        
        
        # Equality constraints
        bal = balance(thr_data, num_thr, T_eff, b_)
        A = bal.balance_matrix()
        
        b = bal.b_array()
        all_lines = bal.add_lines(at)
        A = bal.combination_A(A, all_lines, op, pp)
        
        options = len(G_new)  # G_new
        
        #print(options)
        power = []
        sol = []
        k_res = []
        l = -1
        w = 10
        
        for k in range(options):  # for k in range(options):

            if div != 0:
                if k % div == 0:
                    l = l+1
            else:
                l = k

            G_k = G_new[k]
            h_k = h_new[k]
            A_k = A[l]
            
            loss_k = loss_comb[l]
                    
            P_k = thrust.correct_P(P, loss_k, weight, l)
            
            P_k_dT = P_k*1000
            q_dT = q
            
            q_k = np.hstack((q,q_dT))
            P_k = np.vstack((np.hstack((P_k,np.zeros_like(P_k))),np.hstack((np.zeros_like(P_k),P_k_dT))))
            
            add_A_k_down_1 = np.eye(len(ta_0))
            add_A_k_down_2 = np.eye(len(ta_0))*-1
            
            add_A_k_down = np.hstack((add_A_k_down_1,add_A_k_down_2))
            add_A_k_up = np.zeros((len(A_k),len(ta_0)))
            
            A_k = np.vstack((np.hstack((A_k,add_A_k_up)),add_A_k_down))
            b_k = np.hstack((b,ta_0))
            
            add_G_k_up = np.zeros((len(G_k),len(ta_0)))
            add_G_k_down_1 = np.zeros((len(ta_0),len(ta_0)))
            add_G_k_down_2 = np.eye(len(ta_0),len(ta_0))
            
            add_G_k_down = np.hstack((add_G_k_down_1,add_G_k_down_2))
            
            G_k = np.hstack((G_k,add_G_k_up))
            G_k = np.vstack((G_k,add_G_k_down))
            
            add_G_k_down_1 = np.zeros((len(ta_0),len(ta_0)))
            add_G_k_down_2 = np.eye(len(ta_0),len(ta_0))*-1
            
            add_G_k_down = np.hstack((add_G_k_down_1,add_G_k_down_2))
            G_k = np.vstack((G_k,add_G_k_down))
            
            h_k = np.hstack((h_k,d_max,d_max))
            
            try:
                    # CORE FUNCTION
                
                solution = solve_qp(P_k, q_k, G_k, h_k, A_k, b_k, solver='quadprog')  # CORE FUNCTION
               
                solver = 1
                
                
            except:
                
                solver = 0
                solution = None
    
            if solution is None or solver == 0:
                solver = 0
                k_res.append(0)

            else:
                solver = 1
                k_res.append(1)

            if solver == 1:
               
                res = results(solution, weight, thr_int_def, T_eff, thr_data, num_thr, np.ones(num_thr), beta_misc,
                              ax, ay, at, rudder_max_angle, rudder_angle_step, pp)
                Pb_total, Pb_list, P_max_list, uti_power = res.total_power()

                sol.append(solution)
                power.append(Pb_total)
        
        if sum(k_res) != 0:

            solver = 1
            min_power = min(power)
            k_best = power.index(min_power)
            solution = sol[k_best]
            res = results(solution, weight, thr_int_def, T_eff, thr_data, num_thr,
                          np.ones(num_thr), beta_misc, ax, ay, at, rudder_max_angle, rudder_angle_step, pp)
            Pb_total, Pb_list, P_max_list, uti_power = res.total_power()
            Tnom, Teff, Tnom_per_max, Teff_per_Tnom, beta_new, rudrud = res.postproc()
            loss_total = res.all_losses()
            Tx_rud, Ty_rud, angle_rud = res.prop_rudder()
            
            if solution is not None:
                ta = solution[:len(ta_0)]
            #print(ta)
        else:

            solver = 0
            
            ta = None

        end = time.time()
        duration = end-start
        
        if solver==0:
            message='Not capable to keep position and/or heading'
        
        else:
            message='Position and heading maintained'
        
        return duration, message, ta, solution[len(ta_0):] if solution is not None else np.zeros(num_thr*2)


#duration, message, ta = main_site(num_thr, ship_name).solver(thr_data,x_skeg,y_skeg,[-10,0,0])
#print(message)
