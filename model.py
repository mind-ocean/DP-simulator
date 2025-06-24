import numpy as np
from scipy import integrate
from thrusters_models import *
from main_TA import *
from aqwa_read import *
from scipy.signal import savgol_filter
from functions import *
import pandas as pd
from scipy.interpolate import interp2d
import random
from casadi import *
from casadi.tools import *
import do_mpc
#import control.TransferFunction as tf

class model:
    def __init__(self,hull_model_data,thr_models_data,num_thr, ship_name, thr_data,x_skeg,
                 y_skeg,tau_control,rev_actual,angle_actual,time_step,rev_arr,angle_arr,
                 current_ned,wave_coeff,dirr_wave,sea_state,wind_table,wind_data,rates,
                 kp,kd,ki,t,K_est,wave_1st,white_noise,T,scale, activate_observer, conditions):
        
        self.hull_model_data = hull_model_data
        self.thr_models_data = thr_models_data
        self.num_thr = num_thr
        self.ship_name = ship_name
        self.thr_data = thr_data
        self.x_skeg = x_skeg
        self.y_skeg = y_skeg
        self.tau_control = tau_control
        self.rev_actual = rev_actual
        self.angle_actual = angle_actual
        self.time_step = time_step
        self.rev_arr = rev_arr
        self.angle_arr = angle_arr
        self.time_ = t
        self.tau_thr = np.zeros(3)
        self.t = 0
        self.x = np.zeros(3)
        self.tau_arr = np.zeros(3)
        self.current_ned = current_ned
        self.wave_coeff = wave_coeff
        self.dirr_wave = dirr_wave
        self.tau_wave = np.zeros(3)
        self.tau_slow = np.zeros(3)
        self.tau_wind = np.zeros(3)
        self.sea_state = sea_state
        self.wind_data = wind_data
        self.wind_table = wind_table
        self.wave_f_arr = np.zeros(3)
        self.wind_f_arr = np.zeros(3)
        self.v_current=0
        self.vc=np.zeros(3)
        self.rates = rates
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.tau_pid = np.zeros(3)
        self.eta_ned_arr = np.zeros(3)
        self.tau_control_arr = np.zeros(3)
        self.eta_ref = np.zeros(3)
        self.vel_ref = np.zeros(3)
        self.R = np.vstack(([[np.cos(0),-np.sin(0),0],[np.sin(0),np.cos(0),0],[0,0,1]])) 
        self.ta_res_arr = np.zeros(8)
        self.rev_ang_arr = np.zeros(2*self.num_thr)
        self.thr_force_arr = np.zeros(2*self.num_thr)
        self.mean_force_on_the_hull = np.zeros(3)
        self.vc_arr = np.zeros(3)
        self.time_arr = t
        self.tau = np.zeros(3)
        self.eta_est = np.zeros(3)
        self.vel_est = np.zeros(3)
        self.dvel_est = np.zeros(3)
        self.vel_est_arr = np.zeros(3)
        self.deta_est = np.zeros(3)
        self.ksi_est = np.zeros(6)
        self.bias = np.zeros(3)
        self.bias_est = np.zeros(3)
        self.y_est = np.zeros(3)
        self.dzeta_est = np.zeros(3)
        self.eta_w_est = np.zeros(3)
        self.ddzeta_est = np.zeros(3)
        self.deta_w_est = np.zeros(3)
        self.eta_est_arr = np.zeros(3)
        self.dbias_est = np.zeros(3)
        self.ta = np.zeros(2*self.num_thr)
        self.a = 0
        self.delta_tau_control = np.zeros(3)
        self.f_arr = np.array([])
        self.counter = 0
        self.conditions = conditions
        self.dir = 'out/'+str(self.ship_name)+'/angle_'+str(self.conditions[0])+'_DP_num_'+str(self.conditions[1])

        """
        k1 = np.array([])
        k2 = np.diag([2.8,2.8,1.8])
        k3 = np.diag([0.3,0.3,0.002])
        k4 = np.diag([0.1,0.1,0.1])
        """
        
        self.K_est = K_est
        self.T = T
        
        self.wc=0
        
        self.wave_1st = wave_1st
        self.eta_real = np.zeros(3)
        self.white_noise = white_noise
        
        self.NED_est = np.zeros(3)
        self.ned_est = np.zeros(3)
        self.scale = scale
        self.wave_F = np.zeros(3)
        self.time_actual = []
        self.bias_arr = np.zeros(3)
        self.horizon = 7
        
        self.delta_ta_arr = np.zeros(2*self.num_thr)
        
        self.activate_observer = activate_observer

    def matrices(self):
        
        hull_model_data = self.hull_model_data
        
        mass = hull_model_data[0]
        Inertia = hull_model_data[1]
        Added_mass = hull_model_data[2]
        Linear_damping = hull_model_data[3]
        Quadratic_damping = hull_model_data[4]
        
        MA = np.vstack((Added_mass))
        DL = np.vstack((Linear_damping))
        DNL = np.vstack((Quadratic_damping))
        MRB = np.vstack(([[mass,0,0],[0,mass,0],[0,0,Inertia]]))
        M = MA+MRB
        
        return M, DL, DNL
    
    def rotation(self,psi):
        
        psi = np.radians(psi)
        R = [[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]]
        self.R = np.vstack((R))
        
    def current_plot(self):
        
        fig,ax = plt.subplots(3,figsize=(4,7))
        fig.suptitle('Current speed in NED')

        num_cut = int(abs((len(self.time_) - len(self.vc_arr.T[0]))))
        if len(self.time_)>len(self.vc_arr.T[0]):
            ti_ = self.time_[:-(num_cut)]
        elif len(self.time_)<len(self.vc_arr.T[0]):
            ti_ = self.time_.append(self.time_[-(num_cut)])
        else:
            ti_ = self.time_

        for i in range(3):
            ax[i].plot(ti_,self.vc_arr.T[i],c='teal',alpha=0.7)

        print(self.dir)
        fig.savefig(self.dir+'/current.png',dpi = 200)
        
    def thr_control_plots(self):
        
        rev = self.rev_arr.T
        ang = self.angle_arr.T
        time_ = self.time_
        
        figure, ax = plt.subplots(self.num_thr,2,figsize=(6,10))
        figure.tight_layout()
        figure.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
        figure.suptitle('Thrusters actual revolutions and angle in time')
        for i in range(len(self.num_thr)):
            ax[i,0].plot(time_,rev[i][1:],c='deeppink')
            ax[i,1].plot(time_,ang[i][1:],c='mediumaquamarine')
            
            #ax[i,0].grid(linestyle='--',c='silver')
            #ax[i,1].grid(linestyle='--',c='silver')
            
            ax[i,0].set_ylabel('RPM')
            ax[i,1].set_ylabel('Angle')
            
            ax[i,0].set_title('thr '+str(i))
            ax[i,1].set_title('thr '+str(i))
        
        figure.savefig(self.dir+'/thrusters.png',dpi = 200)
        
    
    def bias_plots(self):
        
        figure, ax = plt.subplots(3,1,figsize=(8,8))
        figure.tight_layout()
        figure.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
        
        figure.suptitle('Bias')
        
        ti = np.append(0,self.time_actual)
        label = ['X component [N]','Y component [N]','$\u03C8$ component [Nm]']
        
        if len(ti)>len(self.bias_arr.T[0]):
            ti = ti[:-1]
        elif len(ti)<len(self.bias_arr.T[0]):
            ti = ti.append(ti[-1])
        else:
            pass
        
        for i in range(3):
            
            ax[i].plot(ti, self.bias_arr.T[i], c = 'teal')
            ax[i].set_ylabel(label[i])
            ax[i].set_xlabel('time [s]')
    
        figure.savefig(self.dir+'/bias.png',dpi=200)
    
    def wind_wave_plots(self):
        
        figure, ax = plt.subplots(3,2,figsize=(8,8))
        figure.tight_layout()
        figure.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
        
        figure.suptitle('Forces acting on a hull, \n generated by the wind and waves')
        
        
        c1='deepskyblue'
        c2='teal'
        for i in range(3):
            for j in range(2):
                
                if j==0:
                    ax[i,j].plot(self.time_,self.wind_f_arr.T[i],c=c1,label='wind')
                else:
                    
                    if self.sea_state[0] != 0:
                        ax[i,j].plot(self.time_,self.wave_f_arr.T[i],c=c2,label='wave')
                
                ax[i,j].legend()
                ax[i,j].set_xlabel('time [s]')
                
                if i<2:
                    ax[i,j].set_ylabel('Force [N]')
                    
                else:
                    ax[i,j].set_ylabel('Moment [N]')
        
        figure.savefig(self.dir+'/wind_waves.png',dpi=200)
               
    def NED_plots(self,ned):
        
        NED_est = self.NED_est
        
        ned_df = pd.DataFrame(ned)
        ned_df.to_csv(self.dir+'/NED.csv')
        
        # footprint
        figure, ax = plt.subplots(2,figsize=(6,6))
        figure.tight_layout()
        figure.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.8)
        
        ax[0].set_title('Ship position in NED throughout the simulation')
        ned = np.vstack((ned)).T
        ti = np.append(0,self.time_actual)
        #ax.set_aspect('equal', 'box')
        
        if len(ti)>len(ned[2]):
            ti = ti[:-1]
        elif len(ti)<len(ned[2]):
            ti = ti.append(ti[-1])
        else:
            pass
        
        ax[0].plot(ned[0],ned[1],c='deepskyblue',label = 'ODEint',alpha = 0.6)     
        ax[0].plot(NED_est.T[0],NED_est.T[1],c='deeppink',label = 'Observer', alpha = 0.6, linestyle = '--') 
        ax[0].scatter(0,0,marker='*',color='goldenrod',s=60)
        
        ax[0].legend()
        ax[0].set_ylabel('Position y [m]')
        ax[0].set_xlabel('Position x [m]')
        
        ax[1].plot(ti,ned[2],c='deepskyblue',label = 'ODEint')
        ax[1].plot(ti,NED_est.T[2],c='deeppink',label = 'Observer')

        ax[1].legend()
        ax[1].set_ylabel('Angle [deg]')
        ax[1].set_xlabel('Time [sec]')
        
        figure.savefig(self.dir+'/NED.png',dpi = 300)
    
    def tau_plot(self):
        
        tau_t = self.tau_arr.T
        ti = self.time_
        
        
        figure, ax = plt.subplots(3,1,figsize=(5,8))
        figure.tight_layout()
        figure.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.8)
        figure.suptitle('Diagnosis of the TA and thrusters models functionality')
        
        ylabel = ['Tx [N]','Ty [N]','Nz [Nm]']
        for i in range(3):
            
            num_cut = int(abs((len(ti) - len(tau_t[0]))))
            if len(ti)>len(tau_t[0]):
                ti_ = ti[:-(num_cut)]
            elif len(ti)<len(tau_t[0]):
                ti_ = ti.append(ti[-(num_cut)])
            else:
                ti_ = ti
        

            ax[i].plot(ti_,tau_t[i],c='deepskyblue',label='Actual force on the hull due to thrusters',alpha=0.7)
            
            num_cut = int(abs((len(ti) - len(self.tau_control_arr.T[0]))))
            if len(ti)>len(self.tau_control_arr.T[0]):
                ti_ = ti[:-(num_cut)]
            elif len(ti)<len(self.tau_control_arr.T[0]):
                ti_ = ti.append(ti[-(num_cut)])
            else:
                ti_ = ti
        
            ax[i].plot(ti_,self.tau_control_arr.T[i],c='deeppink',linestyle='-',label='Set control from PID',alpha=0.7)
            
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(ylabel[i])
            ax[i].legend()
            
            
        figure.savefig(self.dir+'/tau_to_hull.png',dpi=200)
                   
    def state_control_plots(self,a_arr,v_arr,eta_arr):
        
        ti = np.append(0,self.time_actual)
        
        figure, ax = plt.subplots(3,3,figsize=(10,10))
        figure.tight_layout()
        figure.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
        figure.suptitle('BODY fixed frame')
        c = ['mediumaquamarine','mediumslateblue','palevioletred']
        l = ['x component','y component','$\psi$ component']
        
        if len(ti)>len(a_arr.T[0]):
            ti = ti[:-1]
        elif len(ti)<len(a_arr.T[0]):
            ti = ti.append(ti[-1])
        else:
            pass
            
        for i in range(3):
            ax[0,i].plot(ti,a_arr.T[i],c=c[0],label=l[i])
            ax[1,i].plot(ti,v_arr.T[i],c=c[1],label=l[i])
            ax[1,i].plot(ti,self.vel_est_arr.T[i],c='black', linestyle = '--',label='Est.')
            ax[2,i].plot(ti,eta_arr.T[i],c=c[2],label=l[i])
            ax[2,i].plot(ti,self.eta_est_arr.T[i],c='black', linestyle = '--',label='Est.')
            
            #ax[0,i].grid(linestyle='--',c='silver')
            #ax[1,i].grid(linestyle='--',c='silver')
            #ax[2,i].grid(linestyle='--',c='silver')
            
            ax[0,i].legend()
            ax[1,i].legend()
            ax[2,i].legend()
            
            for j in range(3):
                ax[i,j].set_xlabel('time [s]')
            
            if i==2:
                ax[2,i].set_ylabel('Position [deg]')
                ax[1,i].set_ylabel('Speed [deg/s]')
                ax[0,i].set_ylabel('Acceleration [deg/$s^2$]')
            else:
                ax[2,i].set_ylabel('Position [m]')
                ax[1,i].set_ylabel('Speed [m/s]')
                ax[0,i].set_ylabel('Acceleration [m/$s^2$]')
        
        figure.savefig(self.dir+'/state.png',dpi=250)
        
        f_df = pd.DataFrame(self.f_arr)
        f_df.to_csv(self.dir+'/objective_f.csv')
        
    def tau_waves(self,psi_ned,inc,t):
        
        alpha_wave = self.dirr_wave - psi_ned
        scale = float(self.scale)
        
        Ist_order, IInd_order = aqwa_read(self.wave_coeff).read_basic_data()
        
        Omega = np.array(IInd_order['Omega'], dtype = 'float64')*scale**0.5
        Angles = np.array(IInd_order['Angles'], dtype = 'float64')
        
        self.wc = Omega[-1]
        
        N = 200
        omega_arr = np.linspace(Omega[0],Omega[-1],N)
        d_omega = omega_arr[1] - omega_arr[0]
        
        # 2nd ORDER WAVE DRIFT FORCES
        f_x = interp2d(Angles, Omega, np.array(IInd_order['X'], dtype='float64'))
        f_y = interp2d(Angles, Omega, np.array(IInd_order['Y'], dtype='float64'))
        f_z = interp2d(Angles, Omega, np.array(IInd_order['RZ'], dtype='float64'))

        Hs = self.sea_state[0]
        Tp = self.sea_state[1]
        gamma = self.sea_state[2]

        X = 0
        Y = 0
        Z = 0
        
        if scale > 1:
            a = 998.6/1025
        else:
            a = 1

        for k in range(N):
            S_k = spectra(omega_arr[k],Hs,Tp,gamma)
            A_k = (2*S_k*d_omega)**0.5
            
            rao_x = f_x(alpha_wave,omega_arr[k])/scale*a
            rao_y = f_y(alpha_wave,omega_arr[k])/scale*a
            rao_z = f_z(alpha_wave,omega_arr[k])/scale**2*a
           
            X += rao_x*A_k**2
            Y += rao_y*A_k**2
            Z += rao_z*A_k**2
            
        tau_2nd = np.array([X,Y,Z])
        
    
        # 1st ORDER WAVE FRUDE_KRYLOW + DYFRCTION FORCES
        
        f_x_amp = interp2d(Angles, Omega, np.array(Ist_order['X'][0], dtype='float64'))
        f_y_amp = interp2d(Angles, Omega, np.array(Ist_order['Y'][0], dtype='float64'))
        f_z_amp = interp2d(Angles, Omega, np.array(Ist_order['RZ'][0], dtype='float64'))
        
        f_x_pha = interp2d(Angles, Omega, np.array(Ist_order['X'][1], dtype='float64'))
        f_y_pha = interp2d(Angles, Omega, np.array(Ist_order['Y'][1], dtype='float64'))
        f_z_pha = interp2d(Angles, Omega, np.array(Ist_order['RZ'][1], dtype='float64'))
        
        np.random.seed(99)
        random.seed(30)
        p = (1-2*np.random.random(N))*np.pi
        
        X = 0
        Y = 0
        Z = 0
        
        for k in range(N):
            S_k = spectra(omega_arr[k],Hs,Tp,gamma)
            A_k = (2*S_k*d_omega)**0.5
            
            amp_x = f_x_amp(alpha_wave,omega_arr[k])/scale**2*a
            amp_y = f_y_amp(alpha_wave,omega_arr[k])/scale**2*a
            amp_z = f_z_amp(alpha_wave,omega_arr[k])/scale**3*a
            
            pha_x = f_x_pha(alpha_wave,omega_arr[k])
            pha_y = f_y_pha(alpha_wave,omega_arr[k])
            pha_z = f_z_pha(alpha_wave,omega_arr[k])
            
            X += amp_x*A_k*np.cos(omega_arr[k]*t + pha_x*np.pi/180 + p[k])
            Y += amp_y*A_k*np.cos(omega_arr[k]*t + pha_y*np.pi/180 + p[k])
            Z += amp_z*A_k*np.cos(omega_arr[k]*t + pha_z*np.pi/180 + p[k])
            
        tau_1st = np.array([X,Y,Z])
        #print(tau_2nd)
        if self.activate_observer:
            self.tau_wave = inc * np.hstack((tau_1st + tau_2nd))
        else:
            self.tau_wave = inc * np.hstack((tau_2nd))

        #self.tau_slow = np.hstack((tau_2nd))
        #print(self.tau_slow)
        self.wave_F = np.vstack((self.wave_F,self.tau_wave))
    
    def plot_wave_force(self):
        
        figure, ax = plt.subplots(3,figsize=(10,10))
        figure.tight_layout()
        figure.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
        figure.suptitle('Wave force in time')
        c = ['mediumaquamarine','mediumslateblue','palevioletred']
        l = ['x component','y component','$\psi$ component']
        
        ti = np.append(0,self.time_actual)
        
        for i in range(3):
            ax[i].plot(ti, self.wave_F.T[i],c = c[i])
            ax[i].set_title(l[i])
            ax[i].set_xlabel('time [s]')
            
        figure.savefig(self.dir+'/wave_force.png',dpi=250)
            
        
    def tau_winds(self,psi_ned,inc):
        
        dirr_wind = self.wind_data[3]
        
        alpha_wind = dirr_wind - psi_ned
        
        if alpha_wind<0:
            alpha_wind = 360 + alpha_wind
            
        for i in range(3):
            self.tau_wind[i] = round(inc*np.interp(alpha_wind,self.wind_table[0],self.wind_table[i+1]),3)
            
        self.wind_f_arr = np.vstack((self.wind_f_arr,self.tau_wind))
    
    def current(self,inc):
        
        alpha_c = self.current_ned[0]
        
        self.v_current = inc*self.current_ned[1]
        
        vcx = round(np.cos(np.radians(alpha_c))*self.v_current,3)
        vcy = round(np.sin(np.radians(alpha_c))*self.v_current,3)
        
        self.vc = np.array([vcx,vcy,0])
        self.vc_arr = np.vstack((self.vc_arr,self.vc))
    
    def ta_plots(self):
        
        figure, ax = plt.subplots(self.num_thr,2,figsize=(10,10))
        figure.tight_layout()
        figure.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
        figure.suptitle('Diagnosis of the thrusters models functionality')
        
        for i in range(self.num_thr):
            for j in range(2):
                
                num_cut = int(abs((len(self.time_) - len(self.ta_res_arr.T[2*i+j]))))
                if len(self.time_)>len(self.ta_res_arr.T[2*i+j]):
                    ti_ = self.time_[:-(num_cut)]
                elif len(self.time_)<len(self.ta_res_arr.T[2*i+j]):
                    ti_ = self.time_.append(self.time_[-(num_cut)])
                else:
                    ti_ = self.time_

                ax[i,j].plot(ti_,self.ta_res_arr.T[2*i+j],c='deeppink',linestyle='-',alpha=0.5,label='Thrust allocation')
                ax[i,j].plot(ti_,self.delta_ta_arr.T[2*i+j],c='blue',linestyle='-',alpha=0.5,label='Delta TA')
                ax[i,j].plot(ti_,self.thr_force_arr.T[2*i+j],c='deepskyblue',alpha=0.5,label='Actual force on the thrusters')
                ax[i,j].set_xlabel('Time [s]')
                ax[i,j].set_ylabel('Force [N]')
                ax[i,j].set_title('X' if j==0 else 'Y')
                ax[i,j].legend()
        
        figure.savefig(self.dir+'/thrust_allocation.png',dpi=300)
                
    def input_thr_plots(self):
        
        figure,ax = plt.subplots(self.num_thr,2,figsize=(10,10))
        figure.tight_layout()
        figure.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
        figure.suptitle('Thrusters settings from TA vs Actaul settings \n Diagnosis of the thrusters dynamics')
        
        num_cut = int(abs((len(self.time_) - len(self.rev_ang_arr.T[0]))))
        if len(self.time_)>len(self.rev_ang_arr.T[0]):
            ti_ = self.time_[:-(num_cut)]
        elif len(self.time_)<len(self.rev_ang_arr.T[0]):
            ti_ = self.time_.append(self.time_[-(num_cut)])
        else:
            ti_ = self.time_
        a=0.7
        for i in range(self.num_thr):
            ax[i,0].plot(ti_,self.rev_ang_arr.T[i],c='teal',label='Input from TA',zorder=2,alpha=a)
            ax[i,1].plot(ti_,self.rev_ang_arr.T[i+self.num_thr],c='mediumaquamarine',label='Input from TA',zorder=2,alpha=a)
            
            ax[i,0].plot(ti_,self.rev_arr.T[i],c='deeppink',label='Actual',zorder=3,alpha=1)
            ax[i,1].plot(ti_,self.angle_arr.T[i],c='orchid',label='Actual',zorder=3,alpha=1)
            
            ax[i,0].set_title('RPM')
            ax[i,1].set_title('Thruster angle')
            
            ax[i,0].legend()
            ax[i,1].legend()
        
        df = pd.DataFrame(self.rev_arr)
        df.to_csv(self.dir+'/rev.csv',sep = ';')
        
        df = pd.DataFrame(self.angle_arr)
        df.to_csv(self.dir+'/ang.csv',sep = ';')
        
        figure.savefig(self.dir+'/thruster_set_controls.png',dpi=300)
            
    def tau_thrusters(self,t):
        
        num_thr = self.num_thr
        ship_name = self.ship_name
        thr_models_data = self.thr_models_data
        thr_data = self.thr_data
        x_skeg = self.x_skeg
        y_skeg = self.y_skeg
        tau_control = self.tau_control
        time_step = self.time_step
        rev_arr = self.rev_arr
        angle_arr = self.angle_arr
        time_ = self.time_
        
        # Calculate thrust allocation based on control thrust vector
        duration, message, ta , dta = main_TA(num_thr, ship_name).solver(thr_data,x_skeg,y_skeg,tau_control,self.ta, thr_models_data, time_step, self.angle_actual) #tau_control
        
        #ta = np.zeros(2*self.num_thr) # !!!
        if ta is None:
           ta = self.ta

           if self.a == 0:
               print('Alocation failed at ', round(self.time_actual[-1]/60,1), ' minutes')
               self.a = self.a+1
               
        delta_ta = ta - self.ta  
        self.delta_ta_arr = np.vstack((self.delta_ta_arr,dta))
        
        self.ta = ta
        self.ta_res_arr = np.vstack((self.ta_res_arr,ta))
        
        # Calculate target revolutions and thruster angle accounting the thruster models
        thr_models_in = thrusters_models_for_input(thr_data,thr_models_data,ta)
        rev_input, angle_input = thr_models_in.thr_input()
        
        #rev_input = np.array([315,252,angle_input[2],angle_input[3]]) # !!!
        #rev_input = np.zeros(4) # !!!
        #angle_input = np.zeros(4) # !!!
        self.rev_ang_arr = np.vstack((self.rev_ang_arr,np.append(rev_input,angle_input)))
       
        # Calculate actual thrusters settings accounting for the mechanical constraints
        self.rev_actual, self.angle_actual = thrusters_inertia(angle_arr,rev_arr,thr_data,thr_models_data,rev_input,angle_input,self.rev_actual,self.angle_actual,time_step).new_data()
       
        # Calculate actual thrust
        thr_models_out = thrusters_models_for_output(self.rev_actual,self.angle_actual,thr_data,thr_models_data)
        thrust = thr_models_out.x_y_componenets()
        
        #thrust = np.zeros(2*num_thr) # !!!
        self.thr_force_arr = np.vstack((self.thr_force_arr,thrust)) 
        
        # Transform into forces acting on the hull
        self.tau_thr = location_matrix(thr_data,thrust).loc()
        
        # Add froces to the array for later plots
        self.tau_arr = np.vstack((self.tau_arr,self.tau_thr)) #tau_thr
        
        # Add revolutions and angle to the array for later plots
        self.rev_arr = np.vstack((self.rev_arr,self.rev_actual))
        self.angle_arr = np.vstack((self.angle_arr,self.angle_actual))
        
        self.time_actual.append(t)
        
        
    def PID(self,v,eta_ned):
        
        kp=self.kp
        kd=self.kd
        ki=self.ki
        
        R=self.R
        
        v = v
        eta_d = eta_ned - self.eta_ref
        
        self.eta_ned_arr = np.vstack((self.eta_ned_arr,eta_d))
        self.tau_control = -(np.dot(np.dot(R.T,kp),eta_d)+np.dot(kd,v)+np.dot(np.dot(R.T,ki),np.trapz(self.eta_ned_arr.T,np.append(0,np.array(self.time_actual)),self.time_step))) #+np.dot(kd,v)
        self.tau_control_arr = np.vstack((self.tau_control_arr,self.tau_control))
        
    def PID_with_observer(self):
        
        kp=self.kp
        kd=self.kd
        ki=self.ki
        
        R=self.R
        
        v = self.vel_est
        eta_ned = self.ned_est
        eta_d = eta_ned - self.eta_ref
       
        self.eta_ned_arr = np.vstack((self.eta_ned_arr,eta_d))
        tau_control = -(np.dot(np.dot(R.T,kp),eta_d)+np.dot(kd,v)+np.dot(np.dot(R.T,ki),np.trapz(self.NED_est.T,np.append(0,np.array(self.time_actual)),self.time_step)))
        
        self.delta_tau_control = self.tau_control - tau_control
        
        
        self.tau_control = tau_control
        self.tau_control_arr = np.vstack((self.tau_control_arr,self.tau_control))
        
        #self.tau_control = np.array([0,0,0])
    
    def objective_function_calc(self, mpc_penalty):
        
        x = float(mpc_penalty[0])
        y = float(mpc_penalty[1])
        psi = float(mpc_penalty[2])
        
        penalty = np.array([x,y,psi])
        
        du = np.dot(np.absolute(self.delta_tau_control),penalty.T)
        
        x = np.hstack((self.ned_est,self.vel_est))
        
        dx = sum1(x**2)
        
        f = dx+du
        
        self.f_arr = np.append(self.f_arr,f)
        
        
    def observer(self,v_real_ned,eta_real_ned, lmb,omega_0,observer_param):
        
        M, DL, DNL = self.matrices()
        M_inv = np.linalg.inv(M)
        
        self.T = np.diag(np.array(observer_param.T[0],dtype='float64'))
        Tinv = np.linalg.inv(self.T)
        
        wo = float(omega_0)
        wc = 1.23*wo
        
        lmd = float(lmb)
        k1 = [-1*lmd*(wc/wo)*np.diag(np.ones(3)),
              2*wo*(1-lmd)*np.diag(np.ones(3))]
        k2 = np.diag([wc, wc, wc])
        k4 = np.diag(np.array(observer_param.T[2],dtype='float64')) # tysiące mniejsze rząd wielkosci 2 trzy razy mniejsza...
        k3 = np.diag(np.array(observer_param.T[1],dtype='float64')) # tysiące
        
        #k4 = np.diag([700,1000,700]) # tysiące mniejsze rxząd wielkosci 2 trzy razy mniejsza...
        #k3 = np.diag([1500,2000,1500]) # tysiące
        
        # load data from previous iteration
        vel_est = self.vel_est 
        eta_est = self.eta_est
        bias_est = self.bias_est 
        dbias_est = self.dbias_est
        dvel_est = self.dvel_est 
        deta_est = self.deta_est 
        ddzeta_est = self.ddzeta_est
        dzeta_est = self.dzeta_est
        deta_w_est = self.deta_w_est
        eta_w_est = self.eta_w_est
        
        psi = eta_est[2]+eta_w_est[2]
        psi = np.radians(psi)
        R = np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
        
        est_err = eta_real_ned - (eta_est+eta_w_est)
        
        self.deta_w_est = np.dot(k1[1],est_err) - dzeta_est*wo**2 - 2*lmd*wo*eta_w_est
        self.eta_w_est = eta_w_est+0.5*self.time_step*(deta_w_est + self.deta_w_est)  
        
        self.ddzeta_est = self.eta_w_est + np.dot(k1[0],est_err)
        self.dzeta_est = dzeta_est+0.5*self.time_step*(ddzeta_est + self.ddzeta_est)
        
        D_v = np.dot(DL,vel_est)+np.dot(DNL,vel_est*np.absolute(vel_est))
        K_term = np.dot(np.dot(R.T,k4),np.dot(R.T,est_err))
        
        # body
        
        tau_thr = self.tau_thr
        tau_wave = self.tau_wave
        tau_wind = self.tau_wind
        
        tau = (tau_thr+tau_wind+tau_wave)
        
        self.dvel_est = np.dot(M_inv,D_v+np.dot(R.T,bias_est)+self.tau_control+K_term) #self.tau_control
        self.vel_est = vel_est+0.5*self.time_step*(dvel_est + self.dvel_est)  
        
        self.vel_est_arr = np.vstack((self.vel_est_arr,self.vel_est))
        # ned
        self.deta_est = np.dot(R,vel_est)+np.dot(k2,est_err)
        self.eta_est = eta_est+0.5*self.time_step*(deta_est + self.deta_est)  
        
        self.eta_est_arr = np.vstack((self.eta_est_arr,self.eta_est))
        
        # ned
        self.dbias_est = np.dot(-Tinv,bias_est)+np.dot(k3,est_err)
        self.bias_est = bias_est + self.time_step * self.dbias_est
        
        self.bias_arr = np.vstack((self.bias_arr,self.bias_est))
        
        self.ned_est = self.eta_est 
        
        self.NED_est = np.vstack((self.NED_est,self.ned_est))

        if self.activate_observer:
            pass
        else:
            self.ned_est = eta_real_ned
            self.vel_est = v_real_ned
        
    def wave_motion(self,t,eta_ned):
        
        index = self.time_.index(t)
        self.eta_real = self.eta_ned+self.wave_1st[index]
        
    def measurement(self,t):
        
        index = self.time_.index(t)
        self.eta_real = self.eta_real + self.white_noise[index]  
        
    def ode_system(self, t, y):
        
        tau_thr = self.tau_thr
        tau_wave = self.tau_wave
        tau_wind = self.tau_wind
        
        R = self.R
        M, DL, DNL = self.matrices()
        M_inv = np.linalg.inv(M)
        
        vr = y[0:3]-np.dot(R.T,self.vc)
        
        acceleration = np.dot(M_inv,(tau_thr+tau_wind+tau_wave)+np.dot(DL,vr)+np.dot(DNL,vr*np.absolute(vr)))
        speed = y[0:3]
        speed_NED = np.dot(R,y[:3])
       
        res = np.hstack((acceleration,speed,speed_NED))
        
        return  res
    
    def template_model(self,symvar_type='SX'):
        
        M, DL, DNL = self.matrices()
        M_inv = np.linalg.inv(M)
        
        model_type = 'continuous' # either 'discrete' or 'continuous'
        model = do_mpc.model.Model(model_type, symvar_type)

        _x = model.set_variable(var_type='_x', var_name='x', shape=(6,1))

        _u = model.set_variable(var_type='_u', var_name='u', shape=(3,1))
        
        model.set_expression(expr_name='cost', expr=sum1(_x**2))
        
        zeros = np.zeros((3,3))
        MDL = M_inv@DL
        MDNL = M_inv@DNL

        A1 = np.vstack((np.hstack((zeros,np.eye(3))),np.hstack((zeros,MDL))))
        A2 = np.vstack((np.hstack((zeros,zeros)),np.hstack((zeros,MDNL))))

        B = np.vstack((zeros,M_inv))
        
        
        # Calculate actual thrust
        thr_models_out_mpc= thrusters_models_for_output(self.rev_actual,self.angle_actual,self.thr_data,self.thr_models_data)
        thrust_mpc = thr_models_out_mpc.x_y_componenets()
        
        # Transform into forces acting on the hull
        tau_thr_mpc = location_matrix(self.thr_data,thrust_mpc).loc()
        tau_slow = self.tau_slow
        
        D = np.hstack((np.array([0,0,0]),M_inv@tau_thr_mpc)).T
        E = np.hstack((np.array([0,0,0]),M_inv@tau_slow)).T 
        
        model.set_rhs('x', A1@_x+A2@_x**2*np.sign(_x)+B@_u+D)
       
        model.setup()

        return model
    
    def template_mpc(self, model, mpc_horizon, mpc_time_step, mpc_boundaries, mpc_penalty,silence_solver=True):
        
        mpc = do_mpc.controller.MPC(model)
        
        setup_mpc = {
            'n_horizon': int(mpc_horizon),
            'n_robust': 0,
            'open_loop': 0,
            't_step': mpc_time_step,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 3,
            'collocation_ni': 1,
            'store_full_solution': True
        }
        mpc.set_param(**setup_mpc)
        
        if silence_solver:
            mpc.settings.supress_ipopt_output()

        mterm = model.aux['cost']
        lterm = model.aux['cost'] # terminal cost
        
        x = float(mpc_penalty[0])
        y = float(mpc_penalty[1])
        psi = float(mpc_penalty[2])
        
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(u=np.array([x,y,psi])) # tuning factor
        
        x = float(mpc_boundaries[1][0])
        y = float(mpc_boundaries[1][1])
        psi = float(mpc_boundaries[1][2])
        
        niu = float(mpc_boundaries[2][0])
        v = float(mpc_boundaries[2][1])
        r = float(mpc_boundaries[2][2])
        
        
        max_x = np.array([[x], [y], [psi], [niu],[v],[r]])

        mpc.bounds['lower','_x','x'] = -max_x
        mpc.bounds['upper','_x','x'] =  max_x
        
        
        Tx = float(mpc_boundaries[0][0])
        Ty = float(mpc_boundaries[0][1])
        Mz = float(mpc_boundaries[0][2])

        """
        self.counter = 0
        if self.counter == 0:
            print('Tx, Ty, Mz',Tx,Ty,Mz)
            print('x, y, psi',x,y,psi)
            print('niu, v, r',niu,v,r)
            self.counter = 1
        """
        max_u = np.array([[Tx],[Ty],[Mz]])
            
        mpc.bounds['lower','_u','u'] = -max_u
        mpc.bounds['upper','_u','u'] = max_u


        mpc.setup()

        return mpc
    
    def mpc_(self,mpc):
        
        #R = self.R
        x = np.hstack((self.ned_est,self.vel_est))
        
        tau_control = np.hstack((mpc.make_step(x)))
        
        self.delta_tau_control = self.tau_control - tau_control
        
        self.tau_control = tau_control
        self.tau_control_arr = np.vstack((self.tau_control_arr,self.tau_control))

   
        
            
       
       
        