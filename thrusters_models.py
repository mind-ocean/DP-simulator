import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class thrusters_models_for_input:
    def __init__(self,thr_data,thr_models_data, ta):
        self.thr_data = thr_data
        self.thr_models_data = thr_models_data
        self.ta = ta
        
    def thr_input(self):
        
        thr_models_data = self.thr_models_data
        thr_data = self.thr_data
        ta = self.ta
        
        # 1. Loop throught thrusters
        # 2. Read bollard polynomials
        # 3. Calculate effective thrust magnitude
        # 4. Find revolutions and angle
        
        
        # 1 ----------
        thr_rev = []
        thr_angle = []
        
        for i in range(len(thr_data)):
            
            # 2 ------------
            T_polynomial = [thr_models_data[i][3],thr_models_data[i][4]]
            
            Tx = ta[2*i]
            Ty = ta[2*i+1]
            
            
            # 3 --------------
            T = (Tx**2+Ty**2)**0.5
            
            max_n = float(thr_models_data[i][0])
           
            # 4 ---------------
            
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
          
            type_thr=str(choices1.get(thr_data[i][0], 'not in the base'))
                         
                         
            if type_thr=='azi':
                
                rev_input = self.bollard_rev(T_polynomial,T,max_n)
                angle_input = self.angle_azi(Tx,Ty)
                
                if T == 0:
                    rev_input = 0
                    
                thr_rev.append(rev_input)
                thr_angle.append(angle_input)
                
                #plt.scatter(rev_input, T,color='deeppink')
                
            elif type_thr=='tunnel':
                
                T = T * np.sign(Ty)
                
                rev_input = self.bollard_rev(T_polynomial,T,max_n)
                
                if Ty == 0:
                    rev_input = 0
                 
                angle_info = 90
                
                thr_rev.append(rev_input)
                thr_angle.append(angle_info)
        
        return thr_rev, thr_angle
                
    
    def rev_bollard(self,polynomial,n):
        
        a = float(polynomial[0])
        b = float(polynomial[1])
        
        force = a*n*abs(n)+b*n
        
        return force
    
    def bollard_rev(self,polynomial,T,max_n):
        
        rev = np.linspace(-max_n,max_n,100)
        
        bollard = []
        for n in rev:
            bollard_i = self.rev_bollard(polynomial,n)
            bollard.append(bollard_i)
        
        rev_input = np.interp(T,bollard,rev)
        
        return rev_input
    
    def angle_azi(self,Tx,Ty):
        
        
        beta = (np.degrees(np.arctan2(Tx,Ty)))
        
        if Tx>=0 and Ty>=0:
            
            beta_prim = 90-beta
            
        elif Tx<=0 and Ty>=0:
            
            beta_prim = 90-beta
            
        elif Tx>=0 and Ty<=0:
            
            beta_prim = 360 + (90 - beta)
        
        elif Tx<=0 and Ty<=0:
            
            beta_prim = 90 - beta
        
        #if beta_prim<0:
            #beta_prim = 360 + beta_prim
            
        return beta_prim
  
class thrusters_models_for_output:
    
    def __init__(self,thr_rev,thr_ang,thr_data,thr_models_data):
        self.thr_models_data = thr_models_data
        self.thr_data = thr_data
        self.thr_rev = thr_rev
        self.thr_ang = thr_ang
        
        
    def x_y_componenets(self):
        
        thr_models_data = self.thr_models_data
        thr_data = self.thr_data
        thr_rev = self.thr_rev
        thr_ang = self.thr_ang
        
        # 1. Find absolute thrust from the bollard pull polynomials
        # 2 . Calculate componenet on x and y
        
        Thrust=[]
        
        # 1 ---------------
        for i in range(len(thr_rev)):
            
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
            
            
            T_polynomial = [thr_models_data[i][3],thr_models_data[i][4]]
            T = self.rev_bollard(T_polynomial,thr_rev[i])
            
            # 2 -------------
            if thr_type=='azi':
                
                angle = thr_ang[i]
                
                #if angle<0:
                    #angle = angle+360
                
                Tx, Ty = self.x_y(angle,T)
                
                
            elif thr_type=='tunnel':
                
                Tx = 0
                Ty = T
            
            Thrust.append(Tx)
            Thrust.append(Ty)
                           
        thrust = np.array(Thrust)
        
        return thrust
    
    def rev_bollard(self,polynomial,n):
                    
        a = float(polynomial[0])
        b = float(polynomial[1])
        
        force = a*n*abs(n)+b*n
        
        return force
    
    def x_y(self,angle,T):
        
        Tx = T*np.cos(np.radians(angle))
        Ty = T*np.sin(np.radians(angle))
        
        return Tx, Ty

class thrusters_inertia:
    def __init__(self,angle_arr,rev_arr,thr_data,thr_models_data,rev_input,angle_input,rev_actual,angle_actual,time_step):
        self.thr_models_data = thr_models_data
        self.thr_data = thr_data
        self.rev_input = rev_input
        self.angle_input = angle_input
        self.rev_actual = rev_actual
        self.angle_actual = angle_actual
        self.time_step = time_step
        self.angle_arr = angle_arr
        self.rev_arr = rev_arr
        
    
    def new_data(self):
        
        self.angle_input = np.array(self.angle_input,dtype='float64')
        self.angle_actual = np.array(self.angle_actual,dtype='float64')
        
        for i in range(len(self.thr_data)):
            
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
          
            thr_type=str(choices1.get(self.thr_data[i][0], 'not in the base'))
            
            ang_rate = float(self.thr_models_data[i][2]) 
            rev_rate = float(self.thr_models_data[i][1])
            max_rev = float(self.thr_models_data[i][0])
            
            time_constant_rev = max_rev/(rev_rate*60)
            time_constant_ang = 360/ang_rate
            
            
            if thr_type == 'tunnel':
                
                self.rev_actual[i] += (self.rev_input[i]-self.rev_actual[i])/time_constant_rev*self.time_step
                
                if abs(self.rev_actual[i])>max_rev:
                    self.rev_actual[i] = np.sign(self.rev_actual[i])*max_rev
                        
                self.angle_actual[i] = 90 if self.rev_actual[i]>0 else 270
                    
                
            elif thr_type == 'azi':
                
                self.rev_actual[i] += (self.rev_input[i]-self.rev_actual[i])/time_constant_rev*self.time_step
                
                if abs(self.rev_actual[i])>max_rev:
                    self.rev_actual[i]=max_rev
                
                if self.angle_input[i] == 360:
                    self.angle_input[i] = 0
                    
                self.angle_actual[i] += (self.angle_input[i]-self.angle_actual[i])/time_constant_ang*self.time_step
               
        return self.rev_actual, self.angle_actual                        
                
class location_matrix:   
    def __init__(self,thr_data,thrust):
        self.thr_data = thr_data
        self.thrust = thrust
        
    def loc(self):
        
        thr_data = self.thr_data
        thrust = self.thrust
        
        A=[]
        
        for i in range(len(thr_data)):
            
            A.append([])
            
            x=thr_data[i][3]
            y=thr_data[i][4]  
            
            balance=[[1,0],[0,1],[-y,x]]
            
            A[i]=balance
        
        A=np.hstack(A)
        
        thrust= np.array(thrust).T
        
        tau_to_hull = np.matmul(A,thrust)
        
        return tau_to_hull
                
                
                
                
        
        
            
            
            
            
            
            
            