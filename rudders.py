import numpy as np

class propeller_rudder:
    def __init__(self,thr_data,num_thr,rudder_angle_step,rudder_max_angle):
        self.thr_data=thr_data
        self.num_thr=num_thr
        self.rudder_angle_step=rudder_angle_step
        self.rudder_max_angle=rudder_max_angle
    
        
    def Cx_Cy(self,i):
        
        thr_data=self.thr_data
        
        rudder_type=thr_data[i][2]
        k1=self.k1(rudder_type)
        
        thr_type=thr_data[i][0]
        k2=self.k2(thr_type)
        
        Ar=thr_data[i][9]
        D=thr_data[i][6]
        
        Cy_i=0.0126*k1*k2*Ar/D**2
        Cx_i=0.02*Cy_i
        
        return Cx_i, Cy_i
        
    def k1(self,rudder_type):
        
        choices={
            'naca': 1.1,
            'hollow': 1.35,
            'flat': 1.1,
            'fish_tail': 1.4,
            'flap': 1.65 ,
            'nozzle': 1.9,
            'mix': 1.21,
            }
        
        k1=choices.get(rudder_type, 'not in the base')
        
        return k1
        
        
    def k2 (self,thr_type):
        
        choices={
            'propeller FPP': 1.0,
            'propeller FPP nozzle': 1.15,
            'propeller CPP': 1.0,
            'propeller CPP nozzle': 1.0,
            'propeller FPP': 1.0,
            'propeller CRP': 1.0,
            }
        
        k2=choices.get(thr_type, 'not in the base')
        
        return k2
    
    def alfa_list(self,i):
        rudder_angle_step=self.rudder_angle_step
        rudder_max_angle=self.rudder_max_angle
        
        alfa=-rudder_max_angle
        alfa_i=[alfa]
        steps=int(rudder_max_angle*2/rudder_angle_step)
        for j in range(steps):
            alfa=alfa+rudder_angle_step
            alfa_i.append(alfa)
        
        return alfa_i
        
    def a(self,i):
        
        alfa_i=self.alfa_list(i)
        Cx_i, Cy_i=self.Cx_Cy(i)
        
        ax_i=[]
        ay_i=[]
        a_i=[]
        for j in range(len(alfa_i)):
            ax_i_j=round((1-Cx_i*alfa_i[j]**2),8)
            ay_i_j=round((Cy_i*alfa_i[j]),8)
            if ay_i_j==0:
                a_i_j=0
            else:
                a_i_j=round((ax_i_j/ay_i_j),8)
               
            ax_i.append(ax_i_j)
            ay_i.append(ay_i_j)
            a_i.append(a_i_j)
            
        ax_i.append(-1)
        ay_i.append(1)
        a_i.append(-1)
            
        return ax_i, ay_i, a_i
    
    def coeff(self):
        
        num_thr=self.num_thr
        thr_data=self.thr_data
        
        ax=[]
        ay=[]
        a=[]
        for i in range(num_thr):
            
            thr_type=thr_data[i][0]
            if thr_data[i][7]!=0 and (thr_type=='propeller FPP' or thr_type=='propeller FPP nozzle' or thr_type=='propeller CPP' or thr_type=='propeller CPP nozzle' or  thr_type=='propeller CRP'):
                
                ax_i, ay_i,a_i=self.a(i)
               
            else:
                ax_i=[]
                ay_i=[]
                a_i=[]
                
            ax.append(ax_i)
            ay.append(ay_i)
            a.append(a_i)
       
       
        return ax, ay, a
            
            
        