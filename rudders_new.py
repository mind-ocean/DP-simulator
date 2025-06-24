import numpy as np
import matplotlib.pyplot as plt

class rudders_new:
    def __init__(self,ang,te,thr_data,step):
        self.ang=ang
        self.te=te
        self.thr_data=thr_data
        self.step=step
        
    def max_ang_thr(self,ruda,t,i):
        
        cx, cy = self.Cx_Cy(i)
        
        
        tx_max=t*(1-cx*ruda**2)
        ty_max=t*cy*ruda
        
        alfa=round(np.degrees(np.arctan2(ty_max,tx_max)))
        
        return alfa,tx_max,ty_max
    
    def boundary(self,start,end):
        # calculates the boundary coefficients for angles
       
        fi_start=start
        fi_end=end  
            
        a11=np.sin(np.radians(fi_start))
        a12=-np.cos(np.radians(fi_start))
        a21=-np.sin(np.radians(fi_end))
        a22=np.cos(np.radians(fi_end))
        
        arr1=[a11, a12]
        arr2=[a21, a22]
        
        G_bound=[arr1,arr2]
        
        h_bound=[0,0]
       
        return G_bound, h_bound
    
    def zones(self,ang_range,step,t,i,G_i_el,h_i_el,num_side,ang):
              
        G_i_el.append([]) 
        h_i_el.append([])
        
        fi_start,tx_max_s,ty_max_s=self.max_ang_thr(ang_range[0],t,i)
        fi_end,tx_max_e,ty_max_e=self.max_ang_thr(ang_range[1],t,i)
        
        if fi_start<0:
            fs=360+fi_start
            fe=360
        else:
            fs=0
            fe=fi_end
            
        G_bound,h_bound=self.boundary(fs,fe)
        
        G_i_el[num_side].append(G_bound[0])
        G_i_el[num_side].append(G_bound[1])
        h_i_el[num_side].append(h_bound[0])
        h_i_el[num_side].append(h_bound[1])
        
        
        rr=np.linspace(ang_range[0],ang_range[1],int(ang/step)+1)
        
        for k in range(len(rr)-1):
            
            a_start=rr[k]
            a_end=rr[k+1]
                
            fi_start_line,tx_max_s,ty_max_s=self.max_ang_thr(a_start,t,i)
            fi_end_line,tx_max_e,ty_max_e=self.max_ang_thr(a_end,t,i)
            
            
            ts=(tx_max_s**2+ty_max_s**2)**0.5
            
            te=(tx_max_e**2+ty_max_e**2)**0.5
            
            x0=ts*np.cos(np.radians(fi_start_line))
            x1=te*np.cos(np.radians(fi_end_line))
            
            y0=ts*np.sin(np.radians(fi_start_line))
            y1=te*np.sin(np.radians(fi_end_line))
            
            
            #plt.plot([y0,y1],[x0,x1],c='deeppink')
            
            a1=y1-y0
            a2=x0-x1

            G=[a1,a2]
            h=(x0*y1-x1*y0)
            
            G_i_el[num_side].append(G)
            h_i_el[num_side].append(h)
        
        return G_i_el, h_i_el
        
    def con_prop(self):
        step=self.step
        te=self.te
        ang=self.ang
        thr_data=self.thr_data
        
        G_group=[]
        h_group=[]
        
        for i, t, in enumerate(te):
            
            
            if thr_data[i][9]==0 or t==0:
                G_i_el=[]
                h_i_el=[]
                
            else:
                
                inlet_mode=thr_data[i][1]
                
                choices={
                    'forward':1.0,
                    'reverse FPP':0.9,
                    'reverse FPP nozzle':0.7,
                    'reverse CPP':0.65,
                    'reverse CPP nozzle':0.5,
                    'broken':1.0,
                    'raunded':1.07,
                    'other':0.93,
                    }
                
                loss_factor=choices.get(inlet_mode, 'not in the base')
                
                
                G_i_el=[]  
                h_i_el=[]
                
                G_i_el, h_i_el = self.zones([0,ang],step,t,i,G_i_el,h_i_el,0,ang)
                G_i_el, h_i_el = self.zones([-ang,0],step,t,i,G_i_el,h_i_el,1,ang)
                
                
                T_rev=((t*loss_factor)**2)**0.5
                G_reverse=[[-1,0],[1,0]]
                h_reverse=[T_rev,0]
                
                G_i_el.append(G_reverse)
                h_i_el.append(h_reverse)
                
            
            G_group.append(G_i_el)
            h_group.append(h_i_el)
            
        
        return G_group, h_group
            
            
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
    
