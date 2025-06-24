
import numpy as np

class thrust_power:
    def __init__(self,thr_data,beta_misc,num_thr):
        self.thr_data=thr_data
        self.num_thr=num_thr
        self.beta_misc=beta_misc
    
    def thrust_val(self):
        thr_data=self.thr_data
        num_thr=self.num_thr
        beta_misc=self.beta_misc
        
        T_n=[]
        T_e=[]
        for i in range(num_thr):
            if thr_data[i][7]!=0:
                T_nominal=self.T_max(i)
                T_effective=self.T_after_vent_losses_and_misc(i,T_nominal,beta_misc)
            
            else:
                T_nominal=0
                T_effective=0
                
            T_n.append(T_nominal)
            T_e.append(T_effective)
          
        return T_n, T_e
    
    def T_max(self,i):
        # calc the nominal thrust delivered by the thruster
        
        thr_data=self.thr_data
        
        P_max=thr_data[i][7]
        D=thr_data[i][6]
        type_thr=thr_data[i][0]
        inlet_mode=thr_data[i][1]
        eta_mech=thr_data[i][8]
        
        eta1=float(self.eta_1(type_thr))
        eta2=float(self.eta_2(inlet_mode))
        eta_mech=float(self.eta_mech_(eta_mech))
        
        b=0.6666666666667
        
        choices={
            'propeller FPP':1,
            'propeller FPP nozzle':1,
            'propeller CPP':1,
            'propeller CPP nozzle':1,
            'propeller CRP':1,
            'azi':2,
            'azi nozzle':2,
            'azi CRP':2,
            'pod':2,
            'pod nozzle':2,
            'pod CRP':2,
            'tunnel':2,
            'cyclo':2,
            }
        
        prop=choices.get(type_thr, 'not in the base')
        
        if prop==1:
            eta2=1
        
        T_nominal=eta1*eta2*(P_max*eta_mech*D)**b
        
        return T_nominal
    
    def T_after_vent_losses_and_misc(self,i,T_nominal,beta_misc):
        # calc the effective thrust after ventilation and other losses
        
        T_eff=T_nominal*beta_misc
        
        return T_eff
        
    
    def eta_1(self,type_thr):
        
        choices={
            'propeller FPP':800,
            'propeller FPP nozzle':1200,
            'propeller CPP':800,
            'propeller CPP nozzle':1200,
            'propeller CRP':950,
            'azi':800,
            'azi nozzle':1200,
            'azi CRP':950,
            'pod':800,
            'pod nozzle':1200,
            'pod CRP':950,
            'tunnel':900,
            'cyclo':900,
            }
        
        eta1=choices.get(type_thr, 'not in the base')
          
        return eta1
    
    def eta_2(self,inlet_mode):
                
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
        eta2=choices.get(inlet_mode, 'not in the base')
        
        return eta2
    
    def eta_mech_(self,eta_m):
       
           
        choices={
            'cyclo':0.91,
            'magnet':0.97,
            'azi tunnel':0.93,
            'rim magnet':0.995,
            'propller':0.97,
            'pod':0.98,
            }
    
        eta_mech=choices.get(eta_m, 'not in the base')
        
        return eta_mech
    
    
    def weight(self,e):
        # get the weight values for P matrix and corresponding max thrust values
        
        num_thr=self.num_thr
        thr_data=self.thr_data
        T_nom,T_eff=self.thrust_val()
        
        
        weight=[]
        radius=[]
        for i in range(num_thr):
            if T_eff[i]!=0:
                w=self.weight_i(i)
                w_pair=[w,w]
                weight.append(w_pair)
                
                type_thr=thr_data[i][0]
                
                if type_thr!='tunnel' and type_thr!='propeller FPP' and type_thr!='propeller FPP nozzle' \
                    and type_thr!='propeller CPP' and type_thr!='propeller CPP nozzle' and type_thr!='propeller CRP':
                    radius.append(T_eff[i])
                else:
                    radius.append(e*T_eff[i])
                    
        return weight, radius
          
    
    def weight_i(self,i):
        # get the weight coefficient relevant for the quadratic relation thrust-power
        
        c=self.coef(i)
        T_n,T_e=self.thrust_val()
        
        if T_e[i]==0:
            w=0
        else:
            if c==0:
                w=0
            else:
                w=c*T_e[i]**(-0.5)
        
        return w
    
    def coef(self,i):
        # get the coeff defining the real thrust_power curve:
        
        beta_misc=self.beta_misc
        
        thr_data=self.thr_data
        D=thr_data[i][6]
        
        D=thr_data[i][6]
        
        type_thr=thr_data[i][0]
        inlet_mode=thr_data[i][1]
        eta_mech=thr_data[i][8]
        
        eta1=float(self.eta_1(type_thr))
        eta2=float(self.eta_2(inlet_mode))
        eta_mech=float(self.eta_mech_(eta_mech))
        beta_T=beta_misc
        
        b=0.6666666666667  
        
        a=beta_T*eta1*eta2*(eta_mech*D)**b
        
        if a==0:
            c=0
        else:
            c=(1/a)**(1/b)
        
        return c
    
    def P_matrix(self,weight):
        
        n=len(weight)
        
        P=np.zeros((n*2,n*2))
        for j in range(n):
            P[2*j][2*j]=2*weight[j][0]
            P[2*j+1][2*j+1]=2*weight[j][1]
            
        q=np.zeros((n*2))
        
        return P,q

    def correct_P(self,P,loss,weight,k):
        # create a k matrix P having a list of losses for k and list of ax,ay combinations
        
        num_thr=self.num_thr
        thr_data=self.thr_data
        
        n=len(weight)
        
        P_k=np.zeros((n*2,n*2))
        
        for j in range(len(loss)):
            if loss[j]!=[]:
                loss_factor=1/loss[j]**2
            else:
                loss_factor=1
            
        
            
            P_k[2*j][2*j]=P[2*j][2*j]*loss_factor**2
            P_k[2*j+1][2*j+1]=P[2*j+1][2*j+1]*loss_factor**2
        
        return P_k
        
        
        
 
        
