


import numpy as np

from thrust_power_TA import*

class results:
    def __init__(self,solution,weight,thr_int_def,T_eff,thr_data,num_thr,beta_vent_loss,beta_misc,ax,ay,at,rudder_max_angle,rudder_angle_step,pp):
        self.solution=solution
        self.weight=weight
        self.thr_int_def=thr_int_def
        self.T_eff=T_eff
        self.thr_data=thr_data
        self.num_thr=num_thr
        self.beta_vent_loss=beta_vent_loss
        self.beta_misc=beta_misc
        self.ax=ax
        self.ay=ay
        self.at=at
        self.rudder_max_angle=rudder_max_angle
        self.rudder_angle_step=rudder_angle_step
        self.pp=pp
    
    def max_ang_thr(self,ruda,t,i):
        
        cx, cy = self.Cx_Cy(i)
        
        tx_max=t*(1-cx*ruda**2)
        ty_max=t*cy*ruda
        
        deg=round(np.degrees(np.arctan2(ty_max,tx_max)))
        
        return deg 
    
    
    def total_power(self):
        Tn_list=self.T_nom_rud()
        solution=self.solution
        num_thr=self.num_thr
        T_eff=self.T_eff
        thr_data=self.thr_data
        
        P_total=0
        P_list=[]
        P_max_list=[]
        uti_power=[]
        j=-1
        for i in range(num_thr):
            if thr_data[i][7]==0:
                P_='N/A'
                P_max='N/A'
                uti_pow='N/A'
            
            else:
                if T_eff[i]==0:
                    P_='vent'
                    P_max='vent'
                    uti_pow='N/A'
                    
                else:
                    j=j+1
                    Tn=Tn_list[j]
                    P_=round(self.power_brake(i,Tn))
                    P_max=round(thr_data[i][7],2)
                    uti_pow=round((P_/P_max*100),2)
                    P_total=P_total+P_
                    
            P_list.append(P_)
            P_max_list.append(P_max)
            uti_power.append(uti_pow)
            
        
        return P_total, P_list,P_max_list,uti_power
    
    def rudder_angle_force(self,j,k):
        # finds a rudder angle and thrust angle x, y forces for i-th thr
        
        amax=self.rudder_max_angle
        solution=self.solution
        T_eff=self.T_eff
        
        Tx_r=solution[2*j]
        Ty_r=solution[2*j+1]
        
        t=T_eff[k]
        
        bmax=self.max_ang_thr(amax,t,j)
        
        if Tx_r<=0 or round(Ty_r)==0:
            rud_angle=0
        
        else:
            
            beta=np.degrees(np.arctan2(Ty_r,Tx_r))
            rud_angle=round(beta*amax/bmax)
            
        return Tx_r, Ty_r, rud_angle
    
    
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
   
    def T_nom_rud(self):
        # Replace the nominal trhust values with the rudder nominal thrust
        
        Tn_list=self.T_nominal()
        Tx_rud,Ty_rud, angle_rud=self.prop_rudder()
        ax=self.ax
        ay=self.ay
        rudder_max_angle=self.rudder_max_angle
        rudder_angle_step=self.rudder_angle_step
        T_eff=self.T_eff
        num_thr=self.num_thr
        thr_data=self.thr_data
        beta_vent_loss=self.beta_vent_loss
        beta_misc=self.beta_misc
        pp=self.pp
        
        
        for i, t in enumerate(T_eff):
            
            if thr_data[i][9]!=0 and thr_data[i][7]!=0:
                Tn_list[i]=t
                
                if Tx_rud[i]<0:
                    
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
                    
                    loss_factor=choices.get(thr_data[i][1], 'not in the base')
                    
                    Tn_list[i]=t*loss_factor
                    
                
        return Tn_list
    
    def prop_rudder(self):
        num_thr=self.num_thr
        T_eff=self.T_eff
        thr_data=self.thr_data
        
        ith=[]
        for i in range(num_thr):
            if T_eff[i]!=0:
                ith.append(i)
        
        Tx_rud=[]
        Ty_rud=[]
        angle_rud=[]
        
        j=0
        for i in range(num_thr):
            
            thr_type=thr_data[i][0]
            
            
            if thr_data[i][7]!=0:
                
                
                if (thr_type=='propeller FPP' or thr_type=='propeller FPP nozzle' or \
                    thr_type=='propeller CPP' or thr_type=='propeller CPP nozzle' or \
                        thr_type=='propeller CRP') and T_eff[i]!=0:
                            
                    k=i
                
                    Tx_r, Ty_r,rud_angle=self.rudder_angle_force(j,k)
                    Tx_r=round(Tx_r/1000,2)
                    Ty_r=round(Ty_r/1000,2)
                    
                else:
                    Tx_r='N/A'
                    Ty_r='N/A'
                    rud_angle='N/A'
                
                j=j+1
                    
            else:
                Tx_r='N/A'
                Ty_r='N/A'
                rud_angle='N/A'
                
            
            Tx_rud.append(Tx_r)
            Ty_rud.append(Ty_r)
            angle_rud.append(rud_angle)
            
        return Tx_rud,Ty_rud, angle_rud
    
    def power_brake(self,i,Tn):
        
        thr_data=self.thr_data
        num_thr=self.num_thr
        beta_vent_loss=self.beta_vent_loss
        beta_misc=self.beta_misc
        
        T_P=thrust_power(thr_data,beta_misc,num_thr)
        
        D=thr_data[i][6]
        type_thr=thr_data[i][0]
        inlet_mode=thr_data[i][1]
        eta_mech=thr_data[i][8]
        
        eta1=float(T_P.eta_1(type_thr))
        eta2=float(T_P.eta_2(inlet_mode))
        eta_mech=float(T_P.eta_mech_(eta_mech))
        
        a=1/(eta_mech*D)*(1/(eta1*eta2))**1.5
        b=1.5
        if Tn==0:
            P_brake=0
        else:
            P_brake=a*Tn**b
        
        return P_brake
    
    def power_i(self,i):
        
        solution=self.solution
        weight=self.weight
        
        w_i=weight[i][0]
        Tx=solution[2*i]
        Ty=solution[2*i+1]
        
        Power_i=w_i*(Tx)**2+w_i*(Ty)**2
        
        return Power_i 
    
    def percent_T_nom(self):
        
        thr_data=self.thr_data
        num_thr=self.num_thr
        beta_vent_loss=self.beta_vent_loss
        beta_misc=self.beta_misc
        T_eff=self.T_eff
        
        T_P=thrust_power(thr_data,beta_misc,num_thr)
        
        ith=[]
        for i in range(num_thr):
            if T_eff[i]!=0:
                ith.append(i)
                
        Tmax_nom_list=[]      
        for j in range(len(ith)):
            i=ith[j]  
            T_max_nom=T_P.T_max(i)
            Tmax_nom_list.append(T_max_nom)
            
        Tn_list=self.T_nom_rud()   
        
        uti_nom=[]
        for j in range(len(Tn_list)):
            per=round(Tn_list[j]/Tmax_nom_list[j]*100,2)
            uti_nom.append(per)
            
            
        return uti_nom
    
    def T_effective(self):
        # before losses due to dead thr or skeg
        
        solution=self.solution
        losses=self.thr_losses()
        
        T_list=[]
        Te_list=[]
        for i in range(len(losses)):
                
            Tx=solution[2*i]
            Ty=solution[2*i+1]
            
            T=(Tx**2+Ty**2)**0.5
            
            T_list.append(T)
            loss_thr=round(losses[i],3)
           
            Te=round(T/loss_thr)
            
            Te_list.append(Te)
        
        return Te_list,T_list
   
    def T_nominal(self):
        # delivered by the prop
        
        Te_list,T_list=self.T_effective()
        beta_vent_loss=self.beta_vent_loss
        beta_misc=self.beta_misc
        num_thr=self.num_thr
        
        Tn_list=[]
        beta_v=[]
        for i in range(num_thr):
            if beta_vent_loss[i]!=0:
                beta_v.append(beta_vent_loss[i])
                
        for i in range(len(Te_list)):
            beta_loss=beta_v[i]*beta_misc
            Tn=round(Te_list[i]/beta_loss)
            
            Tn_list.append(Tn)
        
        return Tn_list
    
    def beta_thr(self):
        
        solution=self.solution
        T_eff=self.T_eff
        Tx_rud,Ty_rud, angle_rud=self.prop_rudder()
        num_thr=self.num_thr
        thr_data=self.thr_data
        
        a=0
        for j in range(len(T_eff)):
            if T_eff[j]!=0:
                a=a+1
        
        beta_thr_all=[]
        for j in range(a):
            Tx=solution[2*j]
            Ty=solution[2*j+1]
            
            if round(Tx)==0 and round(Ty)==0:
                beta_thr='N/A'
            else:
                beta_thr=self.beta_thr_i(Tx,Ty)
            beta_thr_all.append(beta_thr)
            
        
        angle_rudder_all=[]
        j=-1
        for i in range(num_thr):
            
            if T_eff[i]!=0:
                j=j+1
                
                thr_type=thr_data[i][0]
                if thr_type=='propeller FPP' or thr_type=='propeller FPP nozzle' or \
                    thr_type=='propeller CPP' or thr_type=='propeller CPP nozzle' or \
                        thr_type=='propeller CRP':
                            
                    angle_rudder_all.append(angle_rud[j])
                    
                else:
                    angle_rudder_all.append('N/A')
            
            else:
                angle_rudder_all.append('N/A')
        
        return beta_thr_all, angle_rudder_all
        
    def beta_thr_i(self,Tx,Ty):
        # calculates angle of thr
        
        beta_thr=int(round((np.degrees(np.arctan2(Ty,Tx)))))
        
        if beta_thr<0:
           beta_thr=360+beta_thr
          
        return beta_thr
    
    def thr_losses(self):
        # creates a modified list of losses for working thrusters
        # get the loss for each thr
        
        thr_int_def=self.thr_int_def
        T_eff=self.T_eff
        solution=self.solution
        num_thr=self.num_thr
        
        new_thr_int_def=[]
        for i in range(num_thr):
            if T_eff[i]!=0:
                new_thr_int_def.append(thr_int_def[i])
                
        losses=[]
        for j in range(len(new_thr_int_def)):
            new_thr_int_def_j=new_thr_int_def[j]
            
            if new_thr_int_def_j!=[]:
                Tx=solution[2*j]
                Ty=solution[2*j+1]
                
                beta_thr=self.beta_thr_i(Tx,Ty)
                loss_thr=self.loss(beta_thr,new_thr_int_def_j,Tx,Ty)
            else:
                loss_thr=1
            losses.append(loss_thr)
         
        return losses
                
    def loss(self,beta_thr,new_thr_int_def_j,Tx,Ty):
        # calculates an interpolated loss
        
        if round(Tx)==0 and round(Ty)==0:
            loss_thr=1
        
        else:
            
            for k in range(len(new_thr_int_def_j)-1):
                start=int(new_thr_int_def_j[k][0])
                end=int(new_thr_int_def_j[k+1][0])
                start_loss=new_thr_int_def_j[k][1]
                end_loss=new_thr_int_def_j[k+1][1]
                
                
                if start_loss!=0 and end_loss!=0 and end!=start:
                    
                    if beta_thr>=start and beta_thr<=end:
                        
                        loss_thr=round((start_loss+(end_loss-start_loss)/(end-start)*(beta_thr-start)),3)
                    
        return loss_thr
    
    def all_losses(self):
        # define the total loss with ventilation and interaction losses 
        
        losses=self.thr_losses()
        thr_data=self.thr_data
        beta_vent_loss=self.beta_vent_loss
        beta_misc=self.beta_misc
        num_thr=self.num_thr
        T_eff=self.T_eff
        Tn_list=self.T_nom_rud()
        Te_list,T_list=self.T_effective()
        
       
        beta_v=[]
        for i in range(num_thr):
            if beta_vent_loss[i]!=0:
                beta_v.append(beta_vent_loss[i])
         
        loss_fin=[]
        j=-1
        for i in range(num_thr):
            if thr_data[i][7]==0:
                loss_final='N/A'
                
            else:
                if T_eff[i]==0:
                    loss_final=0
                    
                else:
                    j=j+1
                    thr_type=thr_data[i][0]
                    if thr_type=='propeller FPP' or thr_type=='propeller FPP nozzle' or \
                        thr_type=='propeller CPP' or thr_type=='propeller CPP nozzle' or \
                            thr_type=='propeller CRP':
                                
                        if int(round(Tn_list[j]))!=0:
                            loss_final=round((T_list[j]/Tn_list[j]),2)  
                        else:
                            loss_final='N/A'
                    else:
                        loss_final=round((beta_v[j]*beta_misc*losses[j]),2)
                    
            loss_fin.append(loss_final)
        
        return loss_fin
    
    def postproc(self):
        
        T_eff=self.T_eff
        thr_data=self.thr_data
        num_thr=self.num_thr
        beta_vent_loss=self.beta_vent_loss
        beta_misc=self.beta_misc
            
        beta_thr_all,angle_rudder_all=self.beta_thr()
        Te_list,T_list=self.T_effective()
        Tn_list=self.T_nom_rud()
        
        T_P=thrust_power(thr_data,beta_misc,num_thr)
        
        Tnom=[]
        Teff=[]
        Tnom_per_max=[]
        Teff_per_Tnom=[]
        j=-1
        beta_new=[]
        rudrud=[]
        for i in range(num_thr):
            if thr_data[i][7]==0:
                T_nom='N/A'
                T_e='N/A'
                Tnom_max='N/A'
                Teff_T_nom='N/A'
                beta='N/A'
                rud='N/A'
            
            else:
                if T_eff[i]==0:
                    T_nom='vent'
                    T_e='vent'
                    Tnom_max='N/A'
                    Teff_T_nom='N/A'
                    beta='N/A'
                    rud='N/A'
                else:
                    j=j+1
                    T_nom=int(round(Tn_list[j]))
                    T_e=round(T_list[j]/1000,2)
                    T_max_nom=round(T_P.T_max(i),2)
                    Tnom_max=round((T_nom/T_max_nom*100),2)
                    Teff_T_nom=round((T_e/T_max_nom*100),2)
                    rud=angle_rudder_all[i]
                    if beta_thr_all[j]=='N/A':
                        beta='N/A'
                    else:
                        beta=round(beta_thr_all[j],2)
                        
            Tnom.append(T_nom)
            Teff.append(T_e)
            Tnom_per_max.append(Tnom_max)
            Teff_per_Tnom.append(Teff_T_nom)
            beta_new.append(beta)
            rudrud.append(rud)
            
        return Tnom, Teff, Tnom_per_max, Teff_per_Tnom, beta_new,rudrud
            
            
        
        
        
        
       
        
            
        
         
         
         
        