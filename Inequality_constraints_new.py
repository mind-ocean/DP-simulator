
import numpy as np
from itertools import product

class groups:
    def __init__(self,thr_int_def, thr_data, T_eff,num_thr,e,at):
        self.thr_int_def=thr_int_def
        self.thr_data=thr_data
        self.T_eff=T_eff
        self.num_thr=num_thr
        self.e=e
        self.at=at
    def G_h_all(self):
        # get the coefficients gathered, not ready for the solver yet
        
        num_thr=self.num_thr
        T_eff=self.T_eff
        
        G_group=[]
        h_group=[]
        losses_group=[]
        for i in range(num_thr):
            if T_eff[i]!=0:
                G_i, h_i,losses_i=self.i_th_equations(i)
                
                losses_i_new=[]
                for j in range(len(losses_i)):
                    if losses_i[j]!=0:
                        losses_i_new.append(losses_i[j])
                        
                losses_i=losses_i_new
                losses_i=np.array(losses_i)
                
            else:
                G_i=[]
                h_i=[]
                losses_i=[]
                
            G_group.append(G_i)
            h_group.append(h_i)
            losses_group.append(losses_i)
                    
        return G_group, h_group,losses_group
                
    def i_th_equations(self,i):
        # get all the coefficients for all separate zones for i-th thr
        
        thr_data=self.thr_data
        T_eff=self.T_eff
        at=self.at
        
        thr_type=thr_data[i][0]
        
        if thr_type=='tunnel':
            if thr_data[i][7]==0:
                G_i=[]
                h_i=[]
                losses_i=[]
                
            else:
                G_i=[np.vstack(([0,1],[0,-1])).tolist()]
                h_i=[np.array([T_eff[i],T_eff[i]]).tolist()]
                losses_i=[1]
       
        
        elif thr_type=='propeller FPP' or thr_type=='propeller FPP nozzle' or thr_type=='propeller CPP' or thr_type=='propeller CPP nozzle' or thr_type=='propeller CRP':
            G_i=[]
            h_i=[]
            losses_i=[]
            for j in range(3):
                losses_i.append(1)
            
        else:
            G_i, h_i,losses_i=self.G_azi_thr(i)
       
        return G_i, h_i,losses_i
    
    def G_azi_thr_type(self,i):
        # decide the type zone - circle or complex
        
        thr_int_def=self.thr_int_def
        thr_data=self.thr_data
        
        thr_fun=thr_int_def[i]
        if len(thr_fun)==1:
            thr_fun==thr_fun[0]
            
        power=thr_data[i][7]
        
        if power!=0 and thr_fun==[]:
            type_zone="circle"
        else:
            type_zone="complex"
        
        return type_zone
    
    def G_azi_thr(self,i):
        # divide the range of more than 180 deg into two and calculates the G and h for each i
        # as a list of arrays 
        
        thr_int_def=self.thr_int_def
        type_zone=self.G_azi_thr_type(i)
        
        if len(thr_int_def[i])==1:
            thr_int_def[i]=thr_int_def[i][0]
        
        G_i=[]
        h_i=[]
        losses_i=[]
        if type_zone=="circle":
            start=0
            end=360
            G_i_el,h_i_el=self.circle_shape(i,start,end)
            
            G_i.append(G_i_el)
            h_i.append(h_i_el)
            losses_i.append(1)
            
        elif type_zone=="complex":
            
            for j in range(len(thr_int_def[i])-1):
                
                start=thr_int_def[i][j][0]
                end=thr_int_def[i][j+1][0]
                                
                loss_start=thr_int_def[i][j][1]
                loss_end=thr_int_def[i][j+1][1]
                
                min_loss=min(loss_start,loss_end)
                
                losses_i.append(min_loss)
                
                start_k=start
                end_k=end
                   
                if (end_k-start_k)>180:
                    losses_i.append(min_loss)
                    for k in range(2):
                        if k==0:
                            start=start_k
                            end=int(np.round(start+(end_k-start_k)/2))
                            
                            G_i_el,h_i_el=self.range_G_h(i,start,end,loss_start,loss_end)
                            
                            G_i.append(G_i_el)
                            h_i.append(h_i_el)
                            
                        else:
                            start=end
                            end=end_k
                        
                            G_i_el,h_i_el=self.range_G_h(i,start,end,loss_start,loss_end)
                            
                            G_i.append(G_i_el)
                            h_i.append(h_i_el)
                else:
                    G_i_el,h_i_el=self.range_G_h(i,start,end,loss_start,loss_end)
                    
                    G_i.append(G_i_el)
                    h_i.append(h_i_el)
                  
            G_i_clean=[]
            h_i_clean=[]
            
            for j in range(len(G_i)):
                if G_i[j]!=[]:
                    G_i_clean.append(G_i[j])
                    h_i_clean.append(h_i[j])
            
            G_i, h_i=self.i_th_array(G_i_clean,h_i_clean)
        
        return G_i, h_i,losses_i
    
    def i_th_array(self,G_i_clean,h_i_clean):
        # changes the G_i and h_i into a list of arrays for each i
        
        G_i_arr=[]
        h_i_arr=[]
        
        for j in range(len(G_i_clean)):
            arr_j_G=np.vstack((G_i_clean[j]))
            arr_j_h=np.array(h_i_clean[j])
            
            G_i_arr.append(arr_j_G)
            h_i_arr.append(arr_j_h)
            
        return G_i_arr, h_i_arr
    
    
    def range_G_h(self,i,start,end,loss_start,loss_end):
        # for each i-th thr get the specific G abd h each j-th element
        
        if start==end and ((loss_start==0 and loss_end==1) or (loss_start==1 and loss_end==0)):
            G_i_el=[]
            h_i_el=[]
        
        elif start!=end and loss_start==loss_end and loss_start!=0 and loss_end!=0:
            G_i_el,h_i_el=self.circle_shape(i,start,end)
            
            G_bound,h_bound=self.boundary(start,end)
            
            G_i_el.append(G_bound[0])
            G_i_el.append(G_bound[1])
            h_i_el.append(h_bound[0])
            h_i_el.append(h_bound[1])
            
        
        elif start!=end and loss_start!=loss_end:
            G_i_el,h_i_el=self.triangle_shape(i,start,end,loss_start, loss_end)
            
            G_bound,h_bound=self.boundary(start,end)
            
            G_i_el.append(G_bound[0])
            G_i_el.append(G_bound[1])
            h_i_el.append(h_bound[0])
            h_i_el.append(h_bound[1])
            
        
        elif start!=end and loss_start==0 and loss_end==0:
            G_i_el=[]
            h_i_el=[]
            
        else:
            G_i_el=[]
            h_i_el=[]
            
        
        return G_i_el, h_i_el
    
    
    def circle_shape(self,i,start,end):
        # if the y value for both ranges ends is equal than we have a circle fun
        
        T_eff=self.T_eff
        e=self.e
        range_angle=end-start
        
        N=int(np.ceil(range_angle/2/np.degrees(np.arccos(e))))
        
        G_i_el=[]  
        h_i_el=[]
          
        for k in range(N):            
            fi_k=range_angle/N*(1/2+k)
            fi=start+fi_k
            
            a1=np.cos(np.radians(fi))
            a2=np.sin(np.radians(fi))
            
            G_i_el.append([a1,a2])                        
            
            h=e*T_eff[i]                       
            h_i_el.append(h)
      
        
        return G_i_el, h_i_el
    
    def triangle_shape(self,i,fi_start,fi_end,loss_start,loss_end):
        # in case of losses
        
        T_eff=self.T_eff
        e=self.e
        sub_range=fi_end-fi_start
        
        
        N=int(np.ceil(sub_range/2/np.degrees(np.arccos(e))))
        
        r_max=e*T_eff[i]
        
        G_i_el=[]  
        h_i_el=[]
             
        for k in range(N):
                    
            fi_start_line=fi_start+sub_range/N*(k)
            fi_end_line=fi_start+sub_range/N*(k+1)
            
            loss0=loss_start+(loss_end-loss_start)/(fi_end-fi_start)*(fi_start_line-fi_start)
            loss1=loss_start+(loss_end-loss_start)/(fi_end-fi_start)*(fi_end_line-fi_start)
          
            x0=r_max*loss0*np.cos(np.radians(fi_start_line))
            x1=r_max*loss1*np.cos(np.radians(fi_end_line))
            
            y0=r_max*loss0*np.sin(np.radians(fi_start_line))
            y1=r_max*loss1*np.sin(np.radians(fi_end_line))
            
            a1=y1-y0
            a2=x0-x1
            
            G=[a1,a2]
            h=x0*y1-x1*y0
            
            G_i_el.append(G)
            h_i_el.append(h)
            
        
        return G_i_el, h_i_el
    
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
"""
NEW class
"""     
class clear_and_comb:
        def __init__(self,G,h,loss):
            self.h=h
            self.G=G
            self.loss=loss
            
        def clean(self):
            # clean the G and h matrix form the [] arrays for not working or zero vent loss
            h=self.h
            G=self.G
            
            
            G_clean=[]
            h_clean=[]
            
            for i in range(len(G)):
                if G[i]!=[]:
                    G_clean.append(G[i])
                    h_clean.append(h[i])
            
            return G_clean, h_clean
        
        def combo(self):
            # creates a combinations of all the posibilities combained
            
            G_clean, h_clean=self.clean()
            G_comb=list(product(*G_clean))
            h_comb=list(product(*h_clean))
            
            return G_comb, h_comb
        
        def sym_zero(self,j):
            # calculates the symmetric array for one option
            
            G_comb,h_comb=self.combo()
            
            G_j=G_comb[j] # - one option of G
            
            num_G_j=len(G_j) # - thr/all elements - working thr with no zero loss 
            
            G_sym=np.zeros((num_G_j*2))
            for k in range(num_G_j):
                G_j_k=G_j[k]
                G_j_k_zeros=np.zeros_like(G_j[k]) # zero matrix of an element size
        
                row=[]   
                for n in range(num_G_j):
                    
                    row.append([])
                    
                    if k==n:
                        row[n]=G_j_k
                        
                    else:
                        row[n]=G_j_k_zeros
                        
                        
                row=np.array(row)
                row=np.hstack(row)
                
                
                G_sym=np.vstack((G_sym,row))
            G_sym=np.delete(G_sym, (0), axis=0)
            
           
            return G_sym
        
        def sym_all(self):
            # get all the symetric arrays into one big list
            
            G_comb,h_comb=self.combo()
            
            G=[]
            for j in range(len(G_comb)):
                G_sym=self.sym_zero(j)
                G.append(G_sym)
                
            return G
        
        def h_prep(self):
            
            G_comb,h_comb,=self.combo()
            h=[]
            for k in range(len(h_comb)):
                h_k=h_comb[k]
                
                h_k_j=h_k[0]
                for j in range(len(h_k)-1):
                    h_k_j=np.hstack((h_k_j,h_k[j+1]))
                
                h.append(h_k_j)
                
            return h
        
        def loss_prep(self):
            
            loss=self.loss
            
            loss_clean=[]
            for j in range(len(loss)):
                if len(loss[j])!=0:
                    loss_clean.append(loss[j])
            
            loss_comb=list(product(*loss_clean))
            
            return loss_comb
        
class tunnel_thr:
    def __init__(self,thr_data,num_thr, G_, h_,T_eff):
        self.thr_data=thr_data
        self.num_thr=num_thr
        self.h_=h_
        self.G_=G_
        self.T_eff=T_eff
        
    def add_rows(self):
        thr_data=self.thr_data
        num_thr=self.num_thr
        T_eff=self.T_eff
              
        together_frd=[]
        together_aft=[]
        for i in range(num_thr):
            if T_eff[i]!=0:
                if thr_data[i][0]=="tunnel" and thr_data[i][3]>0:
                    together_frd.append(i)
                    
                elif thr_data[i][0]=="tunnel" and thr_data[i][3]<0:
                    together_aft.append(i)
        ith=[]
        for i in range(num_thr):
            if T_eff[i]!=0:
                ith.append(i)
                
        g_add=[]
        h_add=[]
        
        if len(together_frd)>1:
            g_tt_posf=np.zeros((len(together_frd),2*len(ith))) 
            g_tt_negf=np.zeros((len(together_frd),2*len(ith)))
            h_tt_f=np.zeros((len(together_frd)))
            
            for j in range(len(together_frd)):
                thr=together_frd[j]
                pos=ith.index(thr)
                g_tt_posf[j][2*pos+1]=-1  
                g_tt_negf[j][2*pos+1]=1 
            
            h_add.append(h_tt_f)
            h_add.append(h_tt_f)
            g_add.append(g_tt_posf)
            g_add.append(g_tt_negf)
                
        
        if len(together_aft)>1:
            g_tt_posa=np.zeros((len(together_aft),2*len(ith))) 
            g_tt_nega=np.zeros((len(together_aft),2*len(ith))) 
            h_tt_a=np.zeros((len(together_aft)))
            
            for j in range(len(together_aft)):
                thr=together_aft[j]
                pos=ith.index(thr)
                g_tt_posa[j][2*pos+1]=-1  
                g_tt_nega[j][2*pos+1]=1  
            
            
            h_add.append(h_tt_a)
            h_add.append(h_tt_a)
            g_add.append(g_tt_posa)
            g_add.append(g_tt_nega)
       
        #h_add=[]
        #g_add=[]
        return g_add,h_add
    
    def new_G_new_h(self):
        h_=self.h_
        G_=self.G_
        g_add, h_add=self.add_rows()
        
        div=len(h_add)
        
        if len(g_add)!=0:
            G_new=list(product(G_,g_add))
            h_new=list(product(h_,h_add))
            
            for i in range(len(G_new)):
                G_new[i]=np.vstack(G_new[i])
                h_new[i]=np.hstack(h_new[i])
                
        else:
            G_new=G_
            h_new=h_
       
        return G_new, h_new,div
        
class propellers:
    def __init__(self,ax,ay,a,num_thr,thr_data,T_eff):
        self.ax=ax
        self.ay=ay
        self.a=a
        self.num_thr=num_thr
        self.thr_data=thr_data
        self.T_eff=T_eff
        
    def con(self,i):
        
        thr_data=self.thr_data
        ax=self.ax
        ay=self.ay
        a=self.a
        T_eff=self.T_eff
        
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
        
        G_i=[]
        h_i=[]
        for j in range(len(a[i])-1):
            Te=T_eff[i]
            Tx=((Te*ax[i][j])**2)**0.5
            Ty=((Te*ay[i][j])**2)**0.5
            
            G_j=[[1,0],[0,1],[-1,0],[0,-1]]
            h_j=[Tx,Ty,0,Ty]
            
            G_i.append(G_j)
            h_i.append(h_j)
        
        Tx=((T_eff[i]*loss_factor)**2)**0.5
        G_j=[[-1,0],[1,0]]
        h_j=[Tx,0]
        
        G_i.append(G_j)
        h_i.append(h_j)
            
        return G_i, h_i
    
    def con_all(self):
        
        num_thr=self.num_thr
        T_eff=self.T_eff
        thr_data=self.thr_data
        
        G_prop=[]
        h_prop=[]
        
        for i in range(num_thr):
            thr_type=thr_data[i][0]
            if T_eff[i]!=0 and (thr_type=='propeller FPP' or thr_type=='propeller FPP nozzle' or \
                                thr_type=='propeller CPP' or thr_type=='propeller CPP nozzle' or \
                                    thr_type=='propeller CRP'):
                G_i,h_i=self.con(i)
                
                G_prop.append(G_i)
                h_prop.append(h_i)
       
        return G_prop, h_prop
    
    def combo_ax_ay(self,op,pp):
        
        ax=self.ax
        ay=self.ay
        num_thr=self.num_thr
        thr_data=self.thr_data
        T_eff=self.T_eff
        
        
        ith=[]
        for i in range(num_thr):
            if T_eff[i]!=0:
                ith.append(i)
        
        list_All=[]
        for i in range(len(ith)):
            list_a=[]
            ind=ith[i]
            if ind in pp:
                for j in range(len(ax[ind])):
                    item=[ax[ind][j],ay[ind][j]]
                    list_a.append(item)
                
                list_All.append(list_a)
        
        combo_thr_prop=[]       
        for i in range(len(ith)):
            combo_thr_prop.append([])
            
        for i in range(len(ith)):
            ind=ith[i]
            if ind not in pp:
                for j in range(op[ind]):
                    combo_thr_prop[i].append([])
        
        for i in range(len(pp)):
            ind=ith.index(pp[i])
            combo_thr_prop[ind]=list_All[i]
        
        combo_thr_prop=list(product(*combo_thr_prop))
        
        for i in range(len(combo_thr_prop)):
            prop_dummy=[]
            for j in range(len(combo_thr_prop[i])):
                if len(combo_thr_prop[i][j])!=0:
                    prop_dummy.append(combo_thr_prop[i][j])
                    
            if len(prop_dummy)==0: 
                combo_thr_prop[i]=[]
               
        return combo_thr_prop
        
class concatenate_Gh:
    def __init__(self,G_prop,h_prop,G,h,T_eff, thr_data):
        self.G_prop=G_prop
        self.h_prop=h_prop
        self.G=G
        self.h=h
        self.thr_data=thr_data
        self.T_eff=T_eff
        
    def combain(self):
        
        G_prop=self.G_prop
        h_prop=self.h_prop
        G=self.G
        h=self.h
        T_eff=self.T_eff
        thr_data=self.thr_data
        
        
        pp=[]
        for i in range(len(T_eff)):
            thr_type=thr_data[i][0]
            if T_eff[i]!=0 and (thr_type=='propeller FPP' or thr_type=='propeller FPP nozzle' or \
                                thr_type=='propeller CPP' or thr_type=='propeller CPP nozzle' or \
                                    thr_type=='propeller CRP'):
                pp.append(i)
        
        G_final=[]
        h_final=[]
        op=[]
        
        
        pos_prop=np.zeros((len(G)))
        for i in range(len(G)):
            for j in range(len(pp)):
                if pp[j]==i:
                    pos_prop[i]=1
                   
        for i in range(len(pos_prop)):
            if pos_prop[i]==1:
                G_final.append(G_prop[i])
                h_final.append(h_prop[i])
                op.append(len(G_prop[i]))
            else:
                G_final.append(G[i])
                h_final.append(h[i])
                op.append(len(G[i]))
        
        
        return G_final, h_final,op,pp
            
            
            
            
        