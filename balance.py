"""
Balence constraint
"""
import numpy as np
from itertools import product

class balance:
    def __init__(self,thr_data,num_thr,T_eff,b_):
        self.thr_data=thr_data
        self.num_thr=num_thr
        self.T_eff=T_eff
        self.b_=b_
    
    def balance_matrix(self):
        thr_data=self.thr_data
        num_thr=self.num_thr
        T_eff=self.T_eff
        
        A=[]
        a=0
        p=0
        
        list_thr=[]
        for i in range(len(T_eff)):
            if T_eff[i]!=0:
                list_thr.append(i)
                
        for j in range(len(list_thr)):
            A.append([])
            i=list_thr[j]
            type_thr=thr_data[i][0]
            x=thr_data[i][3]
            y=thr_data[i][4]   
            
            thr_type=type_thr
            
            if type_thr=='tunnel':
                balance=[[0,0],[0,1],[-y,x]]
                a=a+1
                
            elif thr_type=='propeller FPP' or thr_type=='propeller FPP nozzle' or thr_type=='propeller CPP' or thr_type=='propeller CPP nozzle' or   thr_type=='propeller CRP':
                        
                balance=[[1,0],[0,1],[-y,x]]
                p=p+1
            else:
                balance=[[1,0],[0,1],[-y,x]]
           
            A[j]=balance
            
        A=np.hstack(A)
        
        add=np.zeros((a,len(list_thr)*2)) 
        
        a=0
        for j in range(len(list_thr)):
            i=list_thr[j]
            type_thr=thr_data[i][0]            
            if type_thr=='tunnel':
                a=a+1
                n=2*j
                add[a-1,n]=1
        
        A=np.vstack((A,add))     
       
        return A
    
    def b_array(self):
        b_=self.b_
        thr_data=self.thr_data
        T_eff=self.T_eff
        
        list_thr=[]
        for i in range(len(T_eff)):
            if T_eff[i]!=0:
                list_thr.append(i)
         
                
        a=0
        for j in range(len(list_thr)):
            i=list_thr[j]
            type_thr=thr_data[i][0]            
            if type_thr=='tunnel':
                a=a+1
        b=b_       
        for j in range(a):
            b=np.append(b,0)
        
        prop=[]
        for j in range(len(list_thr)):
            i=list_thr[j]
            thr_type=thr_data[i][0]
            
            if thr_type=='propeller FPP' or thr_type=='propeller FPP nozzle' or  thr_type=='propeller CPP' or thr_type=='propeller CPP nozzle' or  thr_type=='propeller CRP':
                        
                prop.append(0)
        prop=np.array(prop)
         
        b=np.append(b,prop)
        
        return b
    
    def add_lines(self,at):
        
        thr_data=self.thr_data
        num_thr=self.num_thr
        A=self.balance_matrix()
        T_eff=self.T_eff
        
        list_i=[]
        for i in range(len(T_eff)):
            if T_eff[i]!=0:
                list_i.append(i)
                
        all_lines=[]
        
        for i in range(len(list_i)):
            j=list_i[i]
            thr_type=thr_data[j][0]
        
            if T_eff[j]!=0 and (thr_type=='propeller FPP' or thr_type=='propeller FPP nozzle' or  thr_type=='propeller CPP' or thr_type=='propeller CPP nozzle' or thr_type=='propeller CRP'):
                
                a_i=at[j]
                
                i_th_thr=self.add_i(j,list_i,a_i)
                
                all_lines.append(i_th_thr)
        
        return all_lines
            
    def add_i(self,i,list_i,a_i):
        
        index=list_i.index(i)                
        ix=index*2
        iy=index*2+1
        
        i_th_thr=[]
        for k in range(3):
            prep=np.zeros((len(list_i)*2))
            
            if k<2:
                prep[ix]=0
                prep[iy]=0
            else:
                prep[ix]=0
                prep[iy]=1
                
                
            i_th_thr.append([prep])
        
        
        return i_th_thr
    
    def combination_A(self,A,all_lines,op,pp):
        
        new_A=[]
        A=np.array(A)
        ith=[]
        for i in range(len(op)):
            if op[i]!=0:
                ith.append(i)
        
        for i in range(len(ith)):
            new_A.append([])
            
        for i in range(len(ith)):
            if ith[i] not in pp:
                for j in range(op[ith[i]]):
                    new_A[i].append([])
        
        for i in range(len(pp)):
            ind=ith.index(pp[i])
            new_A[ind]=all_lines[i]
        
        A_comb=list(product(*new_A))
        
        for i in range(len(A_comb)):
            A_dummy=[]
            for j in range(len(A_comb[i])):
                if len(A_comb[i][j])!=0:
                    A_dummy.append(A_comb[i][j])
                    
            if len(A_dummy)!=0: 
                A_comb[i]=np.vstack(np.array(A_dummy))
            else:
                A_comb[i]=[]
        
        for i in range(len(A_comb)):
            if len(A_comb[i])==0:
                A_comb[i]=A
            else:
                A_comb[i]=np.vstack((A,A_comb[i]))
        
        
        return A_comb
            
            
       
    