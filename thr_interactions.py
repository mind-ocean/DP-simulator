"""
Here we define the existing interactions between all thrusters and the skeg
"""
import numpy as np
import multirange as mr
from itertools import product, permutations

"""
How are interatcions - general information
"""
class thr_int:
    def __init__(self,thr_data,num_thr,y_skeg,x_skeg):
        self.thr_data=thr_data
        self.num_thr=num_thr        
        self.y_skeg=y_skeg
        self.x_skeg=x_skeg
    
        
    def type_thr(self):
        # create a list of thruster types by numbers 0 - rotatable, 1 - rudder (fixed prop), 2 - tunnel
        
        thr_data=self.thr_data
        num_thr=self.num_thr
        
        type_thr_list=[]
        
        for i in range(num_thr):
            thr_type=thr_data[i][0]
            
            if thr_type=='propeller FPP' or thr_type=='propeller FPP nozzle' or \
                thr_type=='propeller CPP' or thr_type=='propeller CPP nozzle' or \
                    thr_type=='propeller CRP' or thr_type=='cyclo':
                        
                type_thr_list.append(1)
                
            elif thr_type=="tunnel":
                type_thr_list.append(2)
                
            else:
                type_thr_list.append(0)
        
        return type_thr_list
    
    def on_off_thr(self):
        # create a list of thrusters on (1) and off (0) based on the power value
        
        thr_data=self.thr_data
        num_thr=self.num_thr
        
        on_off_thr_list=[]
        
        for i in range(num_thr):
            power=thr_data[i][7]
            
            if power==0:
                on_off_thr_list.append(0)
            
            else:
                on_off_thr_list.append(1)
        
        return on_off_thr_list
    
    def skeg_int(self):
        # creates a list of thrusters interacting with the skeg/skegs
        
        y_skeg=self.y_skeg
        x_skeg=self.x_skeg
        num_thr=self.num_thr
        
        on_off_thr_list=self.on_off_thr()
        type_thr_list=self.type_thr()
        
        num_skeg=len(y_skeg)
        
        skegs_int=[]
        for j in range(num_skeg):            
            y_skeg_j=y_skeg[j]
            x_skeg_j=x_skeg[j]
            
            thr_skeg_int_list=self.skeg_int_j(x_skeg_j,y_skeg_j)
            
            for i in range(num_thr):
                on_off=on_off_thr_list[i]
                type_azi=type_thr_list[i]
                skeg_on=thr_skeg_int_list[i]
                
                if on_off==1 and type_azi==0 and skeg_on==1 and x_skeg_j!=0:
                    thr_skeg_int_list[i]=1
                else:
                    thr_skeg_int_list[i]=0
                    
            skegs_int.append(thr_skeg_int_list)
            
        return skegs_int
            
        
    def skeg_int_j(self, x_skeg_j,y_skeg_j):
        # creates a list for each skeg with 0 and 1 values for each thrusters indicating potential skeg interaction with it
        # not taking into account the type of the thruster or that it is not working - just the position x, y relative to the skeg position
        
        num_thr=self.num_thr
        thr_data=self.thr_data
        
        
        thr_skeg_int_list=[]
        for i in range(num_thr):
            y_thr=thr_data[i][4]             
            x_thr=thr_data[i][3]
            D=thr_data[i][6]
            type_thr=thr_data[i][0]
            
            if y_thr!=y_skeg_j and x_thr<0:
                
                if type_thr=="azi nozzle" or type_thr=="pod nozzle":
                    dist_min=8*D
                else:
                    dist_min=15*D
                
                if x_skeg_j<=x_thr:
                    s=abs(y_skeg_j-y_thr)                
                else:
                    s=((x_thr-x_skeg_j)**2+(y_thr-y_skeg_j)**2)**0.5
               
                if s<dist_min:
                    thr_skeg_int_list.append(1)      
                else:
                    thr_skeg_int_list.append(0)  
                    
            else:
                thr_skeg_int_list.append(0)
                
        return thr_skeg_int_list
    
    def work_thr_thr(self):
        # gives a list of thrusters interacting with other thrusters (each with each) if they are working thr
        
        num_thr=self.num_thr
        thr_data=self.thr_data
        
        work_list=self.on_off_thr()
        type_thr_list=self.type_thr()
        
        thr_thr_fb=[]
        for i in range(num_thr):
            
            if work_list[i]==1:
                
                x_thr_i=thr_data[i][3]
                y_thr_i=thr_data[i][4]
                D=thr_data[i][6]
                
                
                for j in range(num_thr):
                    
                    if i!=j and type_thr_list[i]==0 and type_thr_list[j]!=2 and work_list[j]==1:
                        x_thr_j=thr_data[j][3]
                        y_thr_j=thr_data[j][4]
                        
                        int_on=self.int_fb(x_thr_i,y_thr_i,x_thr_j,y_thr_j,D)
                        
                        if int_on==1:
                            pair=[i,j]
                            thr_thr_fb.append(pair)
        
        return thr_thr_fb
                              
    
    def int_fb(self,x_thr_i,y_thr_i,x_thr_j,y_thr_j,D):
        # decide if the distance is enough to implement a forbidden zone - if the thr i with diameter D flushes th j
       
        s=((x_thr_i-x_thr_j)**2+(y_thr_i-y_thr_j)**2)**0.5
                
        if s<15*D:
            int_on=1        
        else:
            int_on=0
            
        return int_on
    
    def dead_thr_thr(self):
        # gives a list of thrusters interacting with other dead thrusters 
        
        num_thr=self.num_thr
        thr_data=self.thr_data
        
        work_list=self.on_off_thr()
        type_thr_list=self.type_thr()
        
        thr_thr_dead=[]
        for i in range(num_thr):
            
            if work_list[i]==1:
                
                x_thr_i=thr_data[i][3]
                y_thr_i=thr_data[i][4]
                D=thr_data[i][6]
                
                
                for j in range(num_thr):
                    
                    if i!=j and type_thr_list[i]==0 and type_thr_list[j]!=2 and work_list[j]==0:
                        x_thr_j=thr_data[j][3]
                        y_thr_j=thr_data[j][4]
                        
                        int_on=self.int_dead(x_thr_i,y_thr_i,x_thr_j,y_thr_j,D,i)
                        
                        if int_on==1:
                            pair=[i,j]
                            thr_thr_dead.append(pair)
                            
        return thr_thr_dead
                            
    def int_dead(self,x_thr_i,y_thr_i,x_thr_j,y_thr_j,D,i):
        # decide if the distance is enough to implement a loss zone - if the thr i with diameter D flushes dead thr j
        
        thr_data=self.thr_data
        
        type_thr=thr_data[i][0]
        
        if type_thr=="azi nozzle" or type_thr=="pod nozzle":
            dist_min=4*D
        else:
            dist_min=8*D
        
        
        s=((x_thr_i-x_thr_j)**2+(y_thr_i-y_thr_j)**2)**0.5
                
        if s<dist_min:
            int_on=1        
        else:
            int_on=0
            
        return int_on
        
    
"""
Specific points of the interaction linear functons
"""
        
class thr_int_points:
    def __init__(self, thr_thr, thr_dead, thr_skeg, thr_data, num_thr, x_skeg, y_skeg):
        self.thr_thr=thr_thr
        self.thr_dead=thr_dead
        self.thr_skeg=thr_skeg
        self.thr_data=thr_data
        self.num_thr=num_thr
        self.x_skeg=x_skeg
        self.y_skeg=y_skeg
        
    def exist_fun(self):
        #for each thruster specify the interactions [thr_thr,thr_dead,thr_skeg], and [[fb][dead][skeg]]
        
        thr_thr=self.thr_thr
        thr_dead=self.thr_dead
        thr_skeg=self.thr_skeg
        num_thr=self.num_thr
        
        int_gather_all_thr=[]
        
        specific_int=[]

        for i in range(num_thr):
            
            int_list=[0,0,0]
            
            line=[]
            
            fb=[]
            dead=[]
            skeg=[]
            
            if thr_thr!=[]:
            
                for j in range(len(thr_thr)):
                    
                    if thr_thr[j][0]==i:
                        fb.append(thr_thr[j][1])
                        int_list[0]=1
             
            if thr_dead!=[]:
                
                for j in range(len(thr_dead)):
                   
                    if thr_dead[j][0]==i:
                        dead.append(thr_dead[j][1])
                        int_list[1]=1
             
            if thr_skeg!=[]:
                
                for j in range(len(thr_skeg)):
                    
                    if thr_skeg[j][i]==1:
                        skeg.append(1)
                        int_list[2]=1
                    else:
                        skeg.append(0)
                        
            line.append(fb)
            line.append(dead)
            line.append(skeg)
            
            specific_int.append(line)
                
            int_gather_all_thr.append(int_list)
            
        
        return specific_int, int_gather_all_thr
    
    def thr_skeg_points(self):
        # gives a set o points describing the skeg loss for each thrusters as a list collection of arrays
        
        num_thr=self.num_thr
        specific_int, int_gather_all_thr=self.exist_fun()
        
        skeg_loss_fun_points=[]
        
        for i in range(num_thr):
            if int_gather_all_thr[i][2]==1:
                skeg_loss_fun_points.append((self.skeg_fun_i(i)))
            else:
                skeg_loss_fun_points.append([])
                
        return skeg_loss_fun_points     
        
    
    def skeg_fun_i(self,i):
        # for each i-th thruster it find all the points describing the linear function of loss for the skeg
        
        specific_int, int_gather_all_thr=self.exist_fun()
        
        all_points_skeg=[]
        for j in range(len(specific_int[i][2])):
            if specific_int[i][2][j]==1:
                num_skeg=j
                skeg_loss_points=self.skeg_fun_i_j(i, num_skeg)
                
                all_points_skeg.append(skeg_loss_points)
        
        
        return all_points_skeg
        
    
    def skeg_fun_i_j(self,i,num_skeg):
        # calculates the points function for a thr i with the skeg j
        
        thr_data=self.thr_data
        x_skeg=self.x_skeg
        y_skeg=self.y_skeg
        
        x_skeg_j=x_skeg[num_skeg]
        y_skeg_j=y_skeg[num_skeg]
        
        x_thr=thr_data[i][3]
        y_thr=thr_data[i][4]
        D=thr_data[i][6]
        
        if x_skeg_j>x_thr:
            s=((x_thr-x_skeg_j)**2+(y_thr-y_skeg_j)**2)**0.5
        else:
            s=((y_skeg_j-y_thr)**2)**0.5
        
        
        if y_thr>y_skeg_j:
            port_side=1
        else:
            port_side=0
            
        if port_side==1:
            skeg_loss_points=self.port_skeg(x_thr, y_thr, x_skeg_j, y_skeg_j,D,s)
            
        else: # stb side
            skeg_loss_points=self.stb_skeg(x_thr, y_thr, x_skeg_j, y_skeg_j,D,s)
        
        
        
        return skeg_loss_points
            
    
    def port_skeg(self, x_thr, y_thr, x_skeg_j, y_skeg_j,D,s):
        # calculates a set of points describing the function for thr to the port of the j-th skeg, DNVGL
        
        alfa_flush=(np.pi/2)-np.arctan((x_skeg_j-x_thr)/(y_skeg_j-y_thr))
        alfa_jet=np.arctan(0.6*D/s)
        alfa_max_loss=min(max((alfa_flush+alfa_jet),np.pi/2),np.pi)
        
        thrust_loss_skeg=2*alfa_max_loss/np.pi-1
        
        alfa_start_loss=max((alfa_flush-alfa_jet),0)
        alfa_end_loss=min((alfa_max_loss+4*alfa_jet),np.pi)
        
        
        points_x=[0,alfa_start_loss,alfa_max_loss,alfa_end_loss,2*np.pi]
        points_y=[1,1,thrust_loss_skeg,1,1]
        
        skeg_loss_points=self.points_skeg(points_x, points_y)
        
        
        return skeg_loss_points
    
    def stb_skeg(self,x_thr, y_thr, x_skeg_j, y_skeg_j,D,s):
        # calculates a set of points describing the function for thr to the stb of the j-th skeg, DNVGL
        
        alfa_flush=(3*np.pi/2)-np.arctan((x_skeg_j-x_thr)/(y_skeg_j-y_thr))
        alfa_jet=np.arctan(0.6*D/s)
        alfa_max_loss=max(min((alfa_flush-alfa_jet),(3*np.pi/2)),np.pi)
        
        thrust_loss_skeg=3-2*alfa_max_loss/np.pi
        
        alfa_start_loss=max((alfa_max_loss-4*alfa_jet),np.pi)
        alfa_end_loss=min((alfa_flush+alfa_jet),2*np.pi)
       
            
        points_x=[0,alfa_start_loss,alfa_max_loss,alfa_end_loss,2*np.pi]
        points_y=[1,1,thrust_loss_skeg,1,1]
        
        skeg_loss_points=self.points_skeg(points_x, points_y)
        
        return skeg_loss_points
    
            
    def points_skeg(self,points_x, points_y):
        # changes the radians into degrees and create a points list that describe the linear funciton of the loss
        
        skeg_points=[]
        
        for k in range(len(points_x)):
            points_x[k]=np.degrees(points_x[k])
            point=[points_x[k],points_y[k]]
            
            skeg_points.append(point)
        
        return skeg_points
    
    def thr_thr_points(self):
        # gives the whole list of all thrusters and their interactions with other thrusters
        
        num_thr=self.num_thr
        specific_int, int_gather_all_thr=self.exist_fun()
        
        thr_thr_points=[]
        
        for i in range(num_thr):
            if int_gather_all_thr[i][0]==1:
                thr_thr_points.append((self.thr_thr_i(i)))
            else:
                thr_thr_points.append([])
                
        return thr_thr_points    
    
    def thr_thr_i(self,i):
        
        specific_int, int_gather_all_thr=self.exist_fun()
        
        all_fb_points=[]
        if len(specific_int[i][0])!=0:
            
            for j in range(len(specific_int[i][0])):
                num_thr_flushed=specific_int[i][0][j]
                fb_points=self.thr_thr_i_j(i,num_thr_flushed)
                all_fb_points.append(fb_points)
         
                    
        return all_fb_points           
    
    def thr_thr_i_j(self,i,j):
        # give the points for each i-th thruster flushing j-th thruster
        
        thr_data=self.thr_data
        
        
        x_thr_i=thr_data[i][3]
        y_thr_i=thr_data[i][4]
        D=thr_data[i][6]
        
        x_thr_j=thr_data[j][3]
        y_thr_j=thr_data[j][4]
        
        s=((x_thr_i-x_thr_j)**2+(y_thr_i-y_thr_j)**2)**0.5
        alfa_range=np.arctan(0.1+1*D/s)
        
        if y_thr_i>y_thr_j:
            port_thr=1
        else:
            port_thr=0
            
        if port_thr==1:
            thrust_dir=np.pi/2-np.arctan((x_thr_j-x_thr_i)/(y_thr_j-y_thr_i))
            
            
        else:
            if (y_thr_j-y_thr_i)==0:
                if x_thr_j>x_thr_i:
                    thrust_dir=0
                else:
                    thrust_dir=np.pi
            else:
                thrust_dir=3*np.pi/2-np.arctan((x_thr_j-x_thr_i)/(y_thr_j-y_thr_i))
            
        alfa_start_fb=thrust_dir-alfa_range
        alfa_end_fb=thrust_dir+alfa_range
        
       
        points_x=[0,alfa_start_fb,alfa_start_fb,alfa_end_fb,alfa_end_fb,2*np.pi]
        points_y=[1,1,0,0,1,1]
        
        fb_points=self.points_fb(points_x, points_y)
        
        return fb_points
        
        
    def points_fb(self,points_x, points_y):
        # changes the radians into degrees and create a points list that describe the linear funciton of the loss
        
        fb_points=[]
        
        for k in range(len(points_x)):
            points_x[k]=np.degrees(points_x[k])
            point=[points_x[k],points_y[k]]
            
            fb_points.append(point)
            
        return fb_points
    
    def thr_dead_points(self):
        # gives the whole list of all thrusters and their interactions with other thrusters
        
        num_thr=self.num_thr
        specific_int, int_gather_all_thr=self.exist_fun()
        
        thr_dead_points=[]
        
        for i in range(num_thr):
            if int_gather_all_thr[i][1]==1:
                thr_dead_points.append((self.thr_dead_i(i)))
            else:
                thr_dead_points.append([])
                
        return thr_dead_points    
    
    def thr_dead_i(self,i):
        # get all teh points of dead thr interactions for i-th thruster
        
        specific_int, int_gather_all_thr=self.exist_fun()
        
        all_dead_points=[]
        if len(specific_int[i][1])!=0:
            
            for j in range(len(specific_int[i][1])):
                num_thr_flushed=specific_int[i][1][j]
                dead_points=self.dead_i_j(i,num_thr_flushed)
                all_dead_points.append(dead_points)
        
        return all_dead_points
         
    
    def dead_i_j(self,i,j):
        # calculates points for i-th thruster for each j-th dead thruster
        
        thr_data=self.thr_data
        
        
        x_thr_i=thr_data[i][3]
        y_thr_i=thr_data[i][4]
        D=thr_data[i][6]
        type_thr=thr_data[i][0]
        
        x_thr_j=thr_data[j][3]
        y_thr_j=thr_data[j][4]
        
        s=((x_thr_i-x_thr_j)**2+(y_thr_i-y_thr_j)**2)**0.5
        
        if type_thr=="azi nozzle" or type_thr=="pod nozzle":
            alfa_range=np.arctan(0.35*D/s)
        else:
            alfa_range=np.arctan(0.6*D/s)
        
        if y_thr_i>y_thr_j:
            port_thr=1
        else:
            port_thr=0
            
        if port_thr==1:
            thrust_dir=np.pi/2-np.arctan((x_thr_j-x_thr_i)/(y_thr_j-y_thr_i))
            
        else:
            if (y_thr_j-y_thr_i)!=0:
                thrust_dir=3*np.pi/2-np.arctan((x_thr_j-x_thr_i)/(y_thr_j-y_thr_i))
            
            else:
                if x_thr_j>x_thr_i:
                    thrust_dir=0
                else:
                    thrust_dir=np.pi
            
          
        alfa_start_dead=thrust_dir-alfa_range
        alfa_end_dead=thrust_dir+alfa_range
        
        thrust_loss=1-(1/(0.02*(s/D)**2+0.25*s/D+1.2))
        
        points_x=[0,alfa_start_dead,thrust_dir,alfa_end_dead,2*np.pi]
        points_y=[1,1,thrust_loss,1,1]
        
        dead_points=self.points_dead(points_x, points_y)
        
        return dead_points
        
        
    def points_dead(self,points_x, points_y):
        # changes the radians into degrees and create a points list that describe the linear funciton of the loss
        
        dead_points=[]
        
        for k in range(len(points_x)):
            points_x[k]=np.degrees(points_x[k])
            point=[points_x[k],points_y[k]]
            
            dead_points.append(point)
        
        return dead_points
    
    def joined_points(self):
        # gives joined points describing the losses and forbiden zone functions for each thrusters separatelly in the array
        
        num_thr=self.num_thr
        
        points=[]
        
        for i in range(num_thr):
            points.append(self.joined_points_i(i))
        
        return points
            
    
    def joined_points_i(self,i):
        
        all_fb_points=self.thr_thr_points()
        all_dead_points=self.thr_dead_points()
        all_skeg_points=self.thr_skeg_points()
        
        specific_int, int_gather_all_thr=self.exist_fun()
        
        all_points_fun_i=[]
        if int_gather_all_thr[i][0]==1:
            all_points_fun_i.append(all_fb_points[i])
        else:
            all_points_fun_i.append([])
            
        if int_gather_all_thr[i][1]==1:
            all_points_fun_i.append(all_dead_points[i])
        else:
            all_points_fun_i.append([])
            
        if int_gather_all_thr[i][2]==1:
            all_points_fun_i.append(all_skeg_points[i])
        else:
            all_points_fun_i.append([])
            
        
        
        return all_points_fun_i
    
"""
New points for each thruster
"""
class new_points:
    def __init__(self, points,num_thr,int_gather_all_thr):
        self.points=points
        self.num_thr=num_thr
        self.int_gather_all_thr=int_gather_all_thr
        
    def all_thr_int_points(self):
        
        num_thr=self.num_thr
        int_gather_all_thr=self.int_gather_all_thr
        
        fun=[]
        for i in range(num_thr):
            int_i=int_gather_all_thr[i]
            points_i=self.cases(int_i,i)
            
            if points_i!=[]:
                
                points_i=self.sorting(1,points_i)
                
                for j in range(len(points_i)):
                    if points_i[j][0]<0:
                        points_i[j][0]=360+points_i[j][0]
                    
                if points_i[0][0]>points_i[len(points_i)-2][0]:
                    ind=points_i.index([0,1])
                    points_i[ind]=[0,0]
                    ind=points_i.index([360,1])
                    points_i[ind]=[360,0]
                
                    
                    points_i=sorted(points_i)
                    
                    points_i=self.sorting(0,points_i)
                    
                    if points_i[1][0]==0:
                        points_i.remove([0,0])
                        points_i.remove([360,0])
                        points_i.append([360,points_i[0][1]])
            
            clean_list=[]
            for n in range(len(points_i)):
                if points_i[n] not in clean_list:
                    clean_list.append(points_i[n])
                    
            fun.append(clean_list)
            
            
            
        return fun
    
    def sorting(self,open_j,points_i):
        
        points_i_sort=[]
        open_j=open_j
        jj=0
        for j in range(len(points_i)-1):
            jj=jj+1
            if points_i[j][0]!=points_i[j+1][0]:
                
                if points_i_sort!=[]:
                    
                    exist=0
                    for k in range(len(points_i_sort)):
                        if points_i[j]==points_i_sort[k]:
                            exist=1
                    if exist==0:
                        
                        points_i_sort.append(points_i[j])
                else:
                    points_i_sort.append(points_i[j])
                    
            else:
                if open_j==1:
                    open_j=0
                else:
                    open_j=1
                
                if open_j==0:
                    points_i_sort.append(points_i[j+1])
                    points_i_sort.append(points_i[j])
                    
                else:
                    points_i_sort.append(points_i[j])
                    points_i_sort.append(points_i[j+1])
                    
        points_i_sort.append(points_i[jj])            
        points_i=points_i_sort  
        
        return points_i
    
    def cases(self,int_i,i):
        
        points=self.points
        
        a=int_i
        
        if a[0]==0 and a[1]==0 and a[2]==0:
            points_i=[]
            
            
        elif a[0]==1 and a[1]==0 and a[2]==0: #fb
            angles=self.forbidden_range(i)
        
            points_i=[]
            for j in range(len(angles)):
                points_j=[[angles[j][0],1],[angles[j][0],0],[angles[j][1],0],[angles[j][1],1]]
                for k in range(len(points_j)):
                    points_i.append(points_j[k])
            
            points_i.append([0,1])
            points_i.append([360,1])
            points_i=list(points_i)
            points_i=sorted(points_i)
        
            
        elif a[0]==0 and (a[1]==1 or a[2]==1):
            points_i=self.add_points_line_1(i)
            
            
        elif a[0]==1 and (a[1]==1 or a[2]==1):
            points_i=self.final_bucket(i)
       
            
        return points_i
        
    def a_b(self,point_0, point_1):
        # specify a and b based on the given = closest located points
        
        x0=point_0[0]
        y0=point_0[1]
        
        x1=point_1[0]
        y1=point_1[1]
        
        if x0==x1 and y1==1 and y0==1:
            a=0.0
            b=1.0
        else:
            a=(y1-y0)/(x1-x0)
            b=(y0*x1-y1*x0)/(x1-x0)
        
        return a, b
        
    
    def coord(self,fun_1, fun_2):
        # get the cross point
       
        a1=fun_1[0]
        b1=fun_1[1]
        
        a2=fun_2[0]
        b2=fun_2[1]
       
        x=(b2-b1)/(a1-a2)
        
        y=self.loss(x, a1, b1)
        
        point=[x,y]
        
        return point
    
    def loss(self,x, a, b):
        # get the function value y = loss from a specific x and a curve a, b
        
        y=a*x+b
        
        return y
    
    def forbidden_range(self,i):
        # a complicated around way to get the forbidden ranges
        
        points=self.points
        
        ranges_int=[]
        ranges=[]
        if points[i][0]!=[]:
            for j in range(len(points[i][0])):
                x_start=points[i][0][j][2][0]
                x_end=points[i][0][j][3][0]
                ranges.append(x_start)
                ranges.append(x_end)
                
                range_x=range(int(np.floor(x_start)),int(np.ceil(x_end)))
                ranges_int.append(range_x)
        
        new=list(mr.normalize_multi(ranges_int))
        
        new_range=[]
        range_final_int=[]
        for k in range(len(new)):
            
            start=min(new[k])
            end=max(new[k])+1
            
            ran=[start,end]
            range_final_int.append(ran)
            new_range=range_final_int
        
        return new_range
    
    
    def curves(self,i):
        # combain all the functions defined by points for dead thr and skeg for each i-th thr
        
        points=self.points
        
        curves_list=[]
        for j in range(len(points[i][1])):
            curves_list.append(points[i][1][j])
            
        for j in range(len(points[i][2])):
            curves_list.append(points[i][2][j])
        
        return curves_list   
    
    def intersections(self,i):
        # finds a, b describing lines - all existing lines except for y=1
        
        curves_list=self.curves(i)
        
        lines=[]
        if curves_list!=[]:
            for j in range(len(curves_list)):
                for k in range(2):
                    point_0=curves_list[j][k+1]
                    point_1=curves_list[j][k+2]
                    
                    a,b=self.a_b(point_0, point_1)
                    
                    lines.append([a,b])
        
        return lines

    def points_bucket(self,i):
        # finds all the intersection points
        
        lines=self.intersections(i)
        
        num_lines=len(lines)
        
        all_exist_points=[]
        for j in range(num_lines):
            for k in range(num_lines):
                if k>j:
                    fun_1=lines[j]
                    fun_2=lines[k]
                    
                    if fun_1!=fun_2:
                        point=self.coord(fun_1,fun_2)
                        
                        point[0]=int(round(point[0]))
                        if round(point[1],1)==1:
                            point[1]=1
                          
                            
                        point=[point[0],round(point[1],3)]
                        
                        all_exist_points.append(point)
                    
        return all_exist_points
    
    def box(self,i):
        # gives all the points in the boundary box below
        
        points_all=self.points_bucket(i)
        curves_list=self.curves(i)
        
        losses=[]
        for j in range(len(curves_list)):
            for k in range(len(curves_list[j])):
                loss=curves_list[j][k][1]
                losses.append(loss)
                
        min_y=min(losses)-0.005
        max_y=1
        
        min_x=0
        max_x=360
        
        in_the_box=[]
        for j in range(len(points_all)):
            if points_all[j][0]>=min_x and points_all[j][0]<=max_x and  points_all[j][1]>=min_y and points_all[j][1]<max_y:
                in_the_box.append(points_all[j])
                
        return in_the_box
    
    def add_points_line_1(self,i):
        # look for the sum of ranges in the boundary of a single loss curve points
        # and add thr boundaries to the rest of the points in the box, then sort by angle
        
        in_the_box=self.box(i)
        curves_list=self.curves(i)
        
        ranges_1=[]
        for j in range(len(curves_list)):
            start=int(round(curves_list[j][1][0]))
            end=int(round(curves_list[j][3][0]))
            
            range_j=range(start,end)
            ranges_1.append(range_j)
        
        new_ranges_1=list(mr.normalize_multi(ranges_1))
        
        min_max_range=[]
        for j in range(len(new_ranges_1)):
            min_max=[min(new_ranges_1[j]),max(new_ranges_1[j])+1]
            min_max_range.append(min_max)
            
        for j in range(len(min_max_range)):
            in_the_box.append([min_max_range[j][0],1])
            in_the_box.append([min_max_range[j][1],1])
        
        in_the_box=self.add_start_end_points(i,in_the_box)
        new_bucket=sorted(in_the_box)  
        
        return new_bucket
    
    def add_start_end_points(self,i,in_the_box):
        # add points [0,1] and [360,1] if not already included
        
        start=[0,1]
        end=[360,1]
        
        s=0
        e=0
        for j in range(len(in_the_box)) :
            if in_the_box[j]==start:
                s=1
            if in_the_box[j]==end:
                e=1
        if s==0:
            in_the_box.append(start)
        
        if e==0:
            in_the_box.append(end)
            
        all_in_the_box=in_the_box
        
        return all_in_the_box
    
    def cut_forbidden(self,i):
        # finds the y interpolated for the boundary 
        
        points_bucket=self.add_points_line_1(i)
        fb_range=self.forbidden_range(i)
                
        x,y=self.list_x_y(points_bucket)
        
        points_fb_all=[]
        for j in range(len(fb_range)):
            start=fb_range[j][0]
            end=fb_range[j][1]
            
            start_val=self.find_val(start,x,y)
            end_val=self.find_val(end,x,y)
            
            points_fb=[[start,start_val],[end,end_val],[start,0],[end,0]]
            points_fb_all.append(points_fb)
    
        return points_fb_all
    
    def find_val(self,val,x,y):
        # find the value y for a forbidden boundary range
               
        for k in range(len(x)-1):
            
            if val>=x[k] and val<=x[k+1]:
                
                y_val=round((y[k]+(y[k+1]-y[k])/(x[k+1]-x[k])*(val-x[k])),3)
                if round(y_val,1)==1:
                    y_val=1
       
        return y_val 
    
    def list_x_y(self,points):
        # create two lists x, y from points list
        
        x=[]
        y=[]
        for j in range(len(points)):
            
            x.append(points[j][0])
            y.append(points[j][1])
        
        return x, y
    
    def getout_points(self,i):
        # gives the points that are inside the forbidden zone:
            
        points_bucket=self.add_points_line_1(i)
        points_fb_all=self.cut_forbidden(i)
        
        x,y=self.list_x_y(points_bucket)
        
        out_points=[]
        for j in range(len(points_fb_all)):
            
            for k in range(len(x)):
                if x[k]>=points_fb_all[j][0][0] and x[k]<=points_fb_all[j][1][0]:
                    out_points.append([x[k],y[k]])
                
        return out_points
    
    def clear_points(self,i):
        # gives the points that are outside the fb zone
        
        out_points=self.getout_points(i)
        points_bucket=self.add_points_line_1(i)
        
        new_bucket=[]
        for j in range(len(points_bucket)):
            new_bucket.append(points_bucket[j])
        
        for j in range(len(points_bucket)):
            for k in range(len(out_points)):
                if out_points[k]==points_bucket[j]:
                    new_bucket.remove(out_points[k])
                    
        return new_bucket
    
    def final_bucket(self,i):
        # gives all the final points for each i-th thr
        
        new_bucket=self.clear_points(i)
        points_fb_all=self.cut_forbidden(i)
        
        bucket=new_bucket
        
        for j in range(len(points_fb_all)):
            bucket.append(points_fb_all[j][0])
            bucket.append(points_fb_all[j][1])
            bucket.append(points_fb_all[j][2])
            bucket.append(points_fb_all[j][3])
        
        bucket=sorted(bucket)
            
        return bucket
    

class multiloss_upgrade:
    def __init__(self,points):
        self.points=points
    
    def newnew(self):
        points=self.points
        thr=len(points)
       
        for i in range(thr):
            
            points=self.options(i)
    
        return points
    
    def options(self,i):
        points=self.points
        
        if len(points[i][1])!=0 and len(points[i][2])==0:
            print('Dead')
            if len(points[i][1])>1:
                dead_options=np.arange(0,len(points[i][1]))
                pairs=list(permutations(dead_options,2))
                
                t=[1,1]
                points=self.iter_pairs(t,pairs,points)
                    
        elif len(points[i][1])==0 and len(points[i][2])!=0:
            
            print('Skeg for thruster number: '+str(i))
            if len(points[i][2])>1:
                skeg_options=np.arange(0,len(points[i][2]))
                
                pairs=list(permutations(skeg_options,2))
                
                t=[2,2]
                points=self.iter_pairs(i,t,pairs,points)
                
                   
        elif len(points[i][1])!=0 and len(points[i][2])!=0:
            
            print('Both')
            if len(points[i][1])>1 and len(points[i][2])==1:
                dead_options=np.arange(0,len(points[i][1]))
                pairs=list(np.zeros(2,len(points[i][1])))
                for it in len(pairs):
                    pairs[dead_options[it]][0]=dead_options[it]
                
                t=[1,2]
                points=self.iter_pairs(t,pairs,points)
            
            if len(points[i][1])==1 and len(points[i][2])>1:
                skeg_options=np.arange(0,len(points[i][2]))
                pairs=list(np.zeros(2,len(points[i][2])))
                
                for it in len(pairs):
                    pairs[skeg_options[it]][1]=skeg_options[it]
                
                t=[1,2]
                points=self.iter_pairs(t,pairs,points)
                    
            if len(points[i][1])>1 and len(points[i][2])>1:
                
                dead_options=np.arange(0,len(points[i][1]))
                skeg_options=np.arange(0,len(points[i][2]))
                
                pairs=list(product(dead_options, skeg_options,2))
                
                t=[1,2]
                points=self.iter_pairs(t,pairs,points)
                    
        return points
            
    def iter_pairs(self,i,t,pairs,points):
        
        for n, pair in enumerate(pairs):
            
            list_1=points[i][t[0]][pair[0]]
            list_2=points[i][t[1]][pair[1]]
            
            x_1=self.list_x(list_1)
            x_2=self.list_x(list_2)
            
            y_new_1=self.inter(x_1,list_1,list_2)
            y_new_2=self.inter(x_2,list_2,list_1)
            
            points_1=self.make_points(x_1,y_new_1)
            points_2=self.make_points(x_2,y_new_2)
            
            points[i][t[0]][pair[0]]=points_1
            points[i][t[1]][pair[1]]=points_2
            
        print(i,points)
        return points
        
        
    def list_x(self,list_points):
        
        x=[]
        
        for i in range(len(list_points)):
            x.append(list_points[i][0])
            
        return x
    
    def inter(self,x,fun1,fun2):
        
        y_new=[]
        for i in range(len(x)):
            x_i=x[i]
            beta_origin=fun1[i][1]
            
            for j in range(len(fun2)-1):
                
                x_j=fun2[j][0]
                x_jnext=fun2[j+1][0]
                
                if x_i>=x_j and x_i<=x_jnext:
                    
                    beta_j=fun2[j][1]
                    beta_jnext=fun2[j+1][1]
                    
                    beta_int=beta_j+(beta_jnext-beta_j)/(x_jnext-x_j)*(x_i-x_j)
                    
                    beta_new=beta_int*beta_origin
                    
                    y_new.append(beta_new)
        
        return y_new
    
    def make_points(self, x,y):
        
        num=len(x)
        
        n_points=[]
        for i in range(num):
            point=[x[i],y[i]]
            
            n_points.append(point)
            
        return n_points
            
       
class multiloss:
    def __init__(self,points):
        self.points=points
        
    def newnew(self):
        points=self.points
        long=len(points)
        
        new_points=[]
        for i in range(long):
            
            if points[i][1]!=[] and points[i][2]!=[]:
                
                list_points_dead=points[i][1][0]
                list_points_skeg=points[i][2][0]
               
                x_dead=self.list_x(list_points_dead)
                x_skeg=self.list_x(list_points_skeg)
                
                y_new_dead=self.inter(x_dead,list_points_dead,list_points_skeg)
                y_new_skeg=self.inter(x_skeg,list_points_skeg,list_points_dead)
                
                dead=self.make_points(x_dead,y_new_dead)
                skeg=self.make_points(x_skeg,y_new_skeg)
            
                points[i][1][0]=dead
                points[i][2][0]=skeg
        
        
        return points
        
    def list_x(self,list_points):
        
        x=[]
        
        for i in range(len(list_points)):
            x.append(list_points[i][0])
            
        return x
    
    def make_points(self, x,y):
        
        num=len(x)
        
        n_points=[]
        for i in range(num):
            point=[x[i],y[i]]
            
            n_points.append(point)
            
        return n_points
        
        
    def inter(self,x,fun1,fun2):
        
        y_new=[]
        for i in range(len(x)):
            x_i=x[i]
            beta_origin=fun1[i][1]
            
            for j in range(len(fun2)-1):
                
                x_j=fun2[j][0]
                x_jnext=fun2[j+1][0]
                
                if x_i>=x_j and x_i<=x_jnext:
                    
                    beta_j=fun2[j][1]
                    beta_jnext=fun2[j+1][1]
                    
                    beta_int=beta_j+(beta_jnext-beta_j)/(x_jnext-x_j)*(x_i-x_j)
                    
                    beta_new=beta_int*beta_origin
                    
                    y_new.append(beta_new)
        
        return y_new
              
        
        
          
        
                    
        
            
        
        
                
   
        
       
            
        
        
                
                
            
        
        
        
    
        
            
        
        
        
        
        
        
    
        
        
        
        
        
        
        
            
            
            
            
        

     
    
    
    
    
        
        
        