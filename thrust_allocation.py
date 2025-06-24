import numpy as np



class TA_module:
    def __init__(self,allocation_df):
        self.allocation_df = allocation_df       
        
    def table_data(self):
        # 1. Find the columns of the data of interest
        # 2. Find the values in the columns
        
        allocation_df = self.allocation_df
        
        # 1 -------------
        
        header=allocation_df.columns.values
        
        Fx_col=[]
        Fy_col=[]
        Mz_col=[]
        sol_=[]
        
        for count, item in enumerate(header):
            item_first=item.split('_')[0]
            
            if item_first=='Fx':
                Fx_col.append(count)
            elif item_first=='Fy':
                Fy_col.append(count)
            elif item_first=='Mz':
                Mz_col.append(count)
            elif item_first=='solution':
                sol_.append(count)
        
        # 2 -----------------
        
        val=allocation_df.values.transpose()
        
        Tx = np.vstack((val[Fx_col[0]],val[Fx_col[1]],val[Fx_col[2]]))*(-1.25)
        Ty = np.vstack((val[Fy_col[0]],val[Fy_col[1]],val[Fy_col[2]]))*(-1.25)
        Nz = np.vstack((val[Mz_col[0]],val[Mz_col[1]],val[Mz_col[2]]))*(-1.25)
        
        tau_x = Tx.sum(axis=0)
        tau_y = Ty.sum(axis=0)
        tau_n = Nz.sum(axis=0)
        
        tau = np.vstack((tau_x,tau_y,tau_n))
        
        tau = tau.transpose()
        
        sol = np.zeros(len(val[0]))
        
        for i in range(len(sol_)):
            sol = np.vstack((sol,val[sol_[i]]))
            
        sol = sol[1:]
        
        sol = sol.transpose()
        
        
        return tau, sol
    
    
    def find_thrust_allocation(self,tau_control):
        
        tau, sol = self.table_data()
        
        # 1. Create the numbering array of each element
        # 2. Compare each tau vector from the table to the tau_control vector
        # 3. Select candidates that are closest to the tau_control direction
        # 4. Pick the vector magnitude that is the closest to the tau_control 
        
        # 1 ----------
        indexes = np.arange(0,len(tau),1)
        
        dang = []
        dmag = []
        
        for i, t in enumerate(tau):
            
            # 2 ----------
            delta_angle, delta_mag = self.vectors_match(t,tau_control)
            
            dang.append(delta_angle)
            dmag.append(delta_mag)
        
        indexes_sorted = [x for _,x in sorted(zip(dang,indexes))]
        
        candidates = 5
        dmag_ang = []
        for i in range(candidates):
            # 3 ------------
            dmag_ang.append(dmag[indexes_sorted[i]])
        
        # 4 ----------
        ind = dmag_ang.index(min(dmag_ang))
        
        index_solution = indexes_sorted[ind]
        
        ta = sol[index_solution]
        
        return ta
    
    def magnitude(self, vector):
        
        return (vector[0]**2+vector[1]**2+ vector[2]**2)**0.5
        
    def vectors_match(self,vector_table, vector_control):
        
        mag_table=self.magnitude(vector_table)
        mag_control=self.magnitude(vector_control)
        
        numerator = np.dot(vector_table, vector_control)
        dominator = mag_table * mag_control

        delta_angle = np.degrees(np.arccos(numerator/dominator+0.000000001))
        
        delta_mag = np.abs(mag_table-mag_control)
        
        return delta_angle, delta_mag
    

        
        
        
