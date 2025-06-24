import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from functions import *
import random
from scipy.signal import welch

class aqwa_read:
    def __init__(self,df):
        self.df = df
        self.freq_data_start = '    * * * * W A V E   F R E Q U E N C I E S / P E R I O D S   A N D   D I R E C T I O N S * * * * '
        self.freq_data_end = '          ---------------------------------------------------------------------------------------------'
        self.aqwa = ' *********1*********2*********3*********4*********5*********6*********7*********8'
        self.dirr = 'DIRECTIONS'
        self.Ist_order = '                         FROUDE KRYLOV + DIFFRACTION FORCES-VARIATION WITH WAVE PERIOD/FREQUENCY'
        self.diffraction_stop = '              * * * * H Y D R O D Y N A M I C   P A R A M E T E R S   F O R   S T R U C T U R E   1 * * * *'
        self.IInd_order = '            * * * * W A V E - D R I F T   L O A D S   F O R   U N I T   W A V E   A M P L I T U D E * * 2   * * * * '
        self.Ist_order_indices = {
            'X' : [2,3],
            'Y' : [4,5],
            'RZ' :[12,13]
            }
        
    def read_basic_data(self):
        
        df = self.df
        
        data = list(df[self.aqwa])
        
        omegas, num_freq = self.fequencies(data)
        
        A1, P1, angles = self.table_1st_order(data,'X',num_freq)
        A2, P2, angles = self.table_1st_order(data,'Y',num_freq)
        A3, P3, angles = self.table_1st_order(data,'RZ',num_freq)
       
        BETA = self.angles_flip(angles)
       
        Ist_wave_dict = {
            'Omega' : omegas,
            'Angles' : BETA,
            'X' : [A1, P1],
            'Y' : [A2, P2],
            'RZ' : [A3, P3]
            }
        
        rao1 = self.table_2nd_order(data,'   SURGE(X) ',num_freq)
        rao2 = self.table_2nd_order(data,'   SWAY(Y)  ',num_freq)
        rao3 = self.table_2nd_order(data,'   YAW(RZ)  ',num_freq)
        
        IInd_wave_dict = {
            'Omega' : omegas,
            'Angles' : BETA,
            'X' : rao1,
            'Y' : rao2,
            'RZ' : rao3
            }
        
        #X = pd.DataFrame(rao1)
        #column_names = np.array(angles, dtype = str)
        #X.to_csv('aga.csv', sep = ';', header = column_names, index = False)
        
        return Ist_wave_dict, IInd_wave_dict
    
        
    def fequencies(self,data):
        
        index = data.index(self.freq_data_start)
        i = index + 6
        omegas = []
        while data[i] != self.freq_data_end:
            line = data[i]
            line = line.split(' ')
            
            while('' in line):
                line.remove('')
            
            if i == index + 6:
                omega = line[2]
            else:
                omega = line[1]
            
            omegas.append(omega)
            
            i += 1
        omegas = np.array(omegas)
        
        i += 1
        line = data[i]
        line = line.split(' ')
        
        while('' in line):
            line.remove('')
        
        if line [0] == self.dirr:
            
            count = 0
            while data[i] != self.freq_data_end :
                count += 1
                i += 1
       
        num_freq = len(omegas)
        
        return omegas, num_freq
    
    def table_1st_order(self,data,axis,num_freq):
        
        ind = list(self.Ist_order_indices.keys()).index(axis)
        position = list(self.Ist_order_indices.values())[ind]
        
        angles = []
        amplitudes = []
        phases = []
        
       
        k = -1
        for i in range(len(data)):
            
            if data[i] == self.Ist_order:
                line = data[i+6]
                line = line.split(' ')
               
                while('' in line):
                    line.remove('')
                
                angles.append(line[2])
                amplitudes.append([])
                phases.append([])
                
                k += 1
                AMP = []
                PHA = []
                for j in range(num_freq):
                    line = data[i+6+j].split(' ')
                    
                    while('' in line):
                        line.remove('')
                     
                    a = position[0]
                    p = position[1]
                    
                    if j == 0:
                        AMP.append(line[a+1])
                        PHA.append(line[p+1])
                    else:
                        AMP.append(line[a])
                        PHA.append(line[p])
                
                amplitudes[k] = AMP
                phases[k] = PHA
                
                
                if data[i + 6 + num_freq + 1] != self.diffraction_stop:
                    
                    amplitudes.append([])
                    phases.append([])
                    k += 1
                    
                    line = data[i + 6 + num_freq]
                    line = line.split(' ')
                    
                    while('' in line):
                        line.remove('')
                   
                    angles.append(line[2])
                    
                    AMP = []
                    PHA = []
                    for j in range(num_freq):
                        line = data[i+6+num_freq+j].split(' ')
                       
                            
                        while('' in line):
                            line.remove('')
                            
                        a = position[0]
                        p = position[1]
                        
                        if j == 0:
                            AMP.append(line[a+1])
                            PHA.append(line[p+1])
                        else:
                            AMP.append(line[a])
                            PHA.append(line[p])
                    
                    amplitudes[k] = AMP
                    phases[k] = PHA
                    
                    
        A = np.array(np.vstack((amplitudes)).T,dtype='float64')
        P = np.array(np.vstack((phases)).T,dtype='float64')
       
        return A, P, angles
                    
    def angles_flip(self,angles):
        
        new_angles = []
        for i, a in enumerate(angles):
            a_new = 180 - float(a)
            
            if a_new<0:
                a_new = 360 + a_new
            
            new_angles.append(a_new)
       
        return new_angles
        
    def table_2nd_order(self,data,axis,num_freq):
        
        rao = []
        k = -1
        for i in range(len(data)):
            
            if data[i] == self.IInd_order and data[i+10] == axis: 
                k += 1
                rao.append([])
                IInd_rao = []
                for j in range(num_freq):
                    line = data[i+11+j].split(' ')
                    
                    while ('' in line):
                        line.remove('')
                        
                    IInd_rao.append([])
                    IInd_rao[j] = np.array(line, dtype='float64')
                    
                IInd_rao = np.vstack((IInd_rao)).T[1:].T
                
                rao[k] = IInd_rao
               
        if len(rao)>1:
            rao = np.hstack((rao))
        
        return rao


