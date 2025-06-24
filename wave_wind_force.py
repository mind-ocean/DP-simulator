import numpy as np
import matplotlib.pyplot as plt

def spectra(omega,Hs,Tp,gamma):
    
    omegap=2*np.pi/Tp
    Agamma=1-0.287*np.log(gamma)
    
    if omega<=omegap:
        sigma=0.07
    else:
        sigma=0.09
        
    Spm=5/16*Hs**2*omegap**4*omega**(-5)*np.exp(-5/4*(omega/omegap)**(-4))
    
    
    n=np.exp(-0.5*((omega-omegap)/(sigma*omegap))**2)
    S=Agamma*Spm*gamma**n
    
    return S

def drift_force(x,y,angle,sea_state):
    
    Hs = sea_state[0]
    Tp = sea_state[1]
    gamma = sea_state[2]
    
    F=0
    
    for i in range(len(x)-1):
        F=F+(spectra(x[i],Hs,Tp,gamma)*y[i]*(x[i+1]-x[i]))
    
    F=F+(spectra(x[len(x)-1],Hs,Tp,gamma)*y[len(x)-1]*(x[len(x)-1]-x[len(x)-2]))
        
    F=2*F
    
    return F

def wave_matrix(sea_state,wave_coeff):
    
    angles = np.arange(0,370,10)
    
    if sea_state[4] != 0:
        drift_table = np.zeros((int(len(wave_coeff)),int(len(angles))))
        
        for i, wave in enumerate(wave_coeff):
            
            for j, a in enumerate(angles):
                
                x = wave[0]
                y = wave[j]
                
                drift_table[i,j] = drift_force(x,y,a,sea_state)
                
        drift_table = np.vstack((angles,drift_table))
            
        fig,ax = plt.subplots(3,1,figsize=(6,6))
        
        fig.suptitle('Waves Hs = '+str(sea_state[0])+' m, Tp = '+str(sea_state[1]))
        
        for i in range(3):
            ax[i].plot(drift_table[0],drift_table[i+1],c='teal',alpha=0.4)
            ax[i].set_xlabel('Environmental angle')
            if i<2:
                ax[i].set_ylabel('Force [N]')
            else:
                ax[i].set_ylabel('Moment [Nm]')
                
        fig.savefig('out/Wave_all_angles.png',dpi=200) 
        plt.close()
    
    else:
        drift_table = 0
    
    return drift_table

def wind_matrix(wind_data,wind):
    
    if wind_data[4] != 0:
        wind_table = np.zeros((len(wind)-1,len(wind[0])))
        ro_air=1.2
        
        for j, a in enumerate(wind[0]):
            
            wind_table[0][j]=0.5*ro_air*wind_data[4]**2*wind_data[0]*wind[1][j]
            wind_table[1][j]=0.5*ro_air*wind_data[4]**2*wind_data[1]*wind[2][j]
            wind_table[2][j]=0.5*ro_air*wind_data[4]**2*wind_data[2]*wind_data[1]*wind[3][j]
        
        wind_table = np.vstack((wind[0],wind_table))
        
        fig,ax = plt.subplots(3,1,figsize=(6,6))
        
        fig.suptitle('Wind')
        
        for i in range(3):
            ax[i].plot(wind_table[0],wind_table[i+1],c='deepskyblue',alpha=0.4)
            ax[i].set_xlabel('Environmental angle')
            if i<2:
                ax[i].set_ylabel('Force [N]')
            else:
                ax[i].set_ylabel('Moment [Nm]')
                
        #fig.savefig('out/Wind_all_angles.png',dpi=200)
        plt.close()
    
    else:
        
        wind_table = 0
        
    return wind_table
        
        
