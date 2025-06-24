"""
This is a loop throught all thrusters to get the thrusters data
in aa list
"""

class prepare_matrix_thr:
    def __init__(self,thrusters):
        self.thrusters=thrusters
        
    def thr_data_m(self):
        
        thrusters=self.thrusters
        
        num_thr=len(thrusters)
        
        thr_data=[]
        name=[]
        for i in range(num_thr):
            thr_data.append([])
            
            type_thruster=thrusters[i][1]
            name.append(thrusters[i][0])
            
            choices1={
                'Shaft line FPP without nozzle':'propeller FPP',
                'Shaft line FPP with nozzle':'propeller FPP nozzle',
                'Shaft line CPP without nozzle':'propeller CPP',
                'Shaft line CPP with nozzle':'propeller CPP nozzle',
                'Shaft line CRP':'propeller CRP',
                'Azimuth without nozzle':'azi',
                'Azimuth with nozzle':'azi nozzle',
                'Azimuth CRP':'azi CRP',
                'pod without nozzle':'pod',
                'pod with nozzle':'pod nozzle',
                'pod CRP':'pod CRP',
                'Tunnel thruster':'tunnel',
                'Cycloidal':'cyclo',   
                }
          
            type_thr=str(choices1.get(type_thruster, 'not in the base'))
            
            
            inlet=thrusters[i][2]
            
            choices2={
                'Only forward mode *except tunnel thruster':'forward',
                'Reverse possible - FFP without nozzle':'reverse FPP',
                'Reverse possible - FFP with nozzle':'reverse FPP nozzle',
                'Reverse possible - CPP without nozzle':'reverse CPP',
                'Reverse possible - CPP with nozzle':'reverse CPP nozzle',
                'If tunnel thruster  - option 1 - broken inlet':'broken',
                'If tunnel thruster - option 2 - rounded inlet':'raunded',
                'If tunnel thruster  - option 3 - other':'other',
                }
            
            
            inlet_mode=str(choices2.get(inlet, 'not in the base'))
            
            
            type_of_rudder=thrusters[i][3]
            
            choices3={
                'No rudder':'none',
                'naca':'naca',
                'hollow':'hollow',
                'flat':'flat',
                'fish_tail':'fish_tail',
                'flap':'flap',
                'nozzle':'nozzle',
                'mix':'mix',
                }
            
            rudder_type=str(choices3.get(type_of_rudder, 'not in the base'))
            
            x_thr=float(thrusters[i][4].replace(',','.'))
            y_thr=float(thrusters[i][5].replace(',','.'))
            z_thr=float(thrusters[i][6].replace(',','.'))
            D=float(thrusters[i][7].replace(',','.'))
            P_max=float(thrusters[i][8].replace(',','.'))
            
            
            mech_e=thrusters[i][9]
            
            choices4={
                'Cycloidal':'cyclo',
                'Permanent magnet cycloidal':'magnet',
                'tunnel or azimuth':'azi tunnel',
                'Rim-driven permanent magnet':'rim magnet',
                'shaft line':'propller',
                'pod':'pod',
                }
            
            eta_mech=str(choices4.get(mech_e,'not in the base'))
            
            s_rudder=float(thrusters[i][10].replace(',','.'))
            
            
            thr_list=[type_thr,inlet_mode,rudder_type,x_thr,y_thr,z_thr,D,P_max,eta_mech,s_rudder]
            thr_data[i]=thr_list
            
        
        return thr_data, name
                


