
import numpy as np
import matplotlib.pyplot as plt


colors=['lightgreen']
w=[1]
m=['-']
names_leg=['Zone']
points=[[0, 1], [69, 1], [79, 0.836], [90, 0.342], [101, 0.447], \
        [113, 0.251], [180, 1], [271,1], [360, 1]]
    
angle=[]
dp=[]
for i,point in enumerate(points):
    angle.append(point[0])
    dp.append(point[1])

angle_rad=np.radians(angle)



angle[8]=0
i=0
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
ax.plot(angle_rad, dp, color=colors[i], linewidth=w[i], linestyle=m[i], label=names_leg[i])
ax.set_rmax(1.1)
ax.set_rticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])  
ax.set_thetagrids(angle)
ax.set_rlabel_position(337.5)  
ax.grid(True)
ax.set_theta_zero_location("N")  # theta=0 at the top
ax.set_theta_direction(1)  # theta increasing clockwise
ax.set_title("Azimuth thruster zones", va='bottom',fontsize=25, color='royalblue')