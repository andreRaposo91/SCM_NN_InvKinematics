import numpy as np
import matplotlib.pyplot as plt

from kinematics_functions import T_beModule
from ISTlogo_traj import ISTlogo_traj

jt2l = lambda p: ((p-316.1)/-2.468)

[traj ,traj_] = ISTlogo_traj()
print(len(traj))

# Enable interactive mode
plt.ion()
global ax

# Create an empty plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot([], [], [], '-o')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')
ax.grid(True)
ax.set_aspect('equal')
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-25, 200)
ax.set_autoscale_on(False)

for i in range(len(traj_)):
    # print(T_beModule([traj_[i,0], traj_[i,1], traj_[i,2]], [], 1, ax), '\n')
    # T_beModule([traj_[i,0], traj_[i,1], traj_[i,2]], [], 1, ax)
    T_beModule(jt2l(np.array([15,15,15])), T_beModule([traj_[i,0], traj_[i,1], traj_[i,2]],[], 1, ax), 1, ax)
    # ax.cla()

plt.ioff()
plt.show()
