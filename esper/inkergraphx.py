import numpy as np
import matplotlib
matplotlib.use('TkAgg') # or 'QtAgg'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def torus(R, r, u, v):
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return x, y, z

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Parameters for the toruses
R = 30   # Major radius
r = 18   # Minor radius
theta = np.linspace(0, 2*np.pi, 30)
phi = np.linspace(0, 2*np.pi, 30)
theta, phi = np.meshgrid(theta, phi)

# Arrange six toruses in a circular formation
angles = np.linspace(0, 2*np.pi, 6, endpoint=False)

for angle in angles:
    x, y, z = torus(R, r, theta, phi)
    x_offset = 6 * np.cos(angle)
    y_offset = 6 * np.sin(angle)
    ax.plot_surface(x + x_offset, y + y_offset, z, color='blue', edgecolor='k')

ax.set_box_aspect([1, 1, 1])
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def torus(R, r, u, v):
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return x, y, z

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

R = 3  # Major radius
r = 1  # Minor radius
theta = np.linspace(0, 2*np.pi, 100)
phi = np.array([0, np.pi])  # Select only inner and outer rings

theta, phi = np.meshgrid(theta, phi)

angles = np.linspace(0, 2*np.pi, 6, endpoint=False)

for angle in angles:
    x, y, z = torus(R, r, theta, phi)
    x_offset = 6 * np.cos(angle)
    y_offset = 6 * np.sin(angle)
    ax.plot_wireframe(x + x_offset, y + y_offset, z, color='blue')

ax.set_box_aspect([1, 1, 1])
plt.show()
