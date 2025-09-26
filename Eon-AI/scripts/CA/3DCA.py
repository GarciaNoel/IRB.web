import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# "pip install PyQt6", if not working!


def count_neighbors(grid, x, y, z):
    neighbors = [
        (i, j, k)
        for i in range(-1, 2)
        for j in range(-1, 2)
        for k in range(-1, 2)
        if not (i == j == k == 0)
    ]
    count = 0
    for dx, dy, dz in neighbors:
        if (
            0 <= x + dx < grid.shape[0]
            and 0 <= y + dy < grid.shape[1]
            and 0 <= z + dz < grid.shape[2]
        ):
            count += grid[x + dx, y + dy, z + dz]
    return count


def update_grid(grid):
    new_grid = np.zeros_like(grid)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                neighbors = count_neighbors(grid, x, y, z)
                if grid[x, y, z] == 1:
                    new_grid[x, y, z] = 1 if neighbors in (2, 3) else 0
                else:
                    new_grid[x, y, z] = 1 if neighbors == 3 else 0
    return new_grid


def plot_grid(grid, ax):
    ax.clear()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_box_aspect(
        [grid.shape[0], grid.shape[1], grid.shape[2]]
    )  # Aspect ratio is 1:1:1

    x, y, z = np.indices(np.array(grid.shape) + 2)
    filled = grid.astype(bool)

    ax.voxels(
        x[:-1, :-1, :-1],
        y[:-1, :-1, :-1],
        z[:-1, :-1, :-1],
        filled,
        edgecolor="k",
        facecolors="cyan",
        alpha=0.5,
    )


def animate(i, grid, ax):
    plot_grid(grid, ax)
    if i in range(1):
        return
    new_grid = update_grid(grid)
    np.copyto(grid, new_grid)  # Copy new_grid into grid


# Initialize the 3D grid (size 10x10x10)
# grid = np.random.choice([0, 1], size=(20, 20, 20))

def init_grid():
    grid = np.zeros(shape=(30, 30, 30))
    x = 15
    h = 20 
    l = 5
    inclen = 12
    grid[x:x+inclen, l, h] = [0,0,0,0,1,0,0,1,0,0,0,0]
    grid[x:x+inclen, l+1, h] = [1,0,1,0,1,1,1,1,0,1,0,1]
    grid[x:x+inclen, l+2, h] = [0,1,0,0,1,0,0,1,0,0,1,0]
    grid[x:x+inclen, l+3, h] = [1,0,1,0,1,1,1,1,0,1,0,1]
    grid[x:x+inclen, l+4, h] = [0,0,0,0,1,0,0,1,0,0,0,0]
    
    return grid

# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

iterations = 10000

# Create the animation
ani = FuncAnimation(fig, animate, fargs=(init_grid(), ax), frames=iterations, interval=7000)

plt.show()
