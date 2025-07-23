import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

def plot_animated_trajectory():


    robot_radius = 0.1


    obstacles = np.array([
        [-1.0-0.016, 0.0, 0.3],  # x=-1.0, y=0.0, radius=0.3 
        [0.0, 1.0, 0.4],   # x=0.0, y=1.0, radius=0.4 
        [1.5, 0.7, 0.5],   # x=1.5, y=0.7, radius=0.5 
    ])


    num_frames = 100  
    dx = 0.016  


    fig, ax = plt.subplots(figsize=(15, 15))
    fig.subplots_adjust(bottom=0.15, left=0.12)

    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1.5, 2.5)
    ax.set_aspect('equal')

    ax.tick_params(axis='both', which='major', labelsize=80, width=5, length=25, direction='out')
    ax.tick_params(axis='both', which='minor', labelsize=80, width=5, length=20, direction='out')

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_xlabel("X [m]", fontsize=80, fontweight='bold', labelpad=20)
    ax.set_ylabel("Y [m]", fontsize=80, fontweight='bold', labelpad=5)

    for spine in ax.spines.values():
        spine.set_linewidth(5)
        spine.set_color("black")

    ax.grid(True, linewidth=5, color='black', alpha=0.5)


    moving_circle = Circle((obstacles[0, 0], obstacles[0, 1]), obstacles[0, 2], color='gray', alpha=0.7)
    ax.add_patch(moving_circle)

    moving_safe_zone = Circle((obstacles[0, 0], obstacles[0, 1]), obstacles[0, 2] + robot_radius,
                              edgecolor='gray', linestyle='dashed', facecolor='none', linewidth=10.0, alpha=0.7)
    ax.add_patch(moving_safe_zone)


    for obs in obstacles[1:]:
        obs_x, obs_y, obs_r = obs
        circle = Circle((obs_x, obs_y), obs_r, color='gray', alpha=0.5)
        ax.add_patch(circle)

        safe_zone = Circle((obs_x, obs_y), obs_r + robot_radius,
                           edgecolor='gray', linestyle='dashed', facecolor='none', linewidth=10.0, alpha=0.7)
        ax.add_patch(safe_zone)

    yellow_circle = Circle((2, 2), 0.15, facecolor='purple', edgecolor='purple', linewidth=2.0)
    ax.add_patch(yellow_circle)

    black_circle = Circle((-1, -1), 0.15, color='black')
    ax.add_patch(black_circle)

    def update(frame):
        new_x = obstacles[0, 0] + dx * frame
        moving_circle.set_center((new_x, obstacles[0, 1]))
        moving_safe_zone.set_center((new_x, obstacles[0, 1]))
        if frame == num_frames - 1:
            plt.close(fig)  
        return moving_circle, moving_safe_zone

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

    plt.show()

plot_animated_trajectory()
