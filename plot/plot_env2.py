import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_full_trajectory():


    robot_radius = 0.1

    obstacles = np.array([
       [0.0, 0.0, 0.5],  # x=-1.0, y=0.0, radius=0.3
       [0.0, 0.0, 0.5],  # x=-1.0, y=0.0, radius=0.3

    ])

    fig, ax = plt.subplots(figsize=(15, 15))
    fig.subplots_adjust(bottom=0.15, left=0.12) 

    ax.tick_params(axis='both', which='major', labelsize=80, width=5, length=25, direction='out')
    ax.tick_params(axis='both', which='minor', labelsize=80, width=5, length=20, direction='out')
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    

    ax.set_xlabel("X [m]", fontsize=80, fontweight='bold', labelpad=20) 
    ax.set_ylabel("Y [m]", fontsize=80, fontweight='bold', labelpad=5) 

    for spine in ax.spines.values():
        spine.set_linewidth(5)  
        spine.set_color("black") 

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal')

    ax.grid(True, linewidth=5, color='black', alpha=0.5)



    # num_obstacle_steps = 7 
    # obstacle_x_positions = np.linspace(obstacles[0, 0], 0.5, num_obstacle_steps)


    # alphas = np.linspace(0.1, 0.7, num_obstacle_steps)  
    # for i, x_pos in enumerate(obstacle_x_positions):
    #     circle = Circle((x_pos, obstacles[0, 1]), obstacles[0, 2], color='gray', alpha=alphas[i])
    #     ax.add_patch(circle)

    #     safe_zone = Circle((x_pos, obstacles[0, 1]), obstacles[0, 2] + robot_radius, 
    #                        edgecolor='gray', linestyle='dashed', facecolor='none', linewidth=10.0, alpha=alphas[i])
    #     ax.add_patch(safe_zone)

    for obs in obstacles[1:]:
        obs_x, obs_y, obs_r = obs
        circle = Circle((obs_x, obs_y), obs_r, color='gray', alpha=0.5)
        ax.add_patch(circle)


        safe_zone = Circle((obs_x, obs_y), obs_r + robot_radius, 
                           edgecolor='gray', linestyle='dashed', facecolor='none', linewidth=10.0, alpha=0.7)
        ax.add_patch(safe_zone)


    yellow_circle = Circle((1, 0), 0.15, color='purple', edgecolor='black', linewidth=2)
    ax.add_patch(yellow_circle)

    black_circle = Circle((-1, 0), 0.15, color='black')
    ax.add_patch(black_circle)

    plt.show()

plot_full_trajectory()
