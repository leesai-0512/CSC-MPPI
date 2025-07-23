import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util.util as util


def plot_full_trajectory():


    robot_radius = 0.1

    obstacles = np.array([
       [0.0, 0.0, 0.5],  # x=-1.0, y=0.0, radius=0.3
       [0.0, 0.0, 0.5],  # x=-1.0, y=0.0, radius=0.3

    ])
    env_name = "env2"
    batch_size = 300
    log_dir = util.get_log_dir(env_name, batch_size)
    
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

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    ax.grid(True, linewidth=5, color='black', alpha=0.5)

    sampled_trajs = []
    with open(os.path.join(log_dir, "sampled_traj_log.txt"), "r") as f_samp:
        for line in f_samp:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line) 
                sampled_traj = np.array(data["Sampled_Trajectory"]) 
                sampled_trajs.append(sampled_traj)
            except json.JSONDecodeError:
                print(f"JSON Decode Error: {line}")
                continue

    for traj_set in sampled_trajs: 
        for traj in traj_set: 
            ax.plot(traj[:, 0], traj[:, 1],linewidth=12.0 ,linestyle='-', color='red', alpha=0.3)

    optimal_points = []
    with open("optimal_traj_log.txt", "r") as f_opt:
        current_traj = []
        for line in f_opt:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Frame") or "TimeStep" in line:
                if current_traj:
                    for point in current_traj:
                        if len(point) == 3:
                            optimal_points.append(np.array(point))
                    current_traj = []
            else:
                try:
                    current_traj.append([float(x) for x in line.split(",")])
                except ValueError:
                    continue


    optimal_points = np.array(optimal_points)
    if len(optimal_points) > 0:
        ax.plot(optimal_points[:, 0], optimal_points[:, 1], color='cyan', linestyle='-', linewidth=8.0, label="Robot Path")


    # num_obstacle_steps = 2  
    # obstacle_x_positions = np.linspace(obstacles[0, 0], 0.5, num_obstacle_steps)


    # alphas = np.linspace(0.2, 0.7, num_obstacle_steps) 
    # for i, x_pos in enumerate(obstacle_x_positions):
    #     circle = Circle((x_pos, obstacles[0, 1]), obstacles[0, 2], color='gray', alpha=alphas[i])
    #     ax.add_patch(circle)

    #     safe_zone = Circle((x_pos, obstacles[0, 1]), obstacles[0, 2] + robot_radius, 
    #                        edgecolor='gray', linestyle='dashed', facecolor='none', linewidth=12.0, alpha=alphas[i])
    #     ax.add_patch(safe_zone)

    for obs in obstacles[1:]:
        obs_x, obs_y, obs_r = obs
        circle = Circle((obs_x, obs_y), obs_r, color='gray', alpha=0.5)
        ax.add_patch(circle)

        safe_zone = Circle((obs_x, obs_y), obs_r + robot_radius, 
                           edgecolor='gray', linestyle='dashed', facecolor='none', linewidth=12.0, alpha=0.7)
        ax.add_patch(safe_zone)

    plt.show()

plot_full_trajectory()
