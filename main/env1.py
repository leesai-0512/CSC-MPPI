#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from plot.simenv1 import SIMULATIONENV1 as SIM
from solver.torch_mppi import MPPIController
import util.util as util
import time
from matplotlib.animation import FuncAnimation



if __name__ == "__main__":

    fig, ax = plt.subplots(figsize=(15, 15))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    env_name = "env1"
    batch_size = 300
    dt = 0.03
    ref_path = torch.tensor([2.0, 2.0, math.pi / 2, 1.0], device=device)
    state = torch.tensor([-1.0, -1.0, math.pi / 2], device=device)
    radius  = 0.1
    obstacles = torch.tensor([
        [-1.0, 0.0, 0.3],  # x=1.0, y=1.0, r=0.5
        [0.0, 1.0, 0.4], # x=2.5, y=-1.0, r=0.3
        [1.5, 0.7, 0.5],  # x=-1.0, y=2.0, r=0.8
        #[-0.7, 0.5, 0.4]  
    ], dtype=torch.float32)
    obstacles_types = ["circle", "circle", "circle", 
                       #"circle"
                       ]
    tolerance = torch.tensor([0.15,0.25])
    # log_dir = util.get_log_dir(env_name, batch_size)
    execute_dbscan = True
    sim = SIM(state=state, radius=radius,dt=dt, device=device, ax=ax,fig=fig ,obstacles = obstacles ,obstacles_types=obstacles_types)
    mppi = MPPIController(
        delta_t=dt,  # [s]
        max_vx=0.5,  # [m/s]
        max_wz=3.0,  # [rad/s]
        time_step=30,  # timestep
        batch_size=batch_size,  # sample size
        sigma=torch.tensor([[0.1, 0.0], [0.0, 1.0]], dtype=torch.float32),  
        ref_path=ref_path, 
        param_lambda=0.01, 
        param_alpha=0.98, 
        param_exploration=0.0, 
        stage_cost_weight=torch.tensor([10.0, 10.0, 0.0, 0.0], dtype=torch.float32), 
        terminal_cost_weight=torch.tensor([50.0, 50.0, 50.0, 0.0], dtype=torch.float32),
        obstacles=obstacles,
        obstacles_types=obstacles_types,
        robot_radius = radius,
        rho_nu_max = torch.tensor([0.4,0.4], dtype=torch.float32),
        rho_nu_min = torch.tensor([0.4,0.4], dtype=torch.float32),
        eta = 0.6,
        eps=10.0,
        min_samples=2,
        visualize_optimal_traj=True,
        visualize_sampled_trajs=True, 
        execute_dbscan=execute_dbscan,
        device=device
    )

    frame_count = 0
    total_distance = 0.0 

    previous_state = state.clone()



    def update(frame):

        global frame_count, total_distance, previous_state
        frame_count += 1

        mppi.set_state(sim.state)
        mppi.check_collision()
        mppi.set_obstacle()
        start = time.perf_counter()
        optimal_input, optimal_input_sequence,sampled_trajectory = mppi.compute_control_input()
        total_elapsed_time = time.perf_counter() - start

        optimal_traj, opt_clustered_traj,clustered_trajs = mppi.compute_plot_data()
        
        mppi.move_obstacle()
        mppi.set_zero()
        new_state = util.compute_next_state(sim.state, optimal_input,dt)
        total_distance += util.compute_distance(new_state,previous_state)
        
        previous_state = new_state.clone()
        sim.state = new_state
    
        sim.update_plot(optimal_traj=optimal_traj, predicted_traj=sampled_trajectory,opt_clustered_traj = None,
                        clustered_trajs=dict((str(label), traj) for label, traj in clustered_trajs))
        
        if frame_count != 1:
            if total_elapsed_time > 0.1:
                 raise RuntimeError("time over")
            # util.total_time_txt_data(total_elapsed_time,log_dir)
            # util.dbscan_time_txt_data(mppi.dbscan_elapsed_time,log_dir)

        # util.opt_traj_txt_data(frame,optimal_traj,log_dir)
        # if execute_dbscan:
        #     util.clustered_trajs_txt_data(frame,clustered_trajs,log_dir)
        # else:
        #     util.sampled_trajs_txt_data(frame,sampled_trajectory,log_dir)
        
        if util.goal_check(new_state,ref_path,tolerance):
            print("goal arrive")
            # util.frame_count_txt_data(frame_count,log_dir)
            # util.path_length_txt_data(total_distance,log_dir)
            plt.close(fig)  

    ani = FuncAnimation(fig, update, frames=None, interval=100, repeat=False, cache_frame_data=False)
    plt.show()