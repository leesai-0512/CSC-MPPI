#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from plot.simenv1 import SIMULATIONENV1 as SIM
from solver.jax_mppi import JAXMPPIController
import util.util as util
import time

from matplotlib.animation import FuncAnimation, FFMpegWriter

import jax
import jax.numpy as jnp
from jax import profiler

if __name__ == "__main__":

    fig, ax = plt.subplots(figsize=(15, 15))
    

    print(jax.devices())

    env_name = "env1"
    batch_size = 300
    dt = 0.03
    ref_path = jnp.array([2.0, 2.0, math.pi / 2, 1.0])
    state = jnp.array([-1.0, -1.0, math.pi / 2])
    radius  = 0.1
    obstacles = jnp.array([
        [-1.0, 0.0, 0.3],
        [ 0.0, 1.0, 0.4],
        [ 1.5, 0.7, 0.5],
    ], dtype=jnp.float32)

    obstacles_types = ["circle", "circle", "circle", 
                       #"circle"
                       ]
    tolerance = jnp.array([0.15, 0.25])
    log_dir = util.get_log_dir(env_name, batch_size)
    execute_dbscan = True
    sim = SIM(state=state, radius=radius, dt=dt, ax=ax, fig=fig,
              obstacles=obstacles, obstacles_types=obstacles_types)
    
    mppi = JAXMPPIController(
        delta_t=dt,  # [s]
        max_vx=0.5,  # [m/s]
        max_wz=3.0,  # [rad/s]
        time_step=30,  # timestep
        batch_size=batch_size,  # batch
        sigma=jnp.array([[0.1, 0.0], [0.0, 1.0]]),
        ref_path=ref_path,  
        param_lambda=0.01,  
        param_alpha=0.98,  #
        param_exploration=0.0, 
        stage_cost_weight=jnp.array([10.0, 10.0, 0.0, 0.0]),
        terminal_cost_weight=jnp.array([50.0, 50.0, 50.0, 0.0]),
        obstacles=obstacles,
        obstacles_types=obstacles_types,
        robot_radius = radius,
        rho_nu_max = jnp.array([0.4,0.4]),
        rho_nu_min = jnp.array([0.4,0.4]),
        eta = 0.6,
        eps=10.0,
        min_samples=2,
        visualize_optimal_traj=True,  
        visualize_sampled_trajs=True, 
        execute_dbscan=execute_dbscan,
    )



    frame_count = 0
    total_distance = 0.0  


    previous_state = jnp.copy(state)


    def update(frame):

        global frame_count, total_distance, previous_state
        
        frame_count += 1 

        mppi.set_state(sim.state)
        mppi.check_collision()
        start = time.perf_counter()
        optimal_input, optimal_input_sequence,sampled_trajectory = mppi.compute_control_input()
        total_elapsed_time = time.perf_counter() - start

        optimal_traj, opt_clustered_traj,clustered_trajs = mppi.compute_plot_data(sampled_trajectory)
        
        mppi.move_obstacle()
        mppi.set_zero()
        new_state = util.compute_next_state(sim.state, optimal_input,dt)
        total_distance += util.compute_distance(new_state,previous_state)
        
        
        previous_state = new_state.clone()
        sim.state = new_state
    
        sim.update_plot(optimal_traj=optimal_traj, predicted_traj=sampled_trajectory,opt_clustered_traj = None,
                        best_trajectory = None,
                        clustered_trajs=dict((str(label), traj) for label, traj in clustered_trajs))
        

        util.opt_traj_txt_data(frame,optimal_traj,log_dir)
        if execute_dbscan:
            util.clustered_trajs_txt_data(frame,clustered_trajs,log_dir)
        else:
            util.sampled_trajs_txt_data(frame,sampled_trajectory,log_dir)
        util.constraint_step_txt_data(mppi.step,log_dir)
        # if(mppi.step > 990):
        #     print("optimal_input_sequence: ", optimal_input_sequence)

        if frame_count != 1:

            util.total_time_txt_data(total_elapsed_time,log_dir)
            util.dbscan_time_txt_data(mppi.dbscan_elapsed_time,log_dir)

            if total_elapsed_time > 0.1:
                ani.event_source.stop() 
                print("Time over.")
                exit(0)

        if util.goal_check(new_state,ref_path,tolerance):
            print("goal arrive")
            util.frame_count_txt_data(frame_count,log_dir)
            util.path_length_txt_data(total_distance,log_dir)
            plt.close(fig) 

        
        
    ani = FuncAnimation(fig, update, frames=None, interval=100, repeat=False, cache_frame_data=False)

    # 그래프 표시
    plt.show()