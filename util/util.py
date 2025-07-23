import numpy as np
import json
import os
import jax.numpy as jnp

def get_log_dir(env_name: str, sample_num: int) -> str:
    base_dir = "data"
    folder_name = f"{env_name}_K={sample_num}"
    full_path = os.path.join(base_dir, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path



def opt_traj_txt_data(frame: int, optimal_traj: np.ndarray, log_dir: str):
    file_path = os.path.join(log_dir, "optimal_traj_log.txt")
    with open(file_path, "a") as f_opt:
        np.savetxt(f_opt, np.array(optimal_traj), delimiter=",", fmt="%.6f",
                   header=f"Frame {frame}", comments="")



def clustered_trajs_txt_data(frame: int, clustered_trajs, log_dir: str):
    file_path = os.path.join(log_dir, "clustered_traj_log.txt")
    with open(file_path, "a") as f_clu:
        clustered_traj_dict = {
            str(label): np.array(traj).tolist()
            for label, traj in clustered_trajs
        } if clustered_trajs else {}

        json.dump({
            "TimeStep": frame,
            "Clustered_Trajectory": clustered_traj_dict
        }, f_clu)
        f_clu.write("\n")


def sampled_trajs_txt_data(frame: int, sampled_trajs: np.ndarray, log_dir: str):
    file_path = os.path.join(log_dir, "sampled_traj_log.txt")
    with open(file_path, "a") as f_samp:
        sampled_array = np.array(sampled_trajs).tolist()
        json.dump({
            "TimeStep": frame,
            "Sampled_Trajectory": sampled_array
        }, f_samp)
        f_samp.write("\n")



def frame_count_txt_data(frame_count, log_dir):
    with open(os.path.join(log_dir, "goal_reached.txt"), "a") as file:
        file.write(f"{frame_count}\n")

def path_length_txt_data(distance, log_dir):
    with open(os.path.join(log_dir, "total_distance_travelled.txt"), "a") as file:
        file.write(f"{distance}\n")

def total_time_txt_data(time, log_dir):
    with open(os.path.join(log_dir, "total_time.txt"), "a") as file:
        file.write(f"{time:.6f}\n")

def dbscan_time_txt_data(time, log_dir):
    with open(os.path.join(log_dir, "dbscan_time.txt"), "a") as file:
        file.write(f"{time:.6f}\n")

def constraint_step_txt_data(step, log_dir):
    with open(os.path.join(log_dir, "step.txt"), "a") as file:
        file.write(f"{step}\n")



def compute_next_state(x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:

    theta = x[2]
    v, omega = u[0], u[1]

    x_next = x[0] + v * jnp.cos(theta) * dt
    y_next = x[1] + v * jnp.sin(theta) * dt
    theta_next = theta + omega * dt
    theta_next = (theta_next + jnp.pi) % (2 * jnp.pi) - jnp.pi 

    return jnp.array([x_next, y_next, theta_next])


def goal_check(state: jnp.ndarray, goal: jnp.ndarray, tolerance: jnp.ndarray) -> bool:

    distance_to_goal = jnp.linalg.norm(state[:2] - goal[:2])
    theta_to_goal = jnp.abs((state[2] - goal[2] + jnp.pi) % (2 * jnp.pi) - jnp.pi)  # minimal angle diff

    return (distance_to_goal < tolerance[0]) & (theta_to_goal < tolerance[1])



def compute_distance(new_state: jnp.ndarray, previous_state: jnp.ndarray) -> float:

    return jnp.linalg.norm(new_state[:2] - previous_state[:2])
