#!/usr/bin/env python3
import math
import numpy as np
from cuml.cluster import DBSCAN
# from sklearn.cluster import DBSCAN
from constraint.constraint import Constraint

import time

import jax
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap, lax
from jax import debug

class JAXMPPIController():
    def __init__(
        self,
        delta_t: float = 0.05,
        max_vx: float = 0.523,  # [m/s]
        max_wz: float = 2.000,  # [rad/s]
        time_step: int = 30,
        batch_size: int = 1000,
        sigma: jnp.ndarray = jnp.array([[0.7, 0.0], [0.0, 0.5]]),
        ref_path: jnp.ndarray = jnp.array([0.0, 0.0, 0.0, 1.0]),
        param_lambda: float = 50.0,
        param_alpha: float = 1.0,
        param_exploration: float = 0.0,
        stage_cost_weight: jnp.ndarray = jnp.array([50.0, 50.0, 1.0, 20.0]),
        terminal_cost_weight: jnp.ndarray = jnp.array([50.0, 50.0, 1.0, 20.0]),
        obstacles: jnp.ndarray = jnp.empty((0, 3)),
        obstacles_types = ["circle","circle","circle"],
        robot_radius: float = 0.1,
        eta: float = 0.1,
        eps: float = 10.0,
        min_samples: int = 2,
        rho_nu_max: jnp.ndarray = jnp.array([0.5, 0.5]),
        rho_nu_min: jnp.ndarray = jnp.array([0.5, 0.5]),
        visualize_optimal_traj=True,
        visualize_sampled_trajs=False,
        execute_dbscan=False,
) -> None:
        # mppi parameters
        self.dim_x = 3  # dimension of system state vector
        self.dim_u = 2  # dimension of control input vector
        self.time_step = time_step  # prediction horizon
        self.batch_size = batch_size  # number of sample trajectories
        self.param_exploration = param_exploration  # constant parameter of mppi
        self.param_lambda = param_lambda  # constant parameter of mppi
        self.param_alpha = param_alpha  # constant parameter of mppi
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))  # constant parameter of mppi

        # Sigma-related
        self.sigma = sigma
        self.sigma_diag = jnp.diag(self.sigma)
        self.sigma_matrix = jnp.broadcast_to(self.sigma, (batch_size, time_step, 2, 2))
        self.sigma_inv = jnp.linalg.inv(self.sigma)
        self.sigma_inv_diag = jnp.diag(self.sigma_inv)


        # Reference & cost weight
        self.ref_path = ref_path
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight


        # Obstacles & robot
        self.obstacles = obstacles
        self.obstacles_types = obstacles_types
        self.robot_radius = robot_radius

        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualize_sampled_trajs = visualize_sampled_trajs
        self.execute_dbscan = execute_dbscan

        # Vehicle parameters
        self.dt = delta_t
        self.max_vx = max_vx
        self.max_wz = max_wz
        self.min_v = jnp.array([0.0, -max_wz])
        self.max_v = jnp.array([max_vx, max_wz])
        self.epsilon = 0.0
        self.max_iter = 5000

        self.constraints = []  # 빈 리스트로 초기화
        self.opt_clustered_traj = []
        self.clustered_trajs = []

        # Cost & noise containers
        self.S = jnp.zeros((batch_size,))
        self.stage_cost = jnp.zeros((batch_size,))
        self.terminal_cost = jnp.zeros((batch_size,))

        # MPPI Variables
        self.state = jnp.array([0.0, -2.0, math.pi / 2])
        self.x0 = jnp.zeros((self.dim_x,))
        self.u_prev = jnp.zeros((time_step, self.dim_u))
        self.u = jnp.zeros((time_step, self.dim_u))
        self.v = jnp.zeros((batch_size, time_step, self.dim_u))
        self.noise = jnp.zeros((batch_size, time_step, self.dim_u))
        self.opt_u = jnp.zeros((time_step, self.dim_u))
        self.standard_normal_noise = jnp.zeros((batch_size, time_step, 2))
        self.trajectory = jnp.zeros((batch_size, time_step, self.dim_x))

        self.lambda_g = jnp.zeros((self.batch_size,self.time_step))
        self.nu_min = jnp.zeros((self.batch_size,self.time_step,2))
        self.nu_max = jnp.zeros((self.batch_size,self.time_step,2))
        self.adjustment = jnp.zeros((self.batch_size,self.time_step,2))



        self.eta = eta
        self.rho_nu_max = rho_nu_max
        self.rho_nu_min = rho_nu_min
        
        self.batch_indices = jnp.arange(batch_size).reshape(-1, 1)            # shape: (B, 1)
        self.batch_indices = jnp.tile(self.batch_indices, (1, time_step+1 ))  # shape: (B, T + 1)

        self.time_indices = jnp.arange(time_step+1).reshape(1, -1)        # shape: (1, T + 1)
        self.time_indices = jnp.tile(self.time_indices, (batch_size, 1)) # shape: (B, T + 1)

        self.eps = eps
        self.min_samples = min_samples

        self.dbscan_elapsed_time = 0.0

        self.key = jax.random.PRNGKey(int(time.time()))


        self.obstacle_centers = self.obstacles[:, :2]  # shape: (num_obstacles, 2)
        self.obstacle_radii = self.obstacles[:, 2]     # shape: (num_obstacles,)
        
        #constraint
        self.constraint_model = Constraint(
            robot_radius=self.robot_radius,
        )


    @staticmethod
    @partial(jit, static_argnames=["batch_size", "time_step"])
    def generate_noise_and_sampling(key, batch_size, time_step, std_vx=0.1, std_wz=1.0):
        key1, key2 = jax.random.split(key)

        noise_vx = jax.random.normal(key1, shape=(batch_size, time_step)) * std_vx
        noise_wz = jax.random.normal(key2, shape=(batch_size, time_step)) * std_wz

        standard_noise = jnp.stack([noise_vx, noise_wz], axis=-1)  # shape: (B, T, 2)

        return standard_noise, key2

    
    @staticmethod
    @jit
    def predict_trajectory(initial_state: jnp.ndarray, v: jnp.ndarray, dt: float):

        batch_size, time_step, _ = v.shape
        vx = v[:, :, 0]
        wz = v[:, :, 1]

        # Expand initial state across batch
        x0 = jnp.broadcast_to(initial_state, (batch_size, 3))  # shape: (B, 3)

        # Theta integration
        dtheta = wz * dt
        theta_cumsum = jnp.cumsum(dtheta, axis=1)  # (B, T)
        theta = x0[:, 2:3] + theta_cumsum
        theta = (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi  # wrap to [-pi, pi]

        # Shift theta for dx, dy integration
        theta_shifted = jnp.concatenate([x0[:, 2:3], theta[:, :-1]], axis=1)
        dx = vx * jnp.cos(theta_shifted) * dt
        dy = vx * jnp.sin(theta_shifted) * dt

        x = x0[:, 0:1] + jnp.cumsum(dx, axis=1)
        y = x0[:, 1:2] + jnp.cumsum(dy, axis=1)

        trajectory = jnp.stack([x, y, theta], axis=-1)
        return trajectory
    
    @staticmethod
    @jit
    def predict_cluster_trajectory(initial_state: jnp.ndarray,
                                   v: jnp.ndarray,
                                   dt: float) -> jnp.ndarray:
        # unpack controls
        vx = v[:, 0]            # (T,)
        wz = v[:, 1]            # (T,)

        # integrate heading
        dtheta = wz * dt                                    # (T,)
        theta_cumsum = jnp.cumsum(dtheta)                   # (T,)
        theta = initial_state[2] + theta_cumsum             # (T,)
        # wrap to [-π, π]
        theta = (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi    # (T,)

        # shift for x/y integration
        theta_shifted = jnp.concatenate([
            initial_state[2:3],      # theta₀ as first element
            theta[:-1]               # previous headings
        ], axis=0)                   # (T,)

        # integrate positions
        dx = vx * jnp.cos(theta_shifted) * dt               # (T,)
        dy = vx * jnp.sin(theta_shifted) * dt               # (T,)
        x = initial_state[0] + jnp.cumsum(dx)               # (T,)
        y = initial_state[1] + jnp.cumsum(dy)               # (T,)

        # stack into trajectory
        trajectory = jnp.stack([x, y, theta], axis=1)       # (T, 3)
        return trajectory
    
    
    
    @staticmethod
    @jit
    def compute_total_cost(
        trajectory: jnp.ndarray,      # (B, T, 3)
        v: jnp.ndarray,               # (B, T, 2)
        u: jnp.ndarray,               # (T, 2)
        ref_path: jnp.ndarray,        # (4,)
        stage_cost_weight: jnp.ndarray,    # (4,)
        terminal_cost_weight: jnp.ndarray, # (4,)
        robot_radius: float,
        obstacles: jnp.ndarray,       # (num_obs, 3)
        sigma_inv_diag: jnp.ndarray,  # (T, 2)
        param_gamma: float
    ) -> jnp.ndarray:
        ref_x, ref_y, ref_yaw, ref_v = ref_path

        # Stage cost (exclude final step)
        stage_cost = (
            stage_cost_weight[0] * jnp.square(trajectory[:, :-1, 0] - ref_x) +
            stage_cost_weight[1] * jnp.square(trajectory[:, :-1, 1] - ref_y) +
            stage_cost_weight[2] * jnp.square(trajectory[:, :-1, 2] - ref_yaw) +
            stage_cost_weight[3] * jnp.square(v[:, :-1, 0] - ref_v)
        ).sum(axis=1)

        # Terminal cost (last step only)
        terminal_cost = (
            terminal_cost_weight[0] * jnp.square(trajectory[:, -1, 0] - ref_x) +
            terminal_cost_weight[1] * jnp.square(trajectory[:, -1, 1] - ref_y) +
            terminal_cost_weight[2] * jnp.square(trajectory[:, -1, 2] - ref_yaw) +
            terminal_cost_weight[3] * jnp.square(v[:, -1, 0] - ref_v)
        )

        # Collision penalty
        # collision_penalty = JAXMPPIController.compute_collision_penalty(
        #     trajectory[:, :, 0], trajectory[:, :, 1],
        #     robot_radius, obstacles
        # )

        # Quad term: gamma * sum(u * sigma_inv_diag * v)
        u_exp = jnp.expand_dims(u, axis=0)  # shape: (1, T, 2)
        quad_term = jnp.sum(u_exp * sigma_inv_diag * v, axis=-1)  # shape: (B,)
        quad_cost = param_gamma * jnp.sum(quad_term, axis=1)      # shape: (B,)

        total_cost = stage_cost + terminal_cost + quad_cost #+ collision_penalty
        return total_cost

    @staticmethod
    @jit
    def compute_collision_penalty(
        x: jnp.ndarray,  # shape: (B, T)
        y: jnp.ndarray,  # shape: (B, T)
        robot_radius: float,
        obstacles: jnp.ndarray  # shape: (num_obstacles, 3)
    ) -> jnp.ndarray:

        obs_x = obstacles[:, 0]  # (num_obstacles,)
        obs_y = obstacles[:, 1]
        obs_r = obstacles[:, 2]

        # (B, T, 1)
        x = jnp.expand_dims(x, axis=-1)  # (B, T, 1)
        y = jnp.expand_dims(y, axis=-1)

        dx = x - obs_x  # (B, T, num_obs)
        dy = y - obs_y
        dist_sq = dx**2 + dy**2

        combined_radius_sq = (obs_r + robot_radius)**2  # (num_obstacles,)
        collision_mask = dist_sq <= combined_radius_sq  # (B, T, num_obs)
        collision_detected = jnp.any(collision_mask, axis=(1, 2))  # shape: (B,)

        penalty = jnp.where(collision_detected, 1.0e4, 0.0)
        return penalty  # shape: (B,)
    

    @staticmethod
    @jit
    def compute_cluster_total_cost(
        trajectory: jnp.ndarray,  # (T, 3)
        v: jnp.ndarray,           # (T, 2)
        ref_path: jnp.ndarray,    # (4,)
        stage_cost_weight: jnp.ndarray,    # (4,)
        terminal_cost_weight: jnp.ndarray, # (4,)
        robot_radius: float,
        obstacles: jnp.ndarray    # (num_obs, 3)
    ) -> jnp.ndarray:
        ref_x, ref_y, ref_yaw, ref_v = ref_path

        # Stage cost (exclude final step)
        stage_cost = (
            stage_cost_weight[0] * jnp.square(trajectory[:-1, 0] - ref_x) +
            stage_cost_weight[1] * jnp.square(trajectory[:-1, 1] - ref_y) +
            stage_cost_weight[2] * jnp.square(trajectory[:-1, 2] - ref_yaw) +
            stage_cost_weight[3] * jnp.square(v[:-1,       0] - ref_v)
        ).sum()

        # Terminal cost (only last step)
        terminal_cost = (
            terminal_cost_weight[0] * jnp.square(trajectory[-1, 0] - ref_x) +
            terminal_cost_weight[1] * jnp.square(trajectory[-1, 1] - ref_y) +
            terminal_cost_weight[2] * jnp.square(trajectory[-1, 2] - ref_yaw) +
            terminal_cost_weight[3] * jnp.square(v[-1,       0] - ref_v)
        )

        # Collision penalty
        # collision_penalty = JAXMPPIController.compute_cluster_collision_penalty(
        #     trajectory[:, 0], trajectory[:, 1],
        #     robot_radius, obstacles
        # )

        total_cost = stage_cost + terminal_cost #+ collision_penalty
        return total_cost
    

    @staticmethod
    @jit
    def compute_cluster_collision_penalty(
        x: jnp.ndarray,          # shape: (time_step,)
        y: jnp.ndarray,          # shape: (time_step,)
        robot_radius: float,
        obstacles: jnp.ndarray   # shape: (num_obstacles, 3)
    ) -> jnp.ndarray:

        # 장애물 정보
        obs_x = obstacles[:, 0]   # (num_obstacles,)
        obs_y = obstacles[:, 1]
        obs_r = obstacles[:, 2]

        x = jnp.expand_dims(x, axis=-1)  # (T, 1)
        y = jnp.expand_dims(y, axis=-1)

        dx = x - obs_x     
        dy = y - obs_y
        dist_sq = dx**2 + dy**2

        combined_radius_sq = (obs_r + robot_radius)**2  # (num_obstacles,)
        collision_mask = dist_sq <= combined_radius_sq  # (T, num_obstacles)
        collision_detected = jnp.any(collision_mask)    # scalar boolean
        penalty = jnp.where(collision_detected, 1.0e4, 0.0)

        return penalty



    
    @staticmethod
    @jit
    def compute_opt_u(S: jnp.ndarray,noise: jnp.ndarray,u: jnp.ndarray ,param_lambda: float) -> jnp.ndarray:
        w = jax.nn.softmax(-S / param_lambda)
        w_expanded = jnp.expand_dims(w, axis=(1,2))
        w_epsilon = jnp.sum(w_expanded * noise, axis=0)  
        return u + w_epsilon
    
    @staticmethod
    @jit
    def apply_constraint(u: jnp.ndarray, min_v: jnp.ndarray, max_v: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(u, min_v, max_v)


    def dbscan(self, v, S, noise, trajectory, u, x0):
        # dbscan clustering
        v_mean = jnp.mean(v, axis=1)  # (N, 2)
        S_expanded = jnp.expand_dims(S, axis=1)  # (N, 1)
        combined_np = jnp.concatenate([v_mean, S_expanded], axis=1)
        combined_np = np.array(combined_np)
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(combined_np)
        self.labels = labels
        unique_labels = set(self.labels)
        self.labels = jnp.array(self.labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if num_clusters == 0:
            opt_u = JAXMPPIController.compute_opt_u(S, noise,u, self.param_lambda)  # (N,)
            return opt_u
        else:
            cluster_us = []
            cluster_costs = []
            self.clustered_labels = []
            self.opt_clustered_traj = []

            for label in unique_labels:
                if label == -1:
                    self.clustered_labels.append(label)
                    continue

                mask = (self.labels == label)
                self.clustered_labels.append(label)
                S_masked = jnp.where(mask, S, jnp.inf)
                cluster_u = JAXMPPIController.compute_opt_u(S_masked, noise, u, self.param_lambda)  # (N,)
                cluster_traj = JAXMPPIController.predict_cluster_trajectory(
                    x0, cluster_u, self.dt
                )             
                cost = JAXMPPIController.compute_cluster_total_cost(
                    cluster_traj,
                    cluster_u,
                    self.ref_path,
                    self.stage_cost_weight,
                    self.terminal_cost_weight,
                    self.robot_radius,
                    self.obstacles
                )
                cluster_us.append(cluster_u)
                cluster_costs.append(cost)
                self.opt_clustered_traj.append((label, cluster_traj))

            cluster_costs = np.array(cluster_costs)
            min_idx = int(np.argmin(cluster_costs))


            return cluster_us[min_idx]
        



    
    def set_zero(self):
        self.S = jnp.zeros_like(self.S)
        self.u = jnp.zeros_like(self.u)
        self.v = jnp.zeros_like(self.v)
        self.trajectory = jnp.zeros_like(self.trajectory)
        self.nu_min = jnp.zeros_like(self.nu_min)
        self.nu_max = jnp.zeros_like(self.nu_max)
        self.adjustment = jnp.zeros_like(self.adjustment)

        

    @staticmethod
    @partial(jit,
             static_argnames=["constraint_model"])
    def update_control_inputs_while(
        v: jnp.ndarray,            
        trajectory: jnp.ndarray,  
        x0: jnp.ndarray,       
        min_v: jnp.ndarray,     
        max_v: jnp.ndarray,    
        eta: float,
        rho_nu_min: float,
        rho_nu_max: float,
        batch_indices: jnp.ndarray, 
        time_indices: jnp.ndarray,  
        constraint_model,
        obstacles_center,
        obstacles_radii,
        dt: float,
        epsilon: float = 1e-3
    ):
        batch_size = v.shape[0]
        nu_min     = jnp.zeros_like(v)
        nu_max     = jnp.zeros_like(v)
        min_indices = jnp.zeros_like(time_indices)
        inside_trajectory_samples = jnp.zeros((batch_size,), dtype=bool)
        # step counter
        step = 0
        # carry: (v, trajectory, nu_min, nu_max, min_indices, inside_samples, step)
        init_carry = (v, trajectory, nu_min, nu_max, min_indices, inside_trajectory_samples, step, jnp.ones((batch_size,), dtype=bool))

        def cond_fn(carry):
            v, trajectory, nu_min, nu_max, min_indices, inside_samples, step, _ = carry
            max_iter = 5000
            # compute violation condition
            g_all = constraint_model.boundary_eq_batch(trajectory[:, :, :2], obstacles_center, obstacles_radii)
            g_values = jnp.min(g_all, axis=0)
            outside_boundary = g_values > 0.0
            all_v_valid = jnp.all((v >= min_v - epsilon) & (v <= max_v + epsilon))
            violation = jnp.logical_not(jnp.all(outside_boundary) & all_v_valid)
            # continue if step < max_iters and violation exists
            return (violation) & (step < max_iter)

        def body_fn(carry):
            v, trajectory, nu_min, nu_max, min_indices, inside_samples, step, _ = carry
            
            traj = jnp.concatenate([jnp.broadcast_to(x0, (batch_size, 1, 3)), trajectory], axis=1)

            # same scan_body logic
            g_all = constraint_model.boundary_eq_batch(traj[:, :, :2], obstacles_center, obstacles_radii)
            g_values = jnp.min(g_all, axis=0)
            g_idx = jnp.argmin(g_all, axis=0)
            min_indices = g_idx[:, :]
            outside_boundary = g_values > 0.0
            inside_trajectory_samples = ~jnp.all(outside_boundary, axis=1)
            v_valid = (v >= min_v - epsilon) & (v <= max_v + epsilon)

            nu_min = jnp.clip(nu_min + rho_nu_min * (min_v - v), a_min=0.0, a_max=None)
            nu_max = jnp.clip(nu_max + rho_nu_max * (v - max_v), a_min=0.0, a_max=None)
            # nu_min = jnp.where(v_valid, 0.0, nu_min)
            # nu_max = jnp.where(v_valid, 0.0, nu_max)

            # apply correction
            adjustment = jnp.zeros_like(v)

            grad_all = constraint_model.gradient_batch(traj[:, :, :2], obstacles_center, obstacles_radii)
            grad_all = jnp.transpose(grad_all, (1, 2, 0, 3))

            grad_sel = grad_all[batch_indices, time_indices, min_indices, :]

            grad_u_all = JAXMPPIController.compute_gradient_u(grad_sel ,traj, v, dt)
            
            grad_v = grad_u_all[:, :, 0]  # (B, T+1)
            grad_w = grad_u_all[:, :, 1] 
            mean_grad_v = jnp.mean(jnp.abs(grad_v), axis=1)  # (B,)
            mean_grad_w = jnp.mean(jnp.abs(grad_w), axis=1)
            
            scale_v = 0.1 / (mean_grad_v + 1e-6)
            scale_w = 1.0 / (mean_grad_w + 1e-6)

            scale_v = scale_v[:, None]  # (B, 1)
            scale_w = scale_w[:, None]  # (B, 1)



            eligible_samples_all = jnp.expand_dims(
                    jnp.expand_dims(inside_trajectory_samples, axis=1),
                    axis=-1                                             # -> (B, 1, 1)
                )
            grad_sel = grad_u_all * eligible_samples_all

            upd1_v = eta * (-nu_min[:, :, 0] + nu_max[:, :, 0] - scale_v * grad_sel[:,:,0])
            upd1_w = eta * (-nu_min[:, :-1, 1] + nu_max[:, :-1, 1] - scale_w * grad_sel[:,:-1,1])
            upd2_w = eta * (-nu_min[:, -1, 1] + nu_max[:, -1, 1])

            adjustment = adjustment.at[:, :, 0].set(upd1_v)
            adjustment = adjustment.at[:, :-1, 1].set(upd1_w)
            adjustment = adjustment.at[:, -1, 1].set(upd2_w)
            v = v - adjustment

            trajectory = JAXMPPIController.predict_trajectory(x0, v, dt)

            # increment step
            step += 1
            violation = jnp.logical_not(jnp.all(outside_boundary, axis=1) & jnp.all(v_valid, axis=(1, 2)))

            return (v, trajectory, nu_min, nu_max, min_indices, inside_trajectory_samples, step,violation)


        v_final, traj_final,nu_min,nu_max, *_, step, violation = lax.while_loop(cond_fn, body_fn, init_carry)
        return traj_final, v_final, step, violation, nu_min, nu_max
    
    @staticmethod
    @jit
    def compute_gradient_u(
        grad_g_values: jnp.ndarray,         
        trajectory: jnp.ndarray,    
        v: jnp.ndarray,           
        dt: float
    ) -> jnp.ndarray:

        theta      = trajectory[:, 1:, 2]
        theta_prev = trajectory[:, :-1, 2]
        vx         = v[:, :, 0]
        
        sin_theta_prev = jnp.sin(theta_prev)
        cos_theta_prev = jnp.cos(theta_prev)
        sin_theta      = jnp.sin(theta)
        cos_theta      = jnp.cos(theta)
        
        dt2 = dt * dt

        # B: shape (B, T, 2, 2)
        B = jnp.stack([
            jnp.stack([cos_theta_prev * dt, sin_theta_prev * dt], axis=-1),
            jnp.stack([-vx * sin_theta * dt2, vx * cos_theta * dt2], axis=-1)
        ], axis=-2)  # shape: (B, T, 2, 2)

        grad_g_t = grad_g_values[:,1:,:]
        grad_g_t = grad_g_t.at[:,0,:].set(grad_g_values[:,0,:])
        grad_g_tp1 = jnp.zeros_like(grad_g_t)

        grad_g_tp1 = grad_g_tp1.at[:,:-1,:].set(grad_g_values[:,2:,:])
        grad_u_trans = jnp.sum(B[:,:,0,:] * grad_g_t, axis=-1)
        grad_u_rot = jnp.sum(B[:,:,1,:]* grad_g_tp1, axis=-1)
        grad_u = jnp.stack([grad_u_trans,grad_u_rot], axis=-1)

        return grad_u
    
    def check_collision(self) -> None:

        pos = self.state[:2] 
        centers = self.obstacles[:, :2] 
        radii = self.obstacles[:, 2]  

        dists = jnp.linalg.norm(centers - pos, axis=1)
        collision_flags = dists < (radii + self.robot_radius)  

        if jnp.any(collision_flags):
            raise RuntimeError("collision")
                
    def compute_plot_data(self, trajectory):


        optimal_traj = jnp.zeros((self.time_step, self.dim_x))
        if self.visualize_optimal_traj:
            optimal_traj = JAXMPPIController.predict_cluster_trajectory(
                self.state,
                self.u_prev,
                self.dt,
            )
        plotted_clustered_trajs = []
        if self.execute_dbscan:

            labels_np = np.array(self.labels)
            traj_np = np.array(trajectory)

            for label in self.clustered_labels:
                idxs = np.where(labels_np == label)[0]
                cluster_traj_batch = traj_np[idxs]

                if label == -1:
                    plotted_clustered_trajs.append(("noise", cluster_traj_batch))
                else:
                    plotted_clustered_trajs.append((label, cluster_traj_batch))

        return optimal_traj, self.opt_clustered_traj, plotted_clustered_trajs

    
    def move_obstacle(self):

        step_size = 0.016
        should_move = self.obstacles[0, 0] < 0.5
        delta = jnp.where(should_move, step_size, 0.0)
        self.obstacles = self.obstacles.at[0, 0].add(delta)
        self.obstacle_centers = self.obstacles[:, :2] 

    def compute_control_input(self):


        x0 = self.state
        u = self.u_prev

        noise, new_key = JAXMPPIController.generate_noise_and_sampling(
            self.key, self.batch_size, self.time_step
        )
        self.key = new_key
        v = u + noise

        # v = JAXMPPIController.apply_constraint(
        #     noise + u, self.min_v, self.max_v
        # )



        trajectory = JAXMPPIController.predict_trajectory(
            x0, v, self.dt
        )

       
        trajectory, v, self.step,violation, nu_min,nu_max  = JAXMPPIController.update_control_inputs_while(
                        v,
                        trajectory,x0,
                        self.min_v,self.max_v,
                        self.eta,
                        self.rho_nu_min,
                        self.rho_nu_max,
                        self.batch_indices,
                        self.time_indices,
                        self.constraint_model,
                        self.obstacle_centers,
                        self.obstacle_radii,
                        self.dt,
                        self.epsilon
                        )
        noise = v - u
        
        S = JAXMPPIController.compute_total_cost(
            trajectory, v, u,
            self.ref_path,
            self.stage_cost_weight,
            self.terminal_cost_weight,
            self.robot_radius,
            self.obstacles,
            self.sigma_inv_diag,
            self.param_gamma
        )

        if self.execute_dbscan:
            dbscan_start_time = time.perf_counter()
            opt_u = self.dbscan(v, S, noise, trajectory, u, x0)
            self.dbscan_elapsed_time = time.perf_counter() - dbscan_start_time
        else:

            opt_u = JAXMPPIController.compute_opt_u(S,noise,u, self.param_lambda) 

        self.u_prev = opt_u
        return opt_u[0], opt_u, trajectory

    
    
    

    def set_state(self,state):
        self.state = state

    def set_ref(self, ref_path):
        self.ref_path = ref_path





