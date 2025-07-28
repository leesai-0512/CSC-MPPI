#!/usr/bin/env python3
import math
import torch
from cuml.cluster import DBSCAN
from constraint.constraint import Constraint
import torch.nn.functional as F
import time



class MPPIController():
    def __init__(
        self,
        delta_t: float = 0.05,
        max_vx: float = 0.523,  # [m/s]
        max_wz: float = 2.000,  # [rad/s]
        time_step: int = 30,
        batch_size: int = 1000,
        sigma: torch.Tensor = torch.tensor([[0.7, 0.0], [0.0, 0.5]]),  
        ref_path: torch.Tensor = torch.tensor([0.0, 0.0, 0.0, 1.0]),
        param_lambda: float = 50.0,
        param_alpha: float = 1.0,
        param_exploration: float = 0.0,
        stage_cost_weight: torch.Tensor = torch.tensor([50.0, 50.0, 1.0, 20.0]), 
        terminal_cost_weight: torch.Tensor = torch.tensor([50.0, 50.0, 1.0, 20.0]), 
        obstacles: torch.Tensor = torch.tensor([], dtype=torch.float32), 
        obstacles_types = ["circle","circle","circle"],
        robot_radius: float = 0.1,
        eta: float = 0.1,
        eps: float = 10.0,
        min_samples: int = 2,
        rho_nu_max: torch.Tensor = torch.tensor([0.5, 0.5]),
        rho_nu_min: torch.Tensor = torch.tensor([0.5, 0.5]),
        visualize_optimal_traj=True,
        visualize_sampled_trajs=False,
        execute_dbscan=False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # specify device
) -> None:
        # mppi parameters
        self.device = device  # Set device (e.g., CPU or GPU)
        self.dim_x = 3  # dimension of system state vector
        self.dim_u = 2  # dimension of control input vector
        self.time_step = time_step  # prediction horizon
        self.batch_size = batch_size  # number of sample trajectories
        self.param_exploration = param_exploration  # constant parameter of mppi
        self.param_lambda = param_lambda  # constant parameter of mppi
        self.param_alpha = param_alpha  # constant parameter of mppi
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))  # constant parameter of mppi

        # Convert all inputs to tensors and move them to the specified device
        self.sigma = sigma.to(self.device)  # deviation of noise
        self.sigma_diag = self.sigma.diag()
        self.sigma_matrix = self.sigma.expand(self.batch_size, self.time_step, -1, -1)
        self.sigma_inv = torch.linalg.inv(self.sigma)
        self.sigma_inv_diag = self.sigma_inv.diag()  # shape: (2,)
        self.ref_path = ref_path.to(self.device)  # reference path
        self.stage_cost_weight = stage_cost_weight.to(self.device)
        self.terminal_cost_weight = terminal_cost_weight.to(self.device)
        self.obstacles = obstacles.to(self.device)
        self.obstacles_types = obstacles_types
        self.robot_radius = robot_radius
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualize_sampled_trajs = visualize_sampled_trajs
        self.execute_dbscan = execute_dbscan

        # vehicle parameters
        self.dt = delta_t #[s]
        self.max_vx = max_vx # [rad]
        self.max_wz = max_wz # [m/s^2]
        self.min_v = torch.tensor([0.0, -self.max_wz], device = self.device)
        self.max_v = torch.tensor([self.max_vx, self.max_wz], device = self.device)

        self.constraints = [] 
        self.opt_clustered_traj = []
        self.clustered_trajs = []

        self.S = torch.zeros((self.batch_size), device=self.device)
        self.stage_cost = torch.zeros((self.batch_size), device=self.device)
        self.terminal_cost = torch.zeros((self.batch_size), device=self.device)

        # mppi variables
        self.state = torch.tensor([0.0, -2.0, math.pi / 2], device=self.device)
        self.u_prev = torch.zeros((self.time_step, self.dim_u), device=self.device)
        self.u = torch.zeros((self.dim_u), device=self.device)
        self.x0 = torch.zeros((self.dim_x), device=self.device)
        self.u_prev = torch.zeros((self.time_step, self.dim_u), device=self.device)
        self.u = torch.zeros((time_step, self.dim_u), device=self.device)
        self.v = torch.zeros((batch_size,time_step,self.dim_u), device=self.device)
        self.noise = torch.zeros((batch_size,time_step,self.dim_u), device=self.device)
        self.opt_u = torch.zeros((time_step, self.dim_u), device=self.device)
        self.standard_normal_noise = torch.zeros(batch_size, time_step, 2, device=device)
        self.trajectory = torch.zeros((batch_size,time_step,self.dim_x), device=device)

        self.lambda_g = torch.zeros(self.batch_size,self.time_step, device=self.device)
        self.nu_min = torch.zeros(self.batch_size,self.time_step,2, device=self.device)
        self.nu_max = torch.zeros(self.batch_size,self.time_step,2, device=self.device)
        self.adjustment = torch.zeros(self.batch_size,self.time_step,2,device=self.device)


        self.eta = eta
        self.rho_nu_max = rho_nu_max.to(self.device)
        self.rho_nu_min = rho_nu_min.to(self.device)
        
        self.batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, time_step + 1).to(self.device)
        self.time_indices = torch.arange(time_step+1).unsqueeze(0).expand(batch_size, time_step + 1).to(self.device)


        self.eps = eps
        self.min_samples = min_samples
        self.dbscan_elapsed_time = 0.0


        self.total_iter = 0



    def generateNoiseAndSampling(self):

        self.standard_normal_noise[:,:,0].normal_(0,0.1)
        self.standard_normal_noise[:,:,1].normal_(0,1.0)

        return  self.standard_normal_noise

    
    def predict_trajectory(self, initial_state, v):

        num_samples, num_timesteps, _ = v.shape
        trajectory = torch.empty((num_samples, num_timesteps, 3), device=self.device)
        vx = v[:,:,0]
        wz = v[:,:,1]

        x0 = initial_state.unsqueeze(0).expand(num_samples, -1)  


        dtheta = wz * self.dt 
        theta_cumsum = torch.cumsum(dtheta, dim=1)  

        trajectory[:, :, 2].copy_(x0[:, 2].unsqueeze(1) + theta_cumsum)
        trajectory[:, :, 2] = (trajectory[:, :, 2] + torch.pi) % (2 * torch.pi) - torch.pi

        theta_calc = torch.cat([x0[:, 2].unsqueeze(1), trajectory[:, :-1, 2]], dim=1)


        cos_theta, sin_theta = torch.cos(theta_calc), torch.sin(theta_calc)


        dx = vx * cos_theta * self.dt 
        dy = vx * sin_theta * self.dt 

        trajectory[:, :, 0].copy_(x0[:, 0].unsqueeze(1) + torch.cumsum(dx, dim=1))
        trajectory[:, :, 1].copy_(x0[:, 1].unsqueeze(1) + torch.cumsum(dy, dim=1))


        return trajectory
    
    
    def compute_total_cost(self, trajectory: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        ref_x, ref_y, ref_yaw, ref_v = self.ref_path

        stage_cost = (
            self.stage_cost_weight[0] * torch.square(trajectory[:, :-1, 0] - ref_x)
            + self.stage_cost_weight[1] * torch.square(trajectory[:, :-1, 1] - ref_y)
            + self.stage_cost_weight[3] * torch.square(v[:, :-1, 0] - ref_v)
            + self.stage_cost_weight[2] * torch.square(trajectory[:, :-1, 2] - ref_yaw)
        ).sum(dim=1)  # (batch_size,)


        terminal_cost = (
            self.terminal_cost_weight[0] * torch.square(trajectory[:, -1, 0] - ref_x)
            + self.terminal_cost_weight[1] * torch.square(trajectory[:, -1, 1] - ref_y)
            + self.terminal_cost_weight[3] * torch.square(v[:,-1,0] - ref_v)
            + self.terminal_cost_weight[2] * torch.square(trajectory[:, -1, 2] - ref_yaw)
        )
        total_collision_penalty = self.compute_collision_penalty(trajectory[:, :, 0], trajectory[:, :, 1])  # (batch_size,)
        total_cost = stage_cost + terminal_cost + total_collision_penalty
        return total_cost

    def compute_collision_penalty(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:


        obstacles = self.obstacles  # (num_obstacles, 3)
        obs_x, obs_y, obs_r = obstacles[:, 0], obstacles[:, 1], obstacles[:, 2]  # (num_obstacles,)

        x = x[...,None]  # (batch_size, time_step-1, 1)
        y = y[...,None]  # (batch_size, time_step-1, 1)


        dist_sq = (x - obs_x) * (x - obs_x) + (y - obs_y) * (y - obs_y)  # (batch_size, time_step-1, num_obstacles)


        if not hasattr(self, "combined_radius_sq"):
            self.combined_radius_sq = (obs_r + self.robot_radius) * (obs_r + self.robot_radius)  # (num_obstacles,)

        collision_mask = dist_sq <= self.combined_radius_sq  # (batch_size, time_step-1, num_obstacles)
        if(x.dim()==2):
            collision_penalty = collision_mask.any(dim=(1)).float() * 1.0e4  # (batch_size,)
        else:
            collision_penalty = collision_mask.any(dim=(1, 2)).float() * 1.0e4  # (batch_size,)

        return collision_penalty

    
    def compute_weights(self, S: torch.Tensor) -> torch.Tensor:
        return torch.softmax(-S / self.param_lambda, dim=0)
    
    def apply_constraint(self, u: torch.Tensor) -> torch.Tensor:
        u.clamp_(
            min=self.min_v, 
            max=self.max_v  
        )
        return u  
    
    def dbscan(self, v, S, noise, trajectory, u, x0):

            v_mean = v.mean(dim=1)
            combined_np = torch.cat((v_mean, S.unsqueeze(1)), dim=1).cpu().numpy()
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(combined_np)
            unique_labels = set(labels)
            num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

            if(num_clusters == 0):
                w = self.compute_weights(S)
                w_expanded = w.view(-1, 1, 1)
                w_epsilon = torch.sum(w_expanded * noise, dim=0)
                return (u + w_epsilon)

            else:

                cluster_us = []         
                cluster_costs = []  
                self.clustered_trajs = []    
                self.opt_clustered_traj = []

                for label in unique_labels:
                    if label == -1:

                        self.clustered_trajs.append(("noise", trajectory[labels == -1]))
                        continue

                    mask = (labels == label)
                    cluster_v = v[mask]
                    cluster_s = S[mask]
                    cluster_traj_batch = trajectory[mask] 

                    # weighted control
                    cluster_noise = cluster_v - u
                    weights = self.compute_weights(cluster_s).view(-1, 1, 1)
                    weighted_eps = torch.sum(weights * cluster_noise, dim=0)
                    cluster_u = u + weighted_eps

                    # trajectory + cost
                    cluster_traj = self.predict_trajectory(x0, cluster_u.unsqueeze(0)).squeeze(0)
                    cost = self.compute_total_cost(cluster_traj.unsqueeze(0), cluster_u.unsqueeze(0)).item()

                    cluster_us.append(cluster_u)
                    cluster_costs.append(cost)
                    self.clustered_trajs.append((label, cluster_traj_batch))
                    self.opt_clustered_traj.append((label, cluster_traj))

                min_idx = torch.argmin(torch.tensor(cluster_costs))

                return cluster_us[min_idx]

    
    def set_zero(self):
        self.S.zero_()
        self.u.zero_()
        self.v.zero_()
        self.trajectory.zero_()
        self.nu_min.zero_()
        self.nu_max.zero_()
        self.adjustment.zero_()

    def set_obstacle(self):


        self.obstacle_centers = self.obstacles[:, :2]  # (num_obstacles, 2)
        self.obstacle_radii = self.obstacles[:, 2]  # (num_obstacles,)

        type_mapping = {"circle": 0, "cardioid": 1, "moon": 2}
        self.obstacle_types = torch.tensor(
            [type_mapping[t] for t in self.obstacles_types], dtype=torch.int32, device=self.device
        )  

        self.constraint_model = Constraint(
            robot_radius=self.robot_radius, 
            device=self.device,
            obstacle_centers=self.obstacle_centers,
            obstacle_radii=self.obstacle_radii,
            obstacle_types=self.obstacle_types
        )

    def update_control_inputs(self, trajectory, v, x0):

        nu_min = self.nu_min
        nu_max = self.nu_max
        adjustment = self.adjustment
        eta = self.eta
        rho_nu_max = self.rho_nu_max
        rho_nu_min = self.rho_nu_min
        batch_indices = self.batch_indices
        time_indices = self.time_indices
        x0_expand = x0.unsqueeze(0).expand(trajectory.shape[0], -1).unsqueeze(1)
        check_convergence = 0

        while True:
            
            traj = torch.cat([x0_expand, trajectory], dim=1)

            g_values_all_real = self.constraint_model.boundary_eq_batch(traj[:, :, :2]) 
            g_values, g_values_all_min_indices = torch.min(g_values_all_real, dim=0)  
            min_indices = g_values_all_min_indices[:, :]
            outside_boundary = g_values > 0 

            inside_trajectory_samples = ~torch.all(outside_boundary, dim=1)

            v_valid = ((v >= (self.min_v)) & (v <= (self.max_v)))

            nu_min = F.relu(nu_min + rho_nu_min * (self.min_v - v), inplace=True)
            nu_max = F.relu(nu_max + rho_nu_max * (v - self.max_v), inplace=True)

            # nu_min[v_valid] = 0.0
            # nu_max[v_valid] = 0.0
            
            all_v_valid = v_valid.all()

            
            if outside_boundary.flatten().all() and all_v_valid:
                print(check_convergence)
                break

            check_convergence += 1

            grad_g_values_all = self.constraint_model.gradient_batch(traj[:, :, :2]) 
            grad_g_values_all = grad_g_values_all.permute(1, 2, 0, 3)  
            grad_g_values_selected = grad_g_values_all[batch_indices, time_indices, min_indices, :]
            grad_u_all = self.compute_gradient_u(grad_g_values_selected,traj, v)



            grad_v = grad_u_all[:, :, 0]
            grad_w = grad_u_all[:, :, 1]
            mean_grad_v = grad_v.abs().mean(dim=1) 
            mean_grad_w = grad_w.abs().mean(dim=1)
            scale_v = 0.1 / (mean_grad_v + 1e-6)
            scale_w = 1.0 / (mean_grad_w + 1e-6)
            scale_v = scale_v[:, None] 
            scale_w = scale_w[:, None]  

            eligible_samples_all = (inside_trajectory_samples.unsqueeze(1)).unsqueeze(-1)

            grad_g_selected = grad_u_all * eligible_samples_all

            
            adjustment[:,:,0] =   eta*(-nu_min[:,:,0] + nu_max[:,:,0] - scale_v* grad_g_selected[:,:,0]) 
            adjustment[:,:-1,1] = eta*(-nu_min[:,0:-1,1] + nu_max[:,0:-1,1] - scale_w* grad_g_selected[:,:-1,1]) 
            adjustment[:,-1, 1] = eta*(-nu_min[:,-1,1] + nu_max[:,-1,1])
            

            
            if check_convergence > 5000:
                raise RuntimeError("CSC-MPPI control input update failed due to convergence error.")
                
            v -= adjustment  
            trajectory = self.predict_trajectory(x0,v)



        return trajectory, v


    def compute_gradient_u(self, grad_g_values ,trajectory, u_t):

        theta = trajectory[:, 1:, 2]
        theta_prev = trajectory[:, :-1, 2]
        vx = u_t[:, :, 0]

        sin_theta_prev = torch.sin(theta_prev)
        cos_theta_prev = torch.cos(theta_prev)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        dt = self.dt
        dt2 = dt * dt

        
        B_trans = torch.stack([cos_theta_prev * dt, sin_theta_prev * dt], dim=-1)
        B_rot   = torch.stack([-vx * sin_theta * dt2, vx * cos_theta * dt2], dim=-1)
        B = torch.stack([B_trans, B_rot], dim=-2)

        grad_g_t     = grad_g_values[:,1:,:] 
        grad_g_t[:,0,:] = grad_g_values[:,0,:]
        grad_g_tp1   = torch.zeros_like(grad_g_t)
        grad_g_tp1[:,:-1,:] = grad_g_values[:, 2:, :]

        grad_u_trans = torch.sum(B[:, :, 0, :] * grad_g_t, dim=-1)     
        grad_u_rot   = torch.sum(B[:, :, 1, :] * grad_g_tp1, dim=-1)
        grad_u = torch.stack([grad_u_trans, grad_u_rot], dim=-1)

        return grad_u
    
    def check_collision(self):

        for obstacle in self.obstacles:
                center = obstacle[:2]  
                radius = obstacle[2].item() 
                dist = torch.norm(self.state[:2]-center)
                if dist < radius+ self.robot_radius:
                    raise RuntimeError("collision")
                
    def compute_plot_data(self):

        optimal_traj = torch.zeros((self.time_step, self.dim_x), device=self.device)
        if self.visualize_optimal_traj:
            optimal_traj = self.predict_trajectory(self.state, self.u_prev.unsqueeze(0)).squeeze(0)

        return optimal_traj, self.opt_clustered_traj, self.clustered_trajs
    
    def move_obstacle(self):
        step_size = 0.016  
        if (self.obstacles[0][0] < 0.5):
            self.obstacles[0][0] += step_size



    
    def compute_control_input(self,noise=None):

        S = self.S
        x0 = self.x0
        u = self.u
        v = self.v
        noise=self.noise
        opt_u = self.opt_u

        x0.copy_(self.state) 
        u.copy_(self.u_prev)
        noise.copy_(self.generateNoiseAndSampling())

        v.copy_(noise + u)
        # v = self.apply_constraint(v)
        trajectory = self.predict_trajectory(x0, v)
        trajectory, v = self.update_control_inputs(trajectory=trajectory, v = v , x0 = x0)
        noise.copy_(v-u)
        S += self.compute_total_cost(trajectory = trajectory, v = v)
        quad_term = torch.sum(u.unsqueeze(0) * self.sigma_inv_diag * v, dim=-1)

        S += self.param_gamma * quad_term.sum(dim=1)  

        if self.execute_dbscan:
            dbscan_start_time = time.perf_counter()
            opt_u.copy_(self.dbscan(v, S, noise, trajectory, u, x0))
            self.dbscan_elapsed_time = time.perf_counter() - dbscan_start_time
        else:
            w = self.compute_weights(S)
            w_expanded = w.view(-1, 1, 1)
            w_epsilon = torch.sum(w_expanded * noise, dim=0)
            opt_u.copy_(u + w_epsilon)



        self.u_prev.copy_(opt_u)

        self.total_iter+=1
        
        return opt_u[0], opt_u, trajectory
    
    
    

    def set_state(self,state):
        self.state = state

    def set_ref(self, ref_path):
        self.ref_path = ref_path





