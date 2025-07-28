import torch

class Constraint():
    def __init__(self, robot_radius=0.1, device="cpu", 
                 obstacle_centers=None, obstacle_radii=None, obstacle_types=None):
        self.device = device
        self.robot_radius = robot_radius

        self.obstacle_centers = obstacle_centers.to(self.device) if obstacle_centers is not None else None
        self.obstacle_radii = obstacle_radii.to(self.device) if obstacle_radii is not None else None
        self.obstacle_types = obstacle_types.to(self.device) if obstacle_types is not None else None
        self.combined_radius_sq = (self.obstacle_radii + self.robot_radius) ** 2

    def boundary_eq_batch(self, X: torch.Tensor) -> torch.Tensor:


        diff = X.unsqueeze(0) - self.obstacle_centers.view(-1, 1, 1, 2)
        dist_sq = torch.sum(diff ** 2, dim=-1) 
        g_values = dist_sq - self.combined_radius_sq.view(-1, 1, 1)

        return g_values

    
    def gradient_batch(self, X: torch.Tensor) -> torch.Tensor:

        num_obstacles = self.obstacle_centers.shape[0]
        diff = X.unsqueeze(0) - self.obstacle_centers.view(num_obstacles, 1, 1, 2)  
        norm = torch.norm(diff, dim=-1, keepdim=True) + 1e-8 
        normalized_grad = diff / norm  

        return normalized_grad


    