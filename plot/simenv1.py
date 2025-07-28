import torch
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt

class SIMULATIONENV1():
    def __init__(self, state: torch.Tensor = None, radius = 0.2 ,dt = 0.05,device: str = 'cuda' if torch.cuda.is_available() else 'cpu', ax=None,fig = None ,obstacles_types=None, obstacles = None) -> None:

        self.device = device
        self.state = state if state is not None else torch.tensor([0.0, -2.0, math.pi / 2], device=self.device)
        self.dt = dt
        self.radius = radius
        self.ax = ax
        self.fig =fig
        self.obstacles_types = obstacles_types
        self.obstacles = obstacles
        self.obstacle_direction = 1 
        if self.ax is not None:

            self.robot_circle = Circle((self.state[0].item(), self.state[1].item()), self.radius, color='blue')
            self.ax.add_patch(self.robot_circle)
            
            self.wheel_width = 0.05
            self.wheel_height = 0.02
            self.wheel_offset = 0.12

            self.left_wheel = Rectangle((-self.wheel_width / 2, -self.wheel_height / 2),
                                        self.wheel_width, self.wheel_height, color='black')
            self.right_wheel = Rectangle((-self.wheel_width / 2, -self.wheel_height / 2),
                                         self.wheel_width, self.wheel_height, color='black')

            self.ax.add_patch(self.left_wheel)
            self.ax.add_patch(self.right_wheel)

            self.path_line, = self.ax.plot([], [], 'g-')  
            self.x_data, self.y_data = [], []


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

    def update_plot(self, optimal_traj: torch.Tensor = None, 
                    predicted_traj: torch.Tensor = None, 
                    opt_clustered_traj: dict = None,
                    clustered_trajs: dict = None, ):

        step_size = 0.016  

        if (self.obstacles[0][0] < 0.5):
            self.obstacles[0][0] += step_size

        if self.ax is None:
            return

        for artist in reversed(self.ax.patches + self.ax.lines):
            artist.remove()

        x, y, theta = self.state
        self.robot_circle = Circle((x.item(), y.item()), self.radius, color='black')
        self.ax.add_patch(self.robot_circle)


        line_length = self.radius * 1.0 
        line_x = x + line_length * torch.cos(theta) 
        line_y = y + line_length * torch.sin(theta) 
        self.ax.plot(
            [x.item(), line_x.item()], 
            [y.item(), line_y.item()], 
            color = 'red', linestyle = '-', linewidth=2  
        )

        if self.obstacles is not None and self.obstacles_types is not None:
            for idx, obs in enumerate(self.obstacles):
                obs_x, obs_y, obs_r = obs.cpu().numpy()
                obs_type = self.obstacles_types[idx]

                if obs_type == "circle":
                    obstacle_circle = Circle((obs_x, obs_y), obs_r, color='gray', alpha=0.5)
                    self.ax.add_patch(obstacle_circle)
                    safe_circle = Circle(
                        (obs_x, obs_y), 
                        obs_r + self.radius, 
                        facecolor='none', 
                        edgecolor='gray',  
                        linestyle='dashed' 
                        , linewidth=10.0
                        ,alpha = 0.7
                    )

                    self.ax.add_patch(safe_circle)

            if predicted_traj is not None:
                for batch in predicted_traj:
                    self.ax.plot(
                        batch[:, 0].cpu().numpy(),
                        batch[:, 1].cpu().numpy(),
                        'b-', alpha=0.5, linewidth=10.0
                    )

        if clustered_trajs is not None:
            cluster_colors_traj = plt.cm.get_cmap("Set1", len(clustered_trajs)) 
            added_labels = set() 

            for idx, (label, traj) in enumerate(clustered_trajs.items()):
                color = 'gray' if label == "noise" else cluster_colors_traj(idx) 
                legend_label = f"Sampled trajectory of {label}"

                for path in traj:

                    if legend_label not in added_labels:
                        self.ax.plot(
                            path[:, 0].cpu().numpy(),
                            path[:, 1].cpu().numpy(),
                            linestyle='-', color=color, alpha=0.7, label=legend_label, linewidth=10.0
                        )
                        added_labels.add(legend_label)
                    else:
                        self.ax.plot(
                            path[:, 0].cpu().numpy(),
                            path[:, 1].cpu().numpy(),
                            linestyle='-', color=color, alpha=0.7, linewidth=10.0
                        )
        

        if optimal_traj is not None:
            self.ax.plot(
                optimal_traj[:, 0].cpu().numpy(),
                optimal_traj[:, 1].cpu().numpy(),
                linestyle='-', color='cyan', label="Optimal Trajectory", linewidth=10.0
            )

        if opt_clustered_traj is not None:
            cluster_colors_trajectory = plt.cm.get_cmap("Set2", len(opt_clustered_traj)) 
            for idx, (cluster_label, cluster_traj) in enumerate(opt_clustered_traj.items()):
                color = cluster_colors_trajectory(idx) 
                self.ax.plot(
                    cluster_traj[:, 0].cpu().numpy(),
                    cluster_traj[:, 1].cpu().numpy(),
                    linestyle='--', color=color, label=f"Optimal trajectory of {cluster_label}", linewidth=10.0
                )

        
                
        self.ax.figure.canvas.draw()
