from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import math
class SIMULATIONENV2():
    def __init__(self, state=None, radius=0.2, dt=0.05, ax=None, fig=None, obstacles_types=None, obstacles=None) -> None:


        self.state = state if state is not None else jnp.array([0.0, -2.0, math.pi / 2])
        self.dt = dt
        self.radius = radius
        self.ax = ax
        self.fig = fig
        self.obstacles_types = obstacles_types
        self.obstacles = obstacles
        self.obstacle_direction = 1

        if self.ax is not None:

            self.robot_circle = Circle((float(self.state[0]), float(self.state[1])), self.radius, color='blue')
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
            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.0)
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

    def update_plot(self,
                optimal_traj: jnp.ndarray = None,
                predicted_traj: jnp.ndarray = None,
                best_trajectory: jnp.ndarray = None,
                opt_clustered_traj: dict = None,
                before_optimal_traj: jnp.ndarray = None,
                clustered_trajs: dict = None,
                bound_traj: dict = None,
                before_filter_traj: dict = None):


        if self.ax is None:
            return

        for artist in reversed(self.ax.patches + self.ax.lines):
            artist.remove()


        x, y, theta = self.state
        self.robot_circle = Circle((float(x), float(y)), self.radius, color='black')
        self.ax.add_patch(self.robot_circle)

        line_length = self.radius * 1.0
        line_x = x + line_length * jnp.cos(theta)
        line_y = y + line_length * jnp.sin(theta)
        self.ax.plot(
            [float(x), float(line_x)],
            [float(y), float(line_y)],
            color='red', linestyle='-', linewidth=2
        )

  
        if self.obstacles is not None and self.obstacles_types is not None:
            for idx, obs in enumerate(self.obstacles):
                obs_x, obs_y, obs_r = np.array(obs)
                obs_type = self.obstacles_types[idx]

                if obs_type == "circle":
                    obstacle_circle = Circle((obs_x, obs_y), obs_r, color='gray', alpha=0.5)
                    self.ax.add_patch(obstacle_circle)

                    safe_circle = Circle(
                        (obs_x, obs_y),
                        obs_r + self.radius,
                        facecolor='none',
                        edgecolor='gray',
                        linestyle='dashed',
                        linewidth=10.0,
                        alpha=0.7
                    )
                    self.ax.add_patch(safe_circle)

        if predicted_traj is not None:
            for batch in predicted_traj:
                self.ax.plot(
                    np.array(batch[:, 0]),
                    np.array(batch[:, 1]),
                    'b-', alpha=0.5, linewidth=10.0
                )

        

        if clustered_trajs is not None:
            cluster_colors_traj = plt.cm.get_cmap("Set1", len(clustered_trajs))
            added_labels = set()

            for idx, (label, traj) in enumerate(clustered_trajs.items()):
                color = 'gray' if label == "noise" else cluster_colors_traj(idx)
                legend_label = f"Sampled trajectory of {label}"

                for path in traj:
                    x_vals = np.array(path[:, 0])
                    y_vals = np.array(path[:, 1])

                    if legend_label not in added_labels:
                        self.ax.plot(
                            x_vals, y_vals,
                            linestyle='-', color=color, alpha=0.7,
                            label=legend_label, linewidth=10.0
                        )
                        added_labels.add(legend_label)
                    else:
                        self.ax.plot(
                            x_vals, y_vals,
                            linestyle='-', color=color, alpha=0.7, linewidth=10.0
                        )

        if before_optimal_traj is not None:
            self.ax.plot(
                np.array(before_optimal_traj[:, 0]),
                np.array(before_optimal_traj[:, 1]),
                'g-', label="Before Optimal Trajectory", alpha=0.7, linewidth=10.0
            )

        if optimal_traj is not None:
            self.ax.plot(
                np.array(optimal_traj[:, 0]),
                np.array(optimal_traj[:, 1]),
                linestyle='-', color='cyan', label="Optimal Trajectory", linewidth=10.0
            )

        if before_filter_traj is not None:
            self.ax.plot(
                np.array(before_filter_traj[:, 0]),
                np.array(before_filter_traj[:, 1]),
                linestyle='--', color='black', label="before_filter Trajectory", linewidth=10.0
            )
 
            
        if best_trajectory is not None:
            self.ax.plot(
                np.array(best_trajectory[:, 0]),
                np.array(best_trajectory[:, 1]),
                linestyle='--', color='gray', label="Best Trajectory"
            )


        if opt_clustered_traj is not None:
            cluster_colors_trajectory = plt.cm.get_cmap("Set2", len(opt_clustered_traj))
            for idx, (cluster_label, cluster_traj) in enumerate(opt_clustered_traj.items()):
                color = cluster_colors_trajectory(idx)
                self.ax.plot(
                    np.array(cluster_traj[:, 0]),
                    np.array(cluster_traj[:, 1]),
                    linestyle='--', color=color, label=f"Optimal trajectory of {cluster_label}", linewidth=10.0
                )

        self.ax.figure.canvas.draw()