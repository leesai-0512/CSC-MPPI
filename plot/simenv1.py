from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import math

class SIMULATIONENV1():
    def __init__(
        self,
        state: jnp.ndarray = None,
        radius: float = 0.2,
        dt: float = 0.05,
        ax=None,
        fig=None,
        obstacles_types=None,
        obstacles=None
    ):

        self.state = state if state is not None else jnp.array([0.0, -2.0, math.pi / 2])
        self.dt = dt
        self.radius = radius
        self.ax = ax
        self.fig = fig
        self.obstacles_types = obstacles_types
        self.obstacles = obstacles
        self.obstacle_direction = 1


        if self.ax is not None:
            self.robot_circle = Circle(
                (float(self.state[0]), float(self.state[1])),
                self.radius,
                color='blue'
            )
            self.ax.add_patch(self.robot_circle)


            self.wheel_width = 0.05
            self.wheel_height = 0.02
            self.wheel_offset = 0.12

            self.left_wheel = Rectangle(
                (-self.wheel_width / 2, -self.wheel_height / 2),
                self.wheel_width, self.wheel_height,
                color='black'
            )
            self.right_wheel = Rectangle(
                (-self.wheel_width / 2, -self.wheel_height / 2),
                self.wheel_width, self.wheel_height,
                color='black'
            )

            self.ax.add_patch(self.left_wheel)
            self.ax.add_patch(self.right_wheel)

            self.path_line, = self.ax.plot([], [], 'g-')  
            self.x_data, self.y_data = [], []


        if self.fig is not None:
            fig.subplots_adjust(bottom=0.15, left=0.12)
        if self.ax is not None:
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


    def update_plot(self,
                    optimal_traj=None,
                    predicted_traj=None,
                    best_trajectory=None,
                    opt_clustered_traj=None,
                    before_optimal_traj=None,
                    clustered_trajs=None,
                    bound_traj=None,
                    before_filter_traj=None):
        step_size = 0.016
        if self.obstacles[0][0] < 0.5:
            self.obstacles = self.obstacles.at[0, 0].set(self.obstacles[0][0] + step_size)

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
                obs_x, obs_y, obs_r = map(float, obs)
                obs_type = self.obstacles_types[idx]
                if obs_type == "circle":
                    obstacle_circle = Circle((obs_x, obs_y), obs_r, color='gray', alpha=0.5)
                    self.ax.add_patch(obstacle_circle)
                    safe_circle = Circle((obs_x, obs_y), obs_r + self.radius, facecolor='none',
                                        edgecolor='gray', linestyle='dashed', linewidth=10.0, alpha=0.7)
                    self.ax.add_patch(safe_circle)


        def plot_traj(traj, style='-', color='cyan', label=None, linewidth=10.0):
            self.ax.plot(np.array(traj[:, 0]), np.array(traj[:, 1]),
                        linestyle=style, color=color, label=label, linewidth=linewidth)

        if predicted_traj is not None:
            for batch in predicted_traj:
                self.ax.plot(np.array(batch[:, 0]), np.array(batch[:, 1]),
                            'b-', alpha=0.5, linewidth=10.0)

        if clustered_trajs is not None:
            cmap = plt.cm.get_cmap("Set1", len(clustered_trajs))
            added_labels = set()
            for idx, (label, traj_list) in enumerate(clustered_trajs.items()):
                color = 'gray' if label == "noise" else cmap(idx)
                label_str = f"Sampled trajectory of {label}"
                for path in traj_list:
                    if label_str not in added_labels:
                        plot_traj(path, '-', color, label_str)
                        added_labels.add(label_str)
                    else:
                        plot_traj(path, '-', color)

        if before_optimal_traj is not None:
            plot_traj(before_optimal_traj, 'g-', label="Before Optimal Trajectory")

        if optimal_traj is not None:
            plot_traj(optimal_traj, '-', 'cyan', label="Optimal Trajectory")

        if before_filter_traj is not None:
            plot_traj(before_filter_traj, '--', 'black', label="before_filter Trajectory")

        if best_trajectory is not None:
            plot_traj(best_trajectory, '--', 'gray', label="Best Trajectory")

        if opt_clustered_traj is not None:
            cmap = plt.cm.get_cmap("Set2", len(opt_clustered_traj))
            for idx, (label, traj) in enumerate(opt_clustered_traj.items()):
                plot_traj(traj, '--', cmap(idx), f"Optimal trajectory of {label}")

        self.ax.figure.canvas.draw()

