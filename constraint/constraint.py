import jax.numpy as jnp

class Constraint:
    def __init__(self, robot_radius: float = 0.1):
        self.robot_radius = robot_radius

    def boundary_eq_batch(
        self,
        X: jnp.ndarray,              # shape: (batch_size, time_step, 2)
        centers: jnp.ndarray,        # shape: (num_obstacles, 2)
        radii: jnp.ndarray           # shape: (num_obstacles,)
    ) -> jnp.ndarray:
       
        # expand dims for batch and time
        X_exp = jnp.expand_dims(X, axis=0)                   # (1, B, T, 2)
        centers_exp = jnp.expand_dims(centers, axis=(1, 2)) # (obs, 1, 1, 2)

        # squared distance: (obs, B, T)
        diff = X_exp - centers_exp                          # (obs, B, T, 2)
        dist_sq = jnp.sum(diff ** 2, axis=-1)                # (obs, B, T)

        combined_radius_sq = (radii + self.robot_radius) ** 2  # (obs,)
        g_values = dist_sq - jnp.expand_dims(combined_radius_sq, axis=(1, 2))
        return g_values

    def gradient_batch(
        self,
        X: jnp.ndarray,              # shape: (batch_size, time_step, 2)
        centers: jnp.ndarray,        # shape: (num_obstacles, 2)
        radii: jnp.ndarray           # shape: (num_obstacles,)
    ) -> jnp.ndarray:

        # expand dims
        centers_exp = jnp.expand_dims(centers, axis=(1, 2))  # (obs, 1, 1, 2)
        X_exp = jnp.expand_dims(X, axis=0)                   # (1, B, T, 2)

        diff = X_exp - centers_exp                           # (obs, B, T, 2)
        norm = jnp.linalg.norm(diff, axis=-1, keepdims=True) + 1e-8  # (obs, B, T, 1)
        grads = diff / norm                                  # (obs, B, T, 2)
        return grads
