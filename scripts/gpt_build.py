import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sign(x):
    return np.sign(x)

def superquadric_point_cloud(a1, a2, a3, eps1, eps2, n_theta=100, n_phi=100):
    """
    Generates a superquadric point cloud.

    Parameters:
    - a1, a2, a3: scale factors along x, y, z axes
    - eps1, eps2: shape exponents (0 < eps <= 1 for round, > 1 for squarish)
    - n_theta, n_phi: number of points along each angular direction
    """
    theta = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    def fexp(base, exp):
        return sign(base) * (np.abs(base) ** exp)

    x = a1 * fexp(cos_theta, eps1) * fexp(cos_phi, eps2)
    y = a2 * fexp(cos_theta, eps1) * fexp(sin_phi, eps2)
    z = a3 * fexp(sin_theta, eps1)

    return x, y, z

def display_point_cloud(x, y, z):
    """
    Displays a 3D point cloud using matplotlib.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x.flatten(), y.flatten(), z.flatten(), s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Superquadric Point Cloud')
    plt.show()

# Example usage
a1, a2, a3 = 1.0, 1.0, 1.0
eps1, eps2 = 0.5, 1.0
x, y, z = superquadric_point_cloud(a1, a2, a3, eps1, eps2)
display_point_cloud(x, y, z)
