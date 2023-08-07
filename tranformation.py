from typing import Tuple, List, Iterable
import numpy as np

def apply_transformation(position: Iterable[float], transformation_matrix: np.ndarray) -> np.ndarray:
    """Apply a transformation matrix to a 2D or 3D point.

    Args:
        position (Tuple[float, ...]): 2D or 3D point represented as a tuple of float values.
        transformation_matrix (np.ndarray): 4x4 transformation matrix to be applied.

    Returns:
        np.ndarray: Transformed point.
    """
    # Assert that the position is either 2D or 3D
    assert len(position) in {2, 3}, "Position must be 2D or 3D"

    # Create 4D vector for transformation
    position_4d = (*position, 0.0, 1.0) if len(position) == 2 else (*position, 1.0)

    # Apply the transformation matrix to the 4D position vector.
    transformed_position = np.dot(transformation_matrix, position_4d)

    # Return the transformed position, The output will be a vector of the same dimension as the input.
    return transformed_position[:len(position)]

def create_translation_matrix(x_translation: float, y_translation: float, z_translation: float = 0.0) -> np.ndarray:
    """Creates a 3D translation matrix based on the provided translation values.

    Args:
        x_translation (float): The translation along the x-axis.
        y_translation (float): The translation along the y-axis.
        z_translation (float, optional): The translation along the z-axis. Defaults to 0.0.

    Returns:
        np.ndarray: The resulting translation matrix as a 4x4 NumPy array.
    """
    translation_matrix = np.array([[1.0, 0.0, 0.0, x_translation],
                                   [0.0, 1.0, 0.0, y_translation],
                                   [0.0, 0.0, 1.0, z_translation],
                                   [0.0, 0.0, 0.0, 1.0]])
    return translation_matrix

def create_rotation_matrix(angle: float = 0.0, axis: List[float] = [0.0, 0.0, 1.0]) -> np.ndarray:
    """Creates a 3D rotation matrix based on the provided angle and axis.

    Args:
        angle (float, optional): The rotation angle in radians. Defaults to 0.0.
        axis (List[float], optional): The rotation axis as a 3D vector. Defaults to [0.0, 0.0, 1.0].

    Returns:
        ndarray: The resulting rotation matrix as a 4x4 NumPy array.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)

    ux, uy, uz = axis
    rotation_matrix = np.array([[cos_theta + ux ** 2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta, 0.0],
                                [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy ** 2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta, 0.0],
                                [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz ** 2 * (1 - cos_theta), 0.0],
                                [0.0, 0.0, 0.0, 1.0]])
    return rotation_matrix

def create_scale_matrix(x_scale: float, y_scale: float, z_scale: float = 1.0) -> np.ndarray:
    """Creates a 3D scale matrix based on the provided scaling factors.

    Args:
        x_scale (float): The scaling factor along the x-axis.
        y_scale (float): The scaling factor along the y-axis.
        z_scale (float, optional): The scaling factor along the z-axis. Defaults to 1.0.

    Returns:
        np.ndarray: The resulting scale matrix as a 4x4 NumPy array.
    """
    scale_matrix = np.array([[x_scale, 0.0, 0.0, 0.0],
                             [0.0, y_scale, 0.0, 0.0],
                             [0.0, 0.0, z_scale, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    return scale_matrix
