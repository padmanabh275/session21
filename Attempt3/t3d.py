import torch
import numpy as np

class Transform3D:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def rotation_matrix_z(self, angle):
        """Create a 3D rotation matrix around Z-axis"""
        angle = torch.tensor(angle, dtype=torch.float64, device=self.device)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        return torch.tensor([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float64, device=self.device)
    
    def translation_matrix(self, position):
        """Create a 3D translation matrix"""
        x, y, z = position
        return torch.tensor([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=torch.float64, device=self.device)
    
    def scale_matrix(self, scale):
        """Create a 3D scale matrix"""
        sx, sy, sz = scale
        return torch.tensor([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float64, device=self.device)
    
    def transform_point(self, point, transform_matrix):
        """Transform a single 3D point using a transformation matrix"""
        point = torch.tensor([*point, 1], dtype=torch.float64, device=self.device)
        transformed = torch.matmul(transform_matrix, point)
        return transformed[:3].numpy()
    
    def transform_points(self, points, transform_matrix):
        """Transform multiple 3D points using a transformation matrix"""
        points = torch.tensor(points, dtype=torch.float64, device=self.device)
        ones = torch.ones(points.shape[0], 1, dtype=torch.float64, device=self.device)
        points_homogeneous = torch.cat([points, ones], dim=1)
        transformed = torch.matmul(points_homogeneous, transform_matrix.t())
        return transformed[:, :3].numpy() 