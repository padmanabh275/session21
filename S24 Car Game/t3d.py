import torch
import numpy as np
import math

class Transform3D:
    def __init__(self):
        self.position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.rotation = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        
        # Move tensors to CPU by default
        self.position = self.position.cpu()
        self.rotation = self.rotation.cpu()
        self.scale = self.scale.cpu()

    def rotation_matrix_z(self, angle):
        """Create a rotation matrix around Z axis"""
        if torch.is_tensor(angle):
            angle = angle.cpu()
        angle = float(angle)
        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))
        return torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32).cpu()
    
    def translation_matrix(self, position):
        """Create a translation matrix"""
        if torch.is_tensor(position):
            position = position.cpu()
        self.position = torch.tensor(position, dtype=torch.float32).cpu()
        return torch.tensor([
            [1, 0, 0, position[0]],
            [0, 1, 0, position[1]],
            [0, 0, 1, position[2]],
            [0, 0, 0, 1]
        ], dtype=torch.float32).cpu()
    
    def scale_matrix(self, scale):
        """Create a scale matrix"""
        if torch.is_tensor(scale):
            scale = scale.cpu()
        self.scale = torch.tensor(scale, dtype=torch.float32).cpu()
        return torch.tensor([
            [scale[0], 0, 0, 0],
            [0, scale[1], 0, 0],
            [0, 0, scale[2], 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32).cpu()
    
    def transform_point(self, point, transform_matrix):
        """Transform a point using a transformation matrix"""
        if torch.is_tensor(point):
            point = point.cpu()
        if torch.is_tensor(transform_matrix):
            transform_matrix = transform_matrix.cpu()
            
        # Convert point to homogeneous coordinates
        point_homogeneous = torch.tensor([point[0], point[1], point[2], 1.0], dtype=torch.float32).cpu()
        
        # Apply transformation
        transformed = torch.matmul(transform_matrix, point_homogeneous)
        
        # Convert back to 3D coordinates
        return transformed[:3].cpu()
    
    def transform_points(self, points, transform_matrix):
        """Transform multiple points using a transformation matrix"""
        if torch.is_tensor(points):
            points = points.cpu()
        if torch.is_tensor(transform_matrix):
            transform_matrix = transform_matrix.cpu()
            
        # Convert points to homogeneous coordinates
        points_homogeneous = torch.cat([points, torch.ones(points.shape[0], 1)], dim=1).cpu()
        
        # Apply transformation
        transformed = torch.matmul(transform_matrix, points_homogeneous.t()).t()
        
        # Convert back to 3D coordinates
        return transformed[:, :3].cpu()

    def get_position(self):
        """Get current position"""
        return self.position.cpu()

    def get_rotation(self):
        """Get current rotation"""
        return self.rotation.cpu()

    def get_scale(self):
        """Get current scale"""
        return self.scale.cpu() 