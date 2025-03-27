from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage
from kivy.vector import Vector
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.uix.widget import Widget
import t3d
import numpy as np
from math import cos, sin, radians

class Car(Image):
    """Car class with enhanced 3D transformations and continuous action support"""
    
    angle = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)
    boundary_awareness = NumericProperty(0)
    steering_angle = NumericProperty(0)  # New property for continuous steering
    acceleration = NumericProperty(0)    # New property for continuous acceleration
    current_speed = NumericProperty(0)   # Current speed of the car

    def __init__(self, **kwargs):
        super(Car, self).__init__(**kwargs)
        self.transform3d = t3d.Transform3D()  # Initialize Transform3D from t3d module
        self.position = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self.scale = [1, 1, 1]
        self.sensor_range = 20
        self.sensor_angles = [-30, 0, 30]
        self.sensor_values = [0, 0, 0]
        self.last_position = None
        self.stuck_counter = 0
        self.max_stuck_steps = 10
        self.max_steering_angle = 45  # Maximum steering angle in degrees
        self.max_acceleration = 5.0    # Maximum acceleration
        self.min_acceleration = -2.0   # Maximum deceleration

    def move(self, action):
        """Move the car with continuous actions"""
        # Unpack the action (steering and acceleration)
        steering, acceleration = action
        
        # Update steering angle (clamped between -max_steering_angle and max_steering_angle)
        self.steering_angle = np.clip(steering * self.max_steering_angle, 
                                    -self.max_steering_angle, 
                                    self.max_steering_angle)
        
        # Update acceleration (clamped between min_acceleration and max_acceleration)
        self.acceleration = np.clip(acceleration * self.max_acceleration,
                                  self.min_acceleration,
                                  self.max_acceleration)
        
        # Update current speed based on acceleration
        self.current_speed = np.clip(self.current_speed + self.acceleration,
                                   0.0,
                                   self.max_acceleration)
        
        # Update angle based on steering
        self.angle += self.steering_angle * (self.current_speed / self.max_acceleration)
        
        # Update 3D rotation
        self.rotation[2] = self.angle
        
        # Create transformation matrices
        rotation_matrix = self.transform3d.rotation_matrix_z(self.angle)
        translation_matrix = self.transform3d.translation_matrix(self.position)
        scale_matrix = self.transform3d.scale_matrix(self.scale)
        
        # Combine transformations
        transform_matrix = torch.matmul(translation_matrix, 
                                      torch.matmul(rotation_matrix, scale_matrix))
        
        # Update position based on current speed and angle
        self.position[0] += self.current_speed * cos(radians(self.angle))
        self.position[1] += self.current_speed * sin(radians(self.angle))
        
        # Update 2D position for Kivy
        self.pos = self.position[:2]
        
        # Update sensors
        self.update_sensors()

        # Check if car is stuck
        if self.last_position:
            distance_moved = Vector(self.pos).distance(self.last_position)
            if distance_moved < 0.1:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        self.last_position = Vector(self.pos)

        # Calculate boundary awareness
        self.calculate_boundary_awareness()

    def get_state(self):
        """Get the current state of the car for the TD3 algorithm"""
        state = np.array([
            self.signal1, self.signal2, self.signal3,  # Sensor values
            self.steering_angle / self.max_steering_angle,  # Normalized steering
            self.current_speed / self.max_acceleration,     # Normalized speed
            self.boundary_awareness,                        # Boundary awareness
            self.angle / 360.0,                            # Normalized angle
            self.velocity_x / self.max_acceleration,       # Normalized velocity x
            self.velocity_y / self.max_acceleration        # Normalized velocity y
        ])
        return state

    def update_sensors(self):
        # Update sensor positions based on car's position and angle
        for i, angle in enumerate(self.sensor_angles):
            sensor_angle = self.angle + angle
            sensor_x = self.center_x + self.sensor_range * cos(radians(sensor_angle))
            sensor_y = self.center_y + self.sensor_range * sin(radians(sensor_angle))
            
            # Update sensor position
            if i == 0:
                self.sensor1 = (sensor_x, sensor_y)
            elif i == 1:
                self.sensor2 = (sensor_x, sensor_y)
            else:
                self.sensor3 = (sensor_x, sensor_y)

    def calculate_boundary_awareness(self):
        # Calculate how close the car is to boundaries
        margin = 20  # Margin from boundaries
        x_awareness = min(self.x / margin, (self.parent.width - self.x) / margin)
        y_awareness = min(self.y / margin, (self.parent.height - self.y) / margin)
        self.boundary_awareness = min(x_awareness, y_awareness)

    def get_sensor_values(self, sand):
        # Get sensor values with improved accuracy
        for i, sensor in enumerate([self.sensor1, self.sensor2, self.sensor3]):
            x = int(sensor[0])
            y = int(sensor[1])
            if 0 <= x < sand.shape[0] and 0 <= y < sand.shape[1]:
                self.sensor_values[i] = sand[x, y]
            else:
                self.sensor_values[i] = 1  # Treat out of bounds as obstacle

        # Normalize sensor values
        self.signal1 = self.sensor_values[0] / 255.0
        self.signal2 = self.sensor_values[1] / 255.0
        self.signal3 = self.sensor_values[2] / 255.0

    def is_stuck(self):
        return self.stuck_counter >= self.max_stuck_steps 

    def reset(self, pos):
        """Reset car position and state"""
        self.position = [pos[0], pos[1], 0]
        self.rotation = [0, 0, 0]
        self.scale = [1, 1, 1]
        self.velocity = Vector(0, 0)
        self.pos = pos
        self.angle = 0
        self.stuck_counter = 0
        self.last_position = None
        self.steering_angle = 0
        self.acceleration = 0
        self.current_speed = 0 