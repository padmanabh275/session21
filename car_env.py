import numpy as np
import pygame
import random
from gym import Env
from gym.spaces import Box

class CarEnv(Env):
    def __init__(self, width=800, height=600):
        super(CarEnv, self).__init__()
        
        # Define action and observation spaces
        self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # Initialize Pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Car Simulation")
        
        # Car parameters
        self.car_width = 40
        self.car_height = 20
        self.car_pos = [width//2, height//2]
        self.car_angle = 0
        self.car_speed = 0
        self.max_speed = 5
        
        # Road parameters
        self.road_width = 100
        self.road_points = self.generate_road()
        
        # Initialize clock
        self.clock = pygame.time.Clock()
        
    def generate_road(self):
        points = []
        x = 0
        y = self.height // 2
        while x < self.width:
            y += random.randint(-20, 20)
            y = max(self.road_width, min(self.height - self.road_width, y))
            points.append((x, y))
            x += 100
        return points
    
    def reset(self):
        self.car_pos = [self.width//2, self.height//2]
        self.car_angle = 0
        self.car_speed = 0
        self.road_points = self.generate_road()
        return self._get_observation()
    
    def _get_observation(self):
        # Get the next road point
        next_point = None
        for point in self.road_points:
            if point[0] > self.car_pos[0]:
                next_point = point
                break
        
        if next_point is None:
            next_point = self.road_points[-1]
        
        # Calculate relative position and angle
        dx = next_point[0] - self.car_pos[0]
        dy = next_point[1] - self.car_pos[1]
        target_angle = np.arctan2(dy, dx)
        angle_diff = target_angle - self.car_angle
        
        return np.array([
            self.car_pos[0] / self.width,
            self.car_pos[1] / self.height,
            np.sin(self.car_angle),
            np.cos(self.car_angle),
            self.car_speed / self.max_speed,
            angle_diff / np.pi
        ])
    
    def step(self, action):
        # Update car position and angle
        steering, acceleration = action
        self.car_angle += steering * 0.1
        self.car_speed += acceleration * 0.5
        self.car_speed = np.clip(self.car_speed, 0, self.max_speed)
        
        # Update position
        self.car_pos[0] += np.cos(self.car_angle) * self.car_speed
        self.car_pos[1] += np.sin(self.car_angle) * self.car_speed
        
        # Check if car is off the road
        reward = -1.0
        done = False
        
        # Check if car is within road bounds
        road_y = np.interp(self.car_pos[0], 
                          [p[0] for p in self.road_points],
                          [p[1] for p in self.road_points])
        
        if abs(self.car_pos[1] - road_y) > self.road_width:
            done = True
            reward = -100.0
        else:
            reward = 1.0 - abs(self.car_pos[1] - road_y) / self.road_width
        
        # Check if car reached the end
        if self.car_pos[0] >= self.width:
            done = True
            reward = 100.0
        
        return self._get_observation(), reward, done, {}
    
    def render(self):
        self.screen.fill((0, 0, 0))
        
        # Draw road
        pygame.draw.lines(self.screen, (100, 100, 100), False, self.road_points, self.road_width)
        
        # Draw car
        car_surface = pygame.Surface((self.car_width, self.car_height), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, (255, 0, 0), (0, 0, self.car_width, self.car_height))
        
        # Rotate car
        rotated_car = pygame.transform.rotate(car_surface, np.degrees(self.car_angle))
        car_rect = rotated_car.get_rect(center=self.car_pos)
        
        self.screen.blit(rotated_car, car_rect)
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        pygame.quit() 