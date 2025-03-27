# Self Driving Car

# Importing the libraries
import numpy as np
from random import random
import random as rand  # Import random module with different name
import matplotlib.pyplot as plt
import time
import logging
import torch
from math import cos, sin, radians
import os
import glob
import math

# Configure matplotlib to use a specific font and suppress debug messages
plt.rcParams['font.family'] = 'DejaVu Sans'
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line, PushMatrix, Translate, Rotate, Rectangle, PopMatrix
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Rectangle
from kivy.uix.gridlayout import GridLayout

# Importing the TD3 object from our AI in ai.py
from ai import TD3

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')
Config.set('graphics', 'position', 'auto')  # Add position setting
Config.set('graphics', 'borderless', '0')   # Disable borderless mode
Config.set('graphics', 'fullscreen', '0')   # Disable fullscreen

# Set default font for Kivy
from kivy.resources import resource_add_path
resource_add_path('C:/Windows/Fonts')
Config.set('kivy', 'default_font', ['DejaVuSans'])

# Global variables for painting
longueur = 1429  # Window width
largeur = 660    # Window height
last_x = 0
last_y = 0
n_points = 0
length = 0
sand = np.zeros((longueur, largeur))

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = TD3(9, 2, 1.0)
action2rotation = [0,5,-5]
last_reward = 0
scores = []
best_accuracy = 0
epoch_count = 0
MAX_EPOCHS = 2  # Changed from 50 to 2

# Load the ideal path mask
ideal_path = CoreImage("./images/MASK1.png")

# Initializing the map
first_update = True
last_distance = float('inf')  # Initialize last_distance with infinity

def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global swap
    global last_distance
    
    # Initialize sand array with window dimensions
    sand = np.zeros((longueur, largeur))
    
    # Load and process the mask image
    img = PILImage.open("./images/MASK1.png").convert('L')
    img = img.resize((longueur, largeur), PILImage.Resampling.LANCZOS)  # Use better resampling
    sand = np.asarray(img)/255
    
    # Load citymap image
    citymap = PILImage.open("./images/citymap.png")
    citymap = citymap.resize((longueur, largeur), PILImage.Resampling.LANCZOS)  # Use better resampling
    citymap = citymap.convert('RGB')  # Convert to RGB to allow colored drawing
    
    # Create a copy of the mask image for drawing
    mask_img = img.copy()
    mask_img = mask_img.convert('RGB')  # Convert to RGB to allow colored drawing
    
    # Define target points with their coordinates
    targets = {
        1: {'x': 42, 'y': 38, 'label': 'A1'},
        2: {'x': 56, 'y': 44, 'label': 'A2'},
        3: {'x': 49, 'y': 26, 'label': 'A3'}
    }
    
    # Draw target points on both images
    for target in targets.values():
        x, y = target['x'] * 20, target['y'] * 20  # Scale coordinates
        
        # Draw outer red circle
        for dx in range(-35, 36):
            for dy in range(-35, 36):
                if dx*dx + dy*dy <= 1225:  # Circle with radius 35
                    px, py = x + dx, y + dy
                    if 0 <= px < longueur and 0 <= py < largeur:
                        # Draw on citymap
                        citymap.putpixel((px, py), (255, 0, 0))  # Red color
                        # Draw on mask
                        mask_img.putpixel((px, py), (255, 0, 0))  # Red color
        
        # Draw middle white circle
        for dx in range(-30, 31):
            for dy in range(-30, 31):
                if dx*dx + dy*dy <= 900:  # Circle with radius 30
                    px, py = x + dx, y + dy
                    if 0 <= px < longueur and 0 <= py < largeur:
                        # Draw on citymap
                        citymap.putpixel((px, py), (255, 255, 255))  # White color
                        # Draw on mask
                        mask_img.putpixel((px, py), (255, 255, 255))  # White color
        
        # Draw inner red circle
        for dx in range(-25, 26):
            for dy in range(-25, 26):
                if dx*dx + dy*dy <= 625:  # Circle with radius 25
                    px, py = x + dx, y + dy
                    if 0 <= px < longueur and 0 <= py < largeur:
                        # Draw on citymap
                        citymap.putpixel((px, py), (255, 0, 0))  # Red color
                        # Draw on mask
                        mask_img.putpixel((px, py), (255, 0, 0))  # Red color
    
    # Save both modified images with high quality
    citymap.save("./images/citymap.png", quality=95)
    mask_img.save("./images/MASK1.png", quality=95)
    
    # Ensure goal coordinates are within bounds
    goal_x = min(1420, longueur - 1)  # Start with Target A1, but ensure it's within bounds
    goal_y = min(622, largeur - 1)
    
    first_update = False
    swap = 0  # Initialize swap to 0 for first target
    last_distance = float('inf')  # Reset last_distance when initializing

# Creating the car class

class Car(Widget):
    """Car class with enhanced movement and sensor capabilities"""
    
    # Kivy properties for position and movement
    angle = NumericProperty(0)
    current_speed = NumericProperty(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size = (20, 10)  # Smaller car size
        self.pos = (0, 0)
        
        # Movement parameters - adjusted for better control
        self.max_speed = 0.2        # Reduced for stability
        self.min_speed = 0.01       # Reduced minimum speed
        self.max_acceleration = 0.02  # Reduced for smoother acceleration
        self.min_acceleration = -0.01  # Reduced for smoother deceleration
        self.max_steering_angle = 2.0  # Reduced for more precise steering
        self.speed_decay = 0.995    # Slower decay for better stability
        
        # Current state
        self.current_speed = self.min_speed
        self.acceleration = 0.005    # Start with very small positive acceleration
        self.steering_angle = 0.0
        
        # Sensor parameters
        self.sensor_length = 30      # Reduced for better stability
        self.sensor_angles = [-30, -15, 0, 15, 30]
        self.sensor_values = [0.0] * len(self.sensor_angles)
        
        # Initialize sensors list and other sensor-related attributes
        self.sensors = []  # List to store sensor lines
        self.sensor_labels = []  # List to store sensor labels
        self.sensor1 = (0, 0)  # Initialize sensor positions
        self.sensor2 = (0, 0)
        self.sensor3 = (0, 0)
        self.signal1 = 0  # Initialize sensor signals
        self.signal2 = 0
        self.signal3 = 0
        
        # State tracking
        self.last_position = self.pos
        self.stuck_counter = 0
        self.max_stuck_steps = 30    # Reduced for faster recovery
        
        # Initialize sensors
        self.init_sensors()
        
        # Draw the car
        with self.canvas:
            Color(0.8, 0.2, 0.2)  # Red color
            Rectangle(pos=self.pos, size=self.size)
            
        # Schedule updates
        Clock.schedule_interval(self.update_graphics, 1.0/60.0)

    def update_graphics(self, dt):
        """Update the car's visual representation"""
        try:
            self.canvas.clear()
            with self.canvas:
                # Save the current transformation matrix
                PushMatrix()
                
                # Translate to car position
                Translate(self.pos[0], self.pos[1])
                
                # Rotate around car center
                Rotate(angle=self.angle, origin=self.center)
                
                # Draw car body
                Color(1, 0, 0)  # Red color
                Rectangle(pos=(-self.width/2, -self.height/2), size=(self.width, self.height))
                
                # Restore the transformation matrix
                PopMatrix()
                
                # Draw sensor lines
                Color(0, 1, 0)  # Green color for sensors
                for sensor in self.sensors:
                    if hasattr(sensor, 'points'):
                        Line(points=sensor.points, width=1)

        except Exception as e:
            print(f"Error in update_graphics: {str(e)}")

    def init_sensors(self):
        """Initialize sensor lines and labels"""
        # Clear existing sensors
        self.sensors.clear()
        self.sensor_labels.clear()
        
        # Create new sensor lines and labels
        for i, angle in enumerate(self.sensor_angles):
            # Create sensor line with initial points
            line = Line(points=[0, 0, 0, 0], width=1)
            self.sensors.append(line)
            
            # Create sensor label with safe initial position
            label = Label(
                text="0.00",
                pos=(0, 0),  # Safe initial position
                size=(30, 20),
                color=(1, 0, 0, 1)  # Red color
            )
            self.sensor_labels.append(label)
            
            # Add label to widget
            self.add_widget(self.sensor_labels[-1])

    def update_sensors(self):
        """Update sensor positions and values"""
        try:
            # Get car's center position
            center_x = self.pos[0] + self.width/2
            center_y = self.pos[1] + self.height/2
            
            # Update each sensor
            for i, angle_offset in enumerate(self.sensor_angles):
                # Calculate sensor angle in radians
                sensor_angle = math.radians(self.angle + angle_offset)
                
                # Calculate sensor end point
                end_x = center_x + self.sensor_length * math.cos(sensor_angle)
                end_y = center_y + self.sensor_length * math.sin(sensor_angle)
                
                # Update sensor line points
                self.sensors[i].points = [center_x, center_y, end_x, end_y]
                
                # Update sensor label position with bounds checking
                label_offset = 10
                label_x = max(0, min(end_x + label_offset * math.cos(sensor_angle), Window.width - 30))
                label_y = max(0, min(end_y + label_offset * math.sin(sensor_angle), Window.height - 20))
                
                # Only update label position if coordinates are valid
                if not (math.isnan(label_x) or math.isnan(label_y)):
                    self.sensor_labels[i].pos = (label_x, label_y)
                
                # Store sensor positions for the first three sensors
                if i == 0:
                    self.sensor1 = (end_x, end_y)
                elif i == 1:
                    self.sensor2 = (end_x, end_y)
                elif i == 2:
                    self.sensor3 = (end_x, end_y)
            
            # Update sensor values if parent exists
            if self.parent and hasattr(self.parent, 'sand'):
                self.get_sensor_values(self.parent.sand)
                
        except Exception as e:
            print(f"Error updating sensors: {str(e)}")

    def get_sensor_values(self, sand):
        """Get sensor values from the sand array with improved error handling"""
        try:
            # Validate input sand array
            if not isinstance(sand, np.ndarray):
                print("Invalid sand array")
                return [1.0] * len(self.sensor_angles)
            
            # Get car's center position
            center_x = self.pos[0] + self.width/2
            center_y = self.pos[1] + self.height/2
            
            # Validate position
            if math.isnan(center_x) or math.isnan(center_y):
                print("Invalid car position")
                return [1.0] * len(self.sensor_angles)
            
            # Update sensor positions and values
            for i, angle_offset in enumerate(self.sensor_angles):
                # Calculate sensor angle in radians
                sensor_angle = math.radians(self.angle + angle_offset)
                
                # Calculate sensor end point
                end_x = center_x + self.sensor_length * math.cos(sensor_angle)
                end_y = center_y + self.sensor_length * math.sin(sensor_angle)
                
                # Validate sensor end point
                if math.isnan(end_x) or math.isnan(end_y):
                    self.sensor_values[i] = 1.0
                    continue
                
                # Convert to int coordinates with bounds checking
                x = max(0, min(int(end_x), sand.shape[0]-1))
                y = max(0, min(int(end_y), sand.shape[1]-1))
                
                # Get sensor value
                self.sensor_values[i] = float(sand[x, y])
                
                # Update label if valid
                if i < len(self.sensor_labels):
                    self.sensor_labels[i].text = f"{self.sensor_values[i]:.2f}"
                
                # Store sensor positions for the first three sensors
                if i == 0:
                    self.sensor1 = (end_x, end_y)
                elif i == 1:
                    self.sensor2 = (end_x, end_y)
                elif i == 2:
                    self.sensor3 = (end_x, end_y)
            
            return self.sensor_values
            
        except Exception as e:
            print(f"Error in get_sensor_values: {str(e)}")
            return [1.0] * len(self.sensor_angles)

    def calculate_boundary_awareness(self):
        # Implementation of calculate_boundary_awareness method
        pass

    def is_stuck(self):
        """Check if car is stuck"""
        if self.last_position:
            current_pos = Vector(self.pos)
            distance = current_pos.distance(Vector(self.last_position))
            if distance < 1.0:  # If moved less than 1 pixel
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            self.last_position = self.pos
        return self.stuck_counter >= self.max_stuck_steps

    def reset(self, pos):
        """Reset car to initial state"""
        self.pos = pos
        self.angle = 0
        self.current_speed = self.min_speed
        self.acceleration = 0.005
        self.steering_angle = 0
        self.stuck_counter = 0
        self.last_position = pos
        self.update_graphics(None)

    def is_in_loop(self):
        # Implementation of is_in_loop method
        pass

    def move(self, action):
        """Move the car based on the action with improved numerical stability"""
        try:
            # Validate input action and handle NaN values
            if not isinstance(action, (list, np.ndarray)) or len(action) != 2 or np.any(np.isnan(action)):
                print("Invalid action format or NaN values detected")
                return
            
            # Extract and validate action components
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            steering = float(np.clip(action[0], -1.0, 1.0))
            acceleration = float(np.clip(action[1], -1.0, 1.0))
            
            # Validate steering and acceleration for NaN
            if math.isnan(steering) or math.isnan(acceleration):
                print("NaN values in steering or acceleration")
                return
            
            # Update steering angle with bounds and validation
            new_steering_angle = float(np.clip(steering * self.max_steering_angle, 
                                           -self.max_steering_angle, 
                                           self.max_steering_angle))
            if not math.isnan(new_steering_angle):
                self.steering_angle = new_steering_angle
            
            # Update acceleration with bounds and validation
            new_acceleration = float(np.clip(acceleration * self.max_acceleration,
                                         self.min_acceleration,
                                         self.max_acceleration))
            if not math.isnan(new_acceleration):
                self.acceleration = new_acceleration
            
            # Update current speed with bounds, decay, and validation
            new_speed = float(self.current_speed + self.acceleration)
            new_speed = float(np.clip(new_speed * self.speed_decay,
                                   self.min_speed,
                                   self.max_speed))
            
            # Validate new speed
            if math.isnan(new_speed):
                print("Invalid speed calculated")
                new_speed = self.min_speed
            
            # Limit speed change per frame with validation
            max_speed_change = 0.005
            speed_diff = new_speed - self.current_speed
            if not math.isnan(speed_diff):
                if abs(speed_diff) > max_speed_change:
                    new_speed = self.current_speed + max_speed_change * np.sign(speed_diff)
            
            self.current_speed = float(new_speed)
            
            # Update angle based on steering with validation
            new_angle = float((self.angle + self.steering_angle) % 360.0)
            if not math.isnan(new_angle):
                self.angle = new_angle
            
            # Calculate movement vector with validation
            angle_rad = math.radians(self.angle)
            if not math.isnan(angle_rad):
                dx = float(self.current_speed * math.cos(angle_rad))
                dy = float(self.current_speed * math.sin(angle_rad))
                
                # Validate movement vector
                if not (math.isnan(dx) or math.isnan(dy)):
                    # Update position with bounds checking
                    new_x = float(np.clip(self.pos[0] + dx, 0, Window.width - self.width))
                    new_y = float(np.clip(self.pos[1] + dy, 0, Window.height - self.height))
                    
                    # Validate final position before updating
                    if not (math.isnan(new_x) or math.isnan(new_y)):
                        self.pos = (new_x, new_y)
                        # Update last position only if valid
                        self.last_position = self.pos
                    else:
                        print("Invalid final position calculated")
                else:
                    print("Invalid movement vector calculated")
            else:
                print("Invalid angle calculation")
            
            # Update graphics
            self.update_graphics(None)
            
        except Exception as e:
            print(f"Error in move: {str(e)}")
            # Reset to safe state if error occurs
            if hasattr(self, 'last_position') and self.last_position:
                self.pos = self.last_position
            else:
                self.pos = (0, 0)
            self.current_speed = self.min_speed
            self.acceleration = 0.0
            self.steering_angle = 0.0

    def set_speed(self, multiplier):
        """Set speed multiplier"""
        self.current_speed = float(np.clip(self.current_speed * multiplier,
                                         self.min_speed,
                                         self.max_speed))

    def get_state(self):
        """Get the current state of the car for the TD3 agent"""
        try:
            # Get sensor values
            sensor_values = self.get_sensor_values(self.parent.sand if self.parent else np.zeros((longueur, largeur)))
            
            # Calculate normalized angle components
            angle_rad = math.radians(self.angle)
            angle_sin = math.sin(angle_rad)
            angle_cos = math.cos(angle_rad)
            
            # Create state vector
            state = np.array([
                *sensor_values[:3],  # First 3 sensor values
                self.steering_angle / self.max_steering_angle,  # Normalized steering
                self.current_speed / self.max_speed,  # Normalized speed
                angle_sin,  # Angle sine component
                angle_cos,  # Angle cosine component
                self.pos[0] / longueur,  # Normalized x position
                self.pos[1] / largeur    # Normalized y position
            ], dtype=np.float32)
            
            return state
            
        except Exception as e:
            print(f"Error in get_state: {str(e)}")
            # Return safe default state if error occurs
            return np.zeros(9, dtype=np.float32)

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

class Game(Widget):
    """Main game widget containing the car and game logic"""
    
    # Widget properties
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    label1 = ObjectProperty(None)
    label2 = ObjectProperty(None)
    label3 = ObjectProperty(None)
    score_label = ObjectProperty(None)
    stats_label = ObjectProperty(None)
    last_reward_label = ObjectProperty(None)

    # Define properties for position
    x = NumericProperty(0)
    y = NumericProperty(0)
    
    # Declare Kivy properties at class level
    car_x = NumericProperty(0)
    car_y = NumericProperty(0)
    
    def __init__(self, **kwargs):
        super(Game, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(self._on_keyboard_down, self)
        self._keyboard.bind(on_key_up=self._on_keyboard_up)
        
        # Game state flags
        self.game_over = False
        self.paused = False    # Initialize pause state
        
        # Initialize window dimensions
        self.longueur = 1429
        self.largeur = 660
        self.center = (self.longueur/2, self.largeur/2)
        
        # Initialize goal coordinates
        self.goal_x = 1420
        self.goal_y = 622
        
        # Delete all existing checkpoints
        checkpoint_dir = "pytorch_models/checkpoints"
        if os.path.exists(checkpoint_dir):
            for f in glob.glob(os.path.join(checkpoint_dir, "*.pth")):
                try:
                    os.remove(f)
                    print(f"Deleted checkpoint: {f}")
                except Exception as e:
                    print(f"Error deleting {f}: {e}")
        
        # Create fresh checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Model saving parameters
        self.checkpoint_dir = checkpoint_dir
        self.save_frequency = 100      # Save every 100 episodes
        self.best_reward = float('-inf')
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        
        # Replay buffer parameters
        self.replay_buffer_size = int(1e6)
        self.batch_size = 64
        self.warmup_steps = 500
        
        # TD3 parameters
        self.discount = 0.99
        self.tau = 0.001
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        
        # TD3 dimensions
        self.state_dim = 9           # 5 sensors + 2 orientation + 2 position
        self.action_dim = 2          # steering and acceleration
        self.max_action = 1.0
        
        # Car movement parameters - much slower for better control
        self.max_speed = 0.3        # Reduced from 3.0
        self.min_speed = 0.05       # Reduced minimum speed 
        self.acceleration_scale = 0.05  # Much gentler acceleration
        self.steering_scale = 0.1    # Gentler steering
        
        # Learning parameters
        self.exploration_noise = 0.5    # Reduced for smoother actions
        self.exploration_decay = 0.9995  # Slower decay
        self.min_exploration = 0.1       # Lower minimum
        self.learning_starts = 1000
        
        # Episode control
        self.max_steps_per_episode = 1000  # Longer episodes
        self.min_steps_per_episode = 100   # More time to learn
        self.steps_this_episode = 0
        self.total_steps = 0
        self.n_games = 0              # Track number of episodes
        
        # Initialize other attributes
        self.moving_avg_reward = 0
        self.best_moving_avg = float('-inf')
        
        # Initialize TD3 agent
        self.brain = TD3(self.state_dim, self.action_dim, self.max_action)
        
        # Initialize game state
        self.score = 0
        self.last_distance_to_goal = float('inf')
        self.consecutive_good_episodes = 0
        self.steps_without_progress = 0
        
        # Initialize state tracking
        self.last_state = np.zeros(self.state_dim)
        self.last_action = [0, 0]
        self.last_reward = 0
        
        # Initialize sand array
        self.sand = np.zeros((self.longueur, self.largeur))
        
        # Load and process the mask image
        img = PILImage.open("./images/MASK1.png").convert('L')
        img = img.resize((self.longueur, self.largeur))
        self.sand = np.asarray(img)/255
        
        # Reset all training history
        self.total_steps = 0
        self.n_games = 0
        self.best_reward = float('-inf')
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        
        # Initialize car movement parameters
        self.speed = self.min_speed
        self.angle = 0.0
        
        # Initialize car position
        self.car_x = self.center[0]
        self.car_y = self.center[1]
        
        # Initialize sensors array for reward calculation
        self.sensors = [1.0, 1.0, 1.0]
        
        # Initialize distance tracking
        self.last_distance = 0.0
        
        # Start the game loop
        Clock.schedule_interval(self.update, 1.0/60.0)
        Clock.schedule_interval(self._update_stats, 1.0/2.0)
        Clock.schedule_interval(self._update_display, 1.0/30.0)

    def update(self, dt=None):
        """Update game state with improved exploration and learning"""
        if self.game_over or self.paused or not self.car:
            return

        # Get current state
        state = np.array(self.car.get_state(), dtype=np.float32)
        
        # Select action with exploration noise
        action = self.brain.select_action(state)
        
        # Convert action to numpy array if it's a tensor
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        
        # Add exploration noise
        noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
        action = np.clip(action + noise, -1, 1)
        
        # Ensure action is a numpy array with float32 dtype
        action = np.array(action, dtype=np.float32)
        
        # Apply action and get next state and reward
        self._apply_action(action)
        
        # Let the car directly handle the movement
        if self.car:
            self.car.move(action)
        
        next_state = np.array(self.car.get_state(), dtype=np.float32)
        reward = self.calculate_reward()
        
        # Update episode tracking
        self.steps_this_episode += 1
        self.total_steps += 1
        self.current_episode_reward += reward
        
        # Check if episode should end with more lenient conditions
        done = False
        if (self.steps_this_episode >= 500 or  # Longer episodes
            (min(self.car.sensor_values) < 0.1 and self.steps_this_episode > 100) or  # Wall collision with more patience
            (self.car.current_speed < self.car.min_speed and self.steps_this_episode > 200)):  # Stuck with more patience
            done = True
        
        # Add experience to replay buffer
        self.brain.replay_buffer.add(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done
        )
        
        # Only train after warmup period
        if self.total_steps > self.warmup_steps:
            self.brain.train(self.brain.replay_buffer, self.batch_size)
        
        # Update exploration noise more slowly
        self.exploration_noise = max(self.min_exploration, 
                                   self.exploration_noise * self.exploration_decay)
        
        if done:
            # Print episode summary
            print(f"Episode {self.n_games} finished after {self.steps_this_episode} steps. "
                  f"Total reward: {self.current_episode_reward:.2f}")
            self.n_games += 1
            self.reset()
            self.serve_car()  # Reset car position after episode ends

    def calculate_reward(self):
        """Calculate reward with improved exploration incentives"""
        try:
            if not self.car:
                return 0.0
            
            # Get current car state
            x, y = self.car.pos
            speed = float(self.car.current_speed)
            
            # Calculate current distance to goal
            current_distance = np.sqrt((self.goal_x - x)**2 + (self.goal_y - y)**2)
            
            # Initialize last_distance if not set
            if not hasattr(self, 'last_distance'):
                self.last_distance = current_distance
            
            # Base reward
            reward = 0.0
            
            # Distance improvement reward
            if current_distance < self.last_distance:
                reward += 0.2  # Reward for moving towards goal
            else:
                reward -= 0.1  # Small penalty for moving away
            
            # Update last_distance for next iteration
            self.last_distance = current_distance
            
            # Get sand density at car position
            sand_x = int(np.clip(x, 0, self.sand.shape[0]-1))
            sand_y = int(np.clip(y, 0, self.sand.shape[1]-1))
            sand_density = float(self.sand[sand_x, sand_y])
            
            # Sand penalty with progressive component
            if sand_density > 0:
                reward -= 1.0
                if speed > self.car.max_speed * 0.5:  # Extra penalty for high speed on sand
                    reward -= 0.5
            else:
                reward += 0.1  # Small reward for being on road
            
            # Speed reward with context
            speed_ratio = speed / self.car.max_speed
            if sand_density > 0:
                # Reward lower speeds on sand
                speed_reward = 0.3 * (1 - speed_ratio)
            else:
                # Reward higher speeds on road
                speed_reward = 0.3 * speed_ratio
            reward += speed_reward
            
            # Goal reaching reward
            goal_threshold = 20
            if current_distance < goal_threshold:
                reward += 5.0  # Significant reward for reaching goal
                print("Goal reached!")
            
            # Living penalty to encourage faster goal reaching
            reward -= 0.01
            
            # Clip final reward
            reward = float(np.clip(reward, -10.0, 10.0))
            
            return reward
            
        except Exception as e:
            print(f"Error in calculate_reward: {str(e)}")
            return 0.0

    def _should_reset_episode(self):
        """Check if episode should end with more lenient conditions"""
        # End if car is truly stuck (not just moving slowly)
        if self.car.is_stuck() and self.car.stuck_counter > 30:  # Increased threshold
            return True
            
        # End if no progress for too long
        if self.steps_without_progress > 1000:  # Increased timeout
            return True
            
        # End if car is off track for too long
        avg_sensor = (self.car.signal1 + self.car.signal2 + self.car.signal3) / 3.0
        if avg_sensor > 0.8 and self.steps_without_progress > 100:  # More lenient
            return True
            
        return False

    def _end_episode(self):
        """Handle episode end with improved model saving"""
        episode_reward = self.current_episode_reward
        self.episode_rewards.append(episode_reward)
        
        # Calculate moving average
        if len(self.episode_rewards) > 10:
            self.moving_avg_reward = np.mean(self.episode_rewards[-10:])
        else:
            self.moving_avg_reward = np.mean(self.episode_rewards)
        
        # Save conditions
        should_save = False
        save_reason = ""
        
        # 1. New best episode
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            should_save = True
            save_reason = "new_best_reward"
        
        # 2. New best moving average
        if self.moving_avg_reward > self.best_moving_avg:
            self.best_moving_avg = self.moving_avg_reward
            should_save = True
            save_reason = "new_best_average"
        
        # 3. Periodic checkpoint
        if self.n_games % self.save_frequency == 0:
            should_save = True
            save_reason = "periodic"
        
        # Save model and parameters
        if should_save:
            checkpoint = {
                'episode': self.n_games,
                'model_state_dict': self.brain.actor.state_dict(),
                'optimizer_state_dict': self.brain.actor_optimizer.state_dict(),
                'best_reward': self.best_reward,
                'best_moving_avg': self.best_moving_avg,
                'episode_rewards': self.episode_rewards,
                'exploration_noise': self.exploration_noise,
                'total_steps': self.total_steps
            }
            
            # Save with descriptive filename
            filename = f"checkpoint_{self.n_games}_{save_reason}.pth"
            torch.save(checkpoint, f"{self.checkpoint_dir}/{filename}")
            
            # Also save as latest
            torch.save(checkpoint, f"{self.checkpoint_dir}/latest.pth")
            
            print(f"\nSaved checkpoint! Reason: {save_reason}")
            print(f"Episode Reward: {episode_reward:.2f}")
            print(f"Moving Average: {self.moving_avg_reward:.2f}")
        
        # Print episode summary
        print(f"\nEpisode {self.n_games + 1} Summary:")
        print(f"Total Steps: {self.total_steps}")
        print(f"Episode Steps: {self.steps_this_episode}")
        print(f"Episode Reward: {episode_reward:.2f}")
        print(f"Moving Average: {self.moving_avg_reward:.2f}")
        print(f"Best Reward: {self.best_reward:.2f}")
        print(f"Exploration: {self.exploration_noise:.3f}")
        print("-" * 50)
        
        # Reset for next episode
        self.n_games += 1
        self.current_episode_reward = 0.0
        self.steps_this_episode = 0

    def _update_stats(self, dt):
        """Update game statistics display"""
        if self.stats_label:
            self.stats_label.text = f"Games: {self.n_games}\nScore: {self.score}"
        if self.last_reward_label:
            self.last_reward_label.text = f"Last Reward: {self.last_reward:.2f}"

    def _update_display(self, dt):
        """Update visual elements"""
        # Update sensor positions
        if self.ball1 and self.car:
            self.ball1.pos = self.car.sensor1
        if self.ball2 and self.car:
            self.ball2.pos = self.car.sensor2
        if self.ball3 and self.car:
            self.ball3.pos = self.car.sensor3
        
        # Update sensor values display
        if self.label1 and self.car:
            self.label1.text = f"{self.car.signal1:.2f}"
        if self.label2 and self.car:
            self.label2.text = f"{self.car.signal2:.2f}"
        if self.label3 and self.car:
            self.label3.text = f"{self.car.signal3:.2f}"
        
        # Update score
        if self.score_label and self.car:
            self.score_label.text = f"Score: {self.score}"

    def serve_car(self):
        """Initialize car with randomized starting conditions"""
        try:
            # Randomize starting position within safe bounds
            margin = 50  # Keep car away from edges
            x = np.random.uniform(margin, self.longueur - margin)
            y = np.random.uniform(margin, self.largeur - margin)
            
            # Randomize initial speed
            initial_speed = np.random.uniform(0.03, 0.07)
            
            # Randomize initial angle
            initial_angle = np.random.uniform(0, 360)
            
            # Create or reset car
            if not self.car:
                self.car = Car()
                self.add_widget(self.car)
            
            # Set randomized initial conditions
            self.car.pos = (x, y)
            self.car.angle = initial_angle
            self.car.current_speed = initial_speed
            
            # Reset episode tracking
            self.steps_this_episode = 0
            self.max_steps = np.random.randint(150, 250)  # Randomize episode length
            
            # Reset distance tracking when serving car
            if self.car:
                x, y = self.car.pos
                self.last_distance = np.sqrt((self.goal_x - x)**2 + (self.goal_y - y)**2)
            
            print(f"Car initialized at ({x:.1f}, {y:.1f}) with speed {initial_speed:.2f} and angle {initial_angle:.1f}")
            
        except Exception as e:
            print(f"Error in serve_car: {str(e)}")
            self.last_distance = 0.0  # Safe default
            if self.car:
                self.car.pos = (self.longueur/2, self.largeur/2)
                self.car.angle = 0
                self.car.current_speed = 0.05

    def _handle_boundary_collisions(self):
        """Check if car has hit track boundaries"""
        return any(signal > 0.9 for signal in [self.car.signal1, self.car.signal2, self.car.signal3])

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'spacebar':
            self.paused = not self.paused
        return True

    def _on_keyboard_up(self, keyboard, keycode):
        return True

    def _check_target_reached(self):
        """Check if car has reached the current target"""
        try:
            # Calculate distance to current target
            current_distance = Vector(self.car.pos).distance(Vector(self.goal_x, self.goal_y))
            
            # Consider target reached if within 50 pixels
            return current_distance < 50
        except Exception as e:
            print(f"Error in target check: {str(e)}")
            return False

    def _apply_action(self, action):
        """Apply action to the car with improved numerical stability"""
        try:
            # Validate input action
            if not isinstance(action, (list, np.ndarray)) or len(action) != 2:
                print("Invalid action format")
                return
            
            # Extract and validate action components
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            steering = float(np.clip(action[0], -1.0, 1.0))
            acceleration = float(np.clip(action[1], -1.0, 1.0))
            
            # Use the car's parameters for movement
            if self.car:
                # Convert actions to acceleration and steering using car's parameters
                acceleration = float(acceleration * self.car.max_acceleration)
                steering = float(steering * self.car.max_steering_angle)
                
                # Update speed with acceleration and momentum
                old_speed = float(self.car.current_speed)
                new_speed = float(np.clip(
                    old_speed + acceleration,
                    self.car.min_speed,
                    self.car.max_speed
                ))
                
                # Limit speed change per frame for smoother movement
                max_speed_change = 0.005
                if abs(new_speed - old_speed) > max_speed_change:
                    new_speed = old_speed + max_speed_change * np.sign(new_speed - old_speed)
                
                # Update car's speed and angle
                self.car.current_speed = float(new_speed)
                self.car.angle = float((self.car.angle + steering) % 360.0)
                
                # Let the car handle its own movement
                self.car.move(action)
                
                # Update car sensors
                if hasattr(self.car, 'update_sensors'):
                    self.car.update_sensors()
                    self.car.get_sensor_values(self.sand)
            
        except Exception as e:
            print(f"Error in _apply_action: {str(e)}")
            # Reset to safe state if error occurs
            if self.car:
                self.car.current_speed = self.car.min_speed
                self.car.angle = 0.0
                self.car.pos = (self.longueur/2, self.largeur/2)

    def get_state(self):
        """Get the current state for the TD3 agent"""
        try:
            if not self.car:
                return np.zeros(9, dtype=np.float32)  # Default state size is 9
            
            # Get state from car
            return self.car.get_state()
            
        except Exception as e:
            print(f"Error in get_state: {str(e)}")
            return np.zeros(9, dtype=np.float32)  # Safe default state

    def reset(self):
        """Reset the game state with improved randomization"""
        try:
            # Randomize starting position within safe bounds
            margin = 50
            x = np.random.uniform(margin, self.longueur - margin)
            y = np.random.uniform(margin, self.largeur - margin)
            
            # Randomize goal position
            self.goal_x = np.random.uniform(margin, self.longueur - margin)
            self.goal_y = np.random.uniform(margin, self.largeur - margin)
            
            # Serve car with random initial conditions
            self.serve_car()
            
            # Reset episode tracking
            self.total_reward = 0
            self.steps_this_episode = 0
            self.max_steps = np.random.randint(150, 250)
            
            # Reset distance tracking
            if self.car:
                x, y = self.car.pos
                self.last_distance = np.sqrt((self.goal_x - x)**2 + (self.goal_y - y)**2)
            
            print(f"Goal set at ({self.goal_x:.1f}, {self.goal_y:.1f})")
            
            # Return initial state
            return self.get_state()
            
        except Exception as e:
            print(f"Error in reset: {str(e)}")
            return np.zeros(9, dtype=np.float32)  # Safe default state

# Adding the painting tools

class MyPaintWidget(Widget):
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y, sand
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            
            # Ensure coordinates are within bounds
            x = max(0, min(int(touch.x), longueur-1))
            y = max(0, min(int(touch.y), largeur-1))
            sand[x, y] = 1
            
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y, sand
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = max(0, min(int(touch.x), longueur-1))
            y = max(0, min(int(touch.y), largeur-1))
            
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            
            # Ensure coordinates are within bounds for sand array
            x_start = max(0, x - 10)
            x_end = min(longueur, x + 10)
            y_start = max(0, y - 10)
            y_end = min(largeur, y + 10)
            sand[x_start:x_end, y_start:y_end] = 1
            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):
    def __init__(self):
        super(CarApp, self).__init__()
        self.brain = TD3(9, 2, 1.0)  # Initialize TD3 agent
        if not os.path.exists("./pytorch_models"):
            os.makedirs("./pytorch_models")
        try:
            self.brain.load("last_brain", "./pytorch_models")
            print("Loaded previous model successfully")
        except:
            print("No previous model found, starting fresh")

    def build(self):
        # Set window properties
        Window.size = (longueur, largeur)
        Window.top = 50  # Position window 50 pixels from top
        Window.left = 50  # Position window 50 pixels from left
        Window.borderless = False
        Window.clearcolor = (1, 1, 1, 1)  # Set window background to white
        
        # Create game instance
        parent = Game()
        
        # Initialize car explicitly with error handling
        try:
            # Serve the car (initialize position and state)
            parent.serve_car()
            print("Car initialized successfully")
            
            # Set initial values for successful start
            if parent.car:
                parent.car.current_speed = 0.05  # Use lower initial speed
                parent.car.max_acceleration = 0.3  # Limit maximum acceleration
                parent.car.min_acceleration = -0.02  # Gentler deceleration
                parent.car.max_steering_angle = 3.0  # Reduced steering angle
                parent.car.update_sensors()     # Initialize sensors
                parent.car.get_sensor_values(parent.sand)  # Get initial sensor values
                print(f"Car position: {parent.car.pos}, Speed: {parent.car.current_speed}")
        except Exception as e:
            print(f"Error initializing car: {str(e)}")
        
        # Schedule game updates
        Clock.schedule_interval(parent.update, 1.0/60.0)
        
        self.painter = MyPaintWidget()
        
        # Create main layout
        root = BoxLayout(orientation='vertical')
        
        # Create game area
        game_area = BoxLayout(orientation='vertical')
        game_area.add_widget(self.painter)
        game_area.add_widget(parent)
        
        root.add_widget(game_area)
        
        return root

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        self.brain.save("last_brain", "./pytorch_models")
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        self.brain.load("last_brain", "./pytorch_models")

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
