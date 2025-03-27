# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import os

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Importing the TD3 object from our AI in ai.py
from ai import TD3

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Training parameters and statistics
TRAINING_STEPS = 50000
TRAINING_MODE = True  # Set to False to run inference with saved model
MODEL_PATH = "./models/car_model"

# Enhanced training statistics
total_steps = 0
episode_rewards = []
current_episode_reward = 0
steps_since_last_goal = 0
goals_reached = 0
off_road_count = 0
avg_speed = []
training_stats = {
    'episode_rewards': [],
    'avg_rewards': [],
    'goals_reached': 0,
    'off_road_count': 0,
    'avg_speed': [],
    'steps_per_goal': []
}

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = TD3(state_dim=9, action_dim=1, max_action=1.0)  # Changed action_dim to 1 for continuous rotation
action2rotation = [-5, -2, 0, 2, 5]  # Wider range of rotation angles
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global longueur
    global largeur
    
    # Initialize dimensions first
    longueur = 1429  # Match the window width
    largeur = 660    # Match the window height
    
    # Create sand array with correct dimensions
    sand = np.zeros((longueur, largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    
    # Set initial goal position
    goal_x = 1420
    goal_y = 622
    first_update = False
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0

def update_stats(reward, speed, reached_goal=False, off_road=False):
    """Update training statistics"""
    global training_stats
    
    training_stats['episode_rewards'].append(reward)
    training_stats['avg_speed'].append(speed)
    
    if reached_goal:
        training_stats['goals_reached'] += 1
        if len(training_stats['steps_per_goal']) > 0:
            training_stats['steps_per_goal'].append(total_steps - sum(training_stats['steps_per_goal']))
        else:
            training_stats['steps_per_goal'].append(total_steps)
    
    if off_road:
        training_stats['off_road_count'] += 1
    
    # Calculate moving averages
    window = min(100, len(training_stats['episode_rewards']))
    if window > 0:
        training_stats['avg_rewards'].append(np.mean(training_stats['episode_rewards'][-window:]))

def print_training_stats():
    """Print current training statistics"""
    print(f"\nTraining Statistics at step {total_steps}:")
    print(f"Average Reward (last 100): {np.mean(training_stats['avg_rewards'][-100:] if training_stats['avg_rewards'] else [0]):.2f}")
    print(f"Goals Reached: {training_stats['goals_reached']}")
    print(f"Off-road Count: {training_stats['off_road_count']}")
    print(f"Average Speed: {np.mean(training_stats['avg_speed'][-100:] if training_stats['avg_speed'] else [0]):.2f}")
    if len(training_stats['steps_per_goal']) > 0:
        print(f"Average Steps per Goal: {np.mean(training_stats['steps_per_goal']):.0f}")
    print("-" * 50)

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
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
    sensor4_x = NumericProperty(0)
    sensor4_y = NumericProperty(0)
    sensor4 = ReferenceListProperty(sensor4_x, sensor4_y)
    sensor5_x = NumericProperty(0)
    sensor5_y = NumericProperty(0)
    sensor5 = ReferenceListProperty(sensor5_x, sensor5_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)
    signal4 = NumericProperty(0)
    signal5 = NumericProperty(0)

    def move(self, rotation):
        # Update position with velocity clamping
        new_pos = Vector(*self.velocity) + self.pos
        
        # Clamp position within bounds with margin
        margin = 10
        new_pos[0] = max(margin, min(new_pos[0], longueur - margin))
        new_pos[1] = max(margin, min(new_pos[1], largeur - margin))
        self.pos = new_pos
        
        # Update angle and rotation with proper modulo to keep in range [0, 360)
        self.rotation = float(rotation)  # Ensure rotation is a float
        self.angle = (self.angle + self.rotation) % 360
        
        # Calculate new velocity based on angle and constant speed
        speed = 3.0  # Constant speed
        angle_rad = np.radians(self.angle)
        # Ensure velocity components are floats
        self.velocity_x = float(speed * np.cos(angle_rad))
        self.velocity_y = float(speed * np.sin(angle_rad))
        
        # Increased sensor range and added more sensors with proper angle calculations
        self.sensor1 = Vector(60, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(60, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(60, 0).rotate((self.angle-30)%360) + self.pos
        self.sensor4 = Vector(60, 0).rotate((self.angle+60)%360) + self.pos
        self.sensor5 = Vector(60, 0).rotate((self.angle-60)%360) + self.pos
        
        # Add bounds checking for sensor positions
        def get_sand_value(x, y):
            x = min(max(int(x), 0), longueur-1)
            y = min(max(int(y), 0), largeur-1)
            return int(np.sum(sand[x-10:x+10, y-10:y+10]))/400.
        
        self.signal1 = get_sand_value(self.sensor1_x, self.sensor1_y)
        self.signal2 = get_sand_value(self.sensor2_x, self.sensor2_y)
        self.signal3 = get_sand_value(self.sensor3_x, self.sensor3_y)
        self.signal4 = get_sand_value(self.sensor4_x, self.sensor4_y)
        self.signal5 = get_sand_value(self.sensor5_x, self.sensor5_y)
        
        # Update boundary checks for all sensors
        for i in range(1, 6):
            sensor_x = getattr(self, f'sensor{i}_x')
            sensor_y = getattr(self, f'sensor{i}_y')
            if sensor_x > longueur-10 or sensor_x < 10 or sensor_y > largeur-10 or sensor_y < 10:
                setattr(self, f'signal{i}', 1.0)

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity_x = float(3.0)  # Set initial velocity to match constant speed
        self.car.velocity_y = float(0.0)  # Set initial velocity to match constant speed
        self.car.angle = 0  # Reset angle to 0

    def update(self, dt):
        global brain, last_reward, scores, last_distance, goal_x, goal_y, swap
        global total_steps, current_episode_reward, episode_rewards, training_stats
        
        if first_update:
            init()

        # Calculate orientation to goal using proper angle calculations
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        goal_angle = np.degrees(np.arctan2(yy, xx)) % 360
        car_angle = self.car.angle % 360
        
        # Calculate the smallest angle difference between current angle and goal angle
        angle_diff = ((goal_angle - car_angle + 180) % 360) - 180
        orientation = angle_diff / 180.0  # Normalize to [-1, 1]
        
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        normalized_distance = distance/np.sqrt(longueur**2 + largeur**2)
        current_speed = np.sqrt(self.car.velocity[0]**2 + self.car.velocity[1]**2)
        normalized_speed = current_speed/3.0
        
        # Enhanced state representation
        last_signal = [
            self.car.signal1, self.car.signal2, self.car.signal3,
            self.car.signal4, self.car.signal5,
            orientation,
            angle_diff / 180.0,  # Add normalized angle difference
            normalized_distance,
            normalized_speed
        ]
        
        # Get action from TD3 with reduced noise during training
        action = brain.select_action(np.array(last_signal))
        if TRAINING_MODE:
            noise = np.random.normal(0, 0.05, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        
        # Convert continuous action to rotation angle and ensure it's a valid number
        rotation = float(action[0] * 5)  # Scale action to rotation range [-5, 5] and convert to float
        
        self.car.move(rotation)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        # Get next state with enhanced representation
        next_signal = [
            self.car.signal1, self.car.signal2, self.car.signal3,
            self.car.signal4, self.car.signal5,
            orientation,
            angle_diff / 180.0,
            normalized_distance,
            normalized_speed
        ]
        
        # Determine if episode is done and calculate reward
        done = False
        reached_goal = False
        off_road = False
        car_x = min(max(int(self.car.x), 0), longueur-1)
        car_y = min(max(int(self.car.y), 0), largeur-1)
        
        # Enhanced reward structure
        if sand[car_x, car_y] > 0:  # Car is on sand/off-road
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            last_reward = -1.0
            done = True
            off_road = True
        else:  # Car is on road
            # Base reward for staying on road
            last_reward = 0.2
            
            # Progressive reward based on distance to goal
            distance_reward = 1.0 * (1 - normalized_distance)
            last_reward += distance_reward
            
            # Enhanced reward for getting closer to goal
            if distance < last_distance:
                last_reward += 1.0
            else:
                last_reward -= 0.1
            
            # Speed reward with lower threshold and exploration bonus
            if current_speed > 1.0:
                last_reward += 0.5 * min(current_speed/3, 1.0)
                # Add exploration bonus for moving in new directions
                if abs(rotation) > 0:
                    last_reward += 0.3
            
            # Stuck penalty with memory
            if current_speed < 0.1:
                last_reward -= 0.5
                done = True

        # Boundary penalties with softer penalties
        margin = 10
        if (self.car.x < margin or self.car.x > self.width - margin or 
            self.car.y < margin or self.car.y > self.height - margin):
            last_reward = -1.0
            done = True
            # Clamp position within bounds
            self.car.x = max(margin, min(self.car.x, self.width - margin))
            self.car.y = max(margin, min(self.car.y, self.height - margin))

        # Enhanced goal reached reward
        if distance < 25:
            last_reward = 15.0
            done = True
            reached_goal = True
            if swap == 1:
                goal_x = 1420
                goal_y = 622
                swap = 0
            else:
                goal_x = 9
                goal_y = 85
                swap = 1

        # Update training statistics
        current_episode_reward += last_reward
        total_steps += 1
        update_stats(last_reward, abs(self.car.velocity[0]), reached_goal, off_road)

        if TRAINING_MODE:
            # Add experience to replay buffer and train
            brain.replay_buffer.add(
                np.array(last_signal),
                np.array([action[0]]),
                np.array(next_signal),
                last_reward,
                done
            )
            brain.train(brain.replay_buffer, brain.batch_size)

            # Handle episode completion
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                self.serve_car()
                
                # Print statistics every 1000 steps
                if total_steps % 1000 == 0:
                    print_training_stats()
                    brain.save("car_model", "./models")
                
                # Check if training is complete
                if total_steps >= TRAINING_STEPS:
                    print("\nTraining completed!")
                    print_training_stats()
                    brain.save("car_model", "./models")
                    
                    # Plot final training results
                    plt.figure(figsize=(15, 10))
                    
                    plt.subplot(2, 2, 1)
                    plt.plot(training_stats['avg_rewards'])
                    plt.title('Average Rewards')
                    plt.xlabel('Episodes')
                    
                    plt.subplot(2, 2, 2)
                    plt.plot(training_stats['avg_speed'])
                    plt.title('Average Speed')
                    plt.xlabel('Steps')
                    
                    plt.subplot(2, 2, 3)
                    plt.plot(training_stats['steps_per_goal'])
                    plt.title('Steps per Goal')
                    plt.xlabel('Goals Reached')
                    
                    plt.tight_layout()
                    plt.savefig('training_results.png')
                    plt.close()
        
        last_distance = distance

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        
        # Only add training controls if in training mode
        if TRAINING_MODE:
            clearbtn = Button(text='clear')
            savebtn = Button(text='save', pos=(parent.width, 0))
            loadbtn = Button(text='load', pos=(2 * parent.width, 0))
            clearbtn.bind(on_release=self.clear_canvas)
            savebtn.bind(on_release=self.save)
            loadbtn.bind(on_release=self.load)
            parent.add_widget(clearbtn)
            parent.add_widget(savebtn)
            parent.add_widget(loadbtn)
        
        parent.add_widget(self.painter)
        return parent

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        global sand
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        global brain, training_stats, total_steps, current_episode_reward
        
        # Create checkpoint with all relevant information
        checkpoint = {
            'total_steps': total_steps,
            'training_stats': training_stats,
            'current_episode_reward': current_episode_reward,
            'timestamp': time.strftime("%Y%m%d-%H%M%S")
        }
        
        # Save the checkpoint with timestamp
        checkpoint_name = f"car_model_checkpoint_{checkpoint['timestamp']}"
        brain.save(checkpoint_name, "./models")
        
        # Save training stats separately
        np.save(f"./models/{checkpoint_name}_stats.npy", checkpoint)
        
        print(f"\nCheckpoint saved successfully!")
        print(f"Checkpoint name: {checkpoint_name}")
        print(f"Total steps: {total_steps}")
        print(f"Goals reached: {training_stats['goals_reached']}")
        print(f"Average reward: {np.mean(training_stats['avg_rewards'][-100:] if training_stats['avg_rewards'] else [0]):.2f}")
        print("-" * 50)

    def load(self, obj):
        global brain, training_stats, total_steps, current_episode_reward
        
        try:
            # List all available checkpoints
            import glob
            checkpoints = glob.glob("./models/car_model_checkpoint_*[0-9]")
            if not checkpoints:
                print("No checkpoints found!")
                return
                
            # Get the latest checkpoint
            latest_checkpoint = max(checkpoints)
            checkpoint_name = latest_checkpoint.split('/')[-1]
            
            # Load the model
            brain.load(checkpoint_name, "./models")
            
            # Load training stats
            stats_file = f"./models/{checkpoint_name}_stats.npy"
            if os.path.exists(stats_file):
                checkpoint = np.load(stats_file, allow_pickle=True).item()
                training_stats = checkpoint['training_stats']
                total_steps = checkpoint['total_steps']
                current_episode_reward = checkpoint['current_episode_reward']
                
                print(f"\nCheckpoint loaded successfully!")
                print(f"Checkpoint name: {checkpoint_name}")
                print(f"Total steps: {total_steps}")
                print(f"Goals reached: {training_stats['goals_reached']}")
                print(f"Average reward: {np.mean(training_stats['avg_rewards'][-100:] if training_stats['avg_rewards'] else [0]):.2f}")
                print("-" * 50)
            else:
                print("Warning: Training stats not found, only model loaded")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

# Running the whole thing
if __name__ == '__main__':
    if not TRAINING_MODE:
        # Load the trained model for inference
        brain.load("car_model", "./models")
        print("Loaded trained model for inference")
    CarApp().run()
