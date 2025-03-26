# Car Simulation with TD3 Reinforcement Learning

This project implements a Twin Delayed DDPG (TD3) reinforcement learning algorithm to train a car to navigate through randomly generated roads. The simulation includes a graphical interface using Pygame.

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pygame
- Gym

Install the requirements using:
```bash
pip install -r requirements.txt
```

## Project Structure

- `car_env.py`: Custom environment implementation for the car simulation
- `td3_agent.py`: TD3 algorithm implementation
- `train.py`: Script to train the model
- `run_model.py`: Script to run the trained model
- `models/`: Directory where trained models are saved

## Training the Model

To train the model, run:
```bash
python train.py
```

The training script will:
1. Train the agent for 50,000 steps
2. Save the best model based on evaluation performance
3. Save checkpoints every 5,000 steps
4. Display the training progress and rewards

## Running the Trained Model

To run the trained model, use:
```bash
python run_model.py
```

This will load the best model and run 5 episodes to demonstrate the car's performance.

## Environment Details

The environment features:
- A car that can be controlled with steering and acceleration
- Randomly generated roads
- Visual feedback using Pygame
- Reward system based on staying on the road and reaching the end

## Model Architecture

The TD3 implementation includes:
- Actor network for action selection
- Twin critic networks for Q-value estimation
- Target networks for stable training
- Experience replay buffer
- Delayed policy updates
- Clipped double Q-learning 