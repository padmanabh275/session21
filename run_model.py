import torch
from car_env import CarEnv
from td3_agent import TD3Agent
import os

def run_model(model_path):
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please ensure that:")
        print("1. You have run the training script (train.py) first")
        print("2. The training completed successfully")
        print("3. The model was saved in the 'models' directory")
        return

    # Create environment and agent
    env = CarEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3Agent(state_dim, action_dim, device)
    
    try:
        # Load the trained model
        print(f"Loading model from {model_path}")
        agent.load(model_path)
        print("Model loaded successfully!")
        
        # Run episodes
        num_episodes = 5
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            print(f"Starting episode {episode + 1}")
            
            while not done:
                # Select action without noise for evaluation
                action = agent.select_action(state, noise=0)
                
                # Take action in environment
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                
                # Render environment
                env.render()
            
            print(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}")
    
    except Exception as e:
        print(f"Error running model: {str(e)}")
    finally:
        env.close()

if __name__ == "__main__":
    model_path = "models/best_model.pth"  # Path to your trained model
    run_model(model_path) 