import torch
import numpy as np
from car_env import CarEnv
from td3_agent import TD3Agent
import os
import sys
import time
from datetime import datetime

def train():
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        print("Created models directory")

        # Create environment and agent
        env = CarEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        agent = TD3Agent(state_dim, action_dim, device)
        
        # Training parameters
        max_steps = 50000
        eval_freq = 1000
        save_freq = 5000
        best_reward = float('-inf')
        
        # Force save initial model
        print("Saving initial model...")
        agent.save('models/initial_model.pth')
        
        # Training loop
        state = env.reset()
        episode_reward = 0
        step = 0
        episode_count = 0
        last_save_time = time.time()
        last_print_time = time.time()
        
        print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        while step < max_steps:
            try:
                # Print progress every 5 seconds
                current_time = time.time()
                if current_time - last_print_time > 5:
                    print(f"Step: {step}/{max_steps}, Episodes: {episode_count}, Buffer size: {len(agent.replay_buffer)}")
                    last_print_time = current_time
                
                # Select action
                action = agent.select_action(state)
                
                # Take action in environment
                next_state, reward, done, _ = env.step(action)
                
                # Store transition in replay buffer
                agent.replay_buffer.append((state, action, reward, next_state, done))
                
                # Train agent only if buffer is large enough
                if len(agent.replay_buffer) >= agent.batch_size:
                    agent.train()
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                step += 1
                
                # Render environment (with reduced frequency)
                if step % 2 == 0:  # Render every other step
                    env.render()
                
                # Handle episode completion
                if done:
                    episode_count += 1
                    print(f"Episode {episode_count} completed at step {step}, Reward: {episode_reward:.2f}")
                    state = env.reset()
                    episode_reward = 0
                    
                    # Evaluate and save best model
                    if step % eval_freq == 0:
                        print(f"Starting evaluation at step {step}...")
                        eval_reward = evaluate(env, agent, num_episodes=5)
                        print(f"Evaluation completed. Average reward: {eval_reward:.2f}")
                        
                        if eval_reward > best_reward:
                            best_reward = eval_reward
                            agent.save('models/best_model.pth')
                            print(f"New best model saved with reward: {best_reward:.2f}")
                    
                    # Regular model saving
                    if step % save_freq == 0:
                        current_time = time.time()
                        if current_time - last_save_time > 60:  # Save at most once per minute
                            agent.save(f'models/model_step_{step}.pth')
                            last_save_time = current_time
                            print(f"Checkpoint saved at step {step}")
                
                # Add a small delay to prevent CPU overload
                time.sleep(0.001)  # Reduced delay
                
            except Exception as e:
                print(f"Error during training step {step}: {str(e)}")
                print("Attempting to recover...")
                state = env.reset()
                episode_reward = 0
                continue
        
        print(f"Training completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Ensure final model is saved
        print("Saving final model...")
        final_path = 'models/final_model.pth'
        agent.save(final_path)
        print(f"Final model saved to {final_path}")
        
        # If no best model was saved, save the final one as best
        if not os.path.exists('models/best_model.pth'):
            print("No best model found, saving final model as best...")
            agent.save('models/best_model.pth')

    except Exception as e:
        print(f"Fatal error during training: {str(e)}")
        # Try to save emergency backup of model
        try:
            agent.save('models/emergency_backup.pth')
            print("Emergency backup saved")
        except:
            print("Could not save emergency backup")
        raise
    finally:
        env.close()

def evaluate(env, agent, num_episodes=5):
    total_reward = 0
    successful_episodes = 0
    
    for episode in range(num_episodes):
        try:
            state = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            max_steps_per_episode = 1000  # Prevent infinite loops
            
            while not done and steps < max_steps_per_episode:
                action = agent.select_action(state, noise=0)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if steps % 2 == 0:  # Render every other step
                    env.render()
                steps += 1
            
            if steps < max_steps_per_episode:  # Only count episodes that didn't timeout
                total_reward += episode_reward
                successful_episodes += 1
                print(f"Evaluation episode {episode + 1}: {episode_reward:.2f} (steps: {steps})")
            else:
                print(f"Evaluation episode {episode + 1} timed out after {steps} steps")
            
        except Exception as e:
            print(f"Error during evaluation episode {episode + 1}: {str(e)}")
            continue
    
    return total_reward / successful_episodes if successful_episodes > 0 else float('-inf')

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1) 