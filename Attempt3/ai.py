# AI for Self Driving Car

# Importing the libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque
import copy

# Creating the architecture of the Neural Network

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        # Increase network capacity
        self.l1 = nn.Linear(state_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.l2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.l3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.l4 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        a = F.relu(self.ln1(self.l1(state)))
        a = self.dropout(a)
        a = F.relu(self.ln2(self.l2(a)))
        a = self.dropout(a)
        a = F.relu(self.ln3(self.l3(a)))
        return self.max_action * torch.tanh(self.l4(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture with increased capacity
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.l2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.l3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.l4 = nn.Linear(256, 1)

        # Q2 architecture
        self.l5 = nn.Linear(state_dim + action_dim, 512)
        self.ln4 = nn.LayerNorm(512)
        self.l6 = nn.Linear(512, 512)
        self.ln5 = nn.LayerNorm(512)
        self.l7 = nn.Linear(512, 256)
        self.ln6 = nn.LayerNorm(256)
        self.l8 = nn.Linear(256, 1)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.ln1(self.l1(sa)))
        q1 = self.dropout(q1)
        q1 = F.relu(self.ln2(self.l2(q1)))
        q1 = self.dropout(q1)
        q1 = F.relu(self.ln3(self.l3(q1)))
        q1 = self.l4(q1)

        q2 = F.relu(self.ln4(self.l5(sa)))
        q2 = self.dropout(q2)
        q2 = F.relu(self.ln5(self.l6(q2)))
        q2 = self.dropout(q2)
        q2 = F.relu(self.ln6(self.l7(q2)))
        q2 = self.l8(q2)
        return q1, q2

# Select device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize replay buffer with larger size
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(2e6))  # Increased from 1e6
        
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)  # Increased from 5e-5

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)  # Increased from 5e-5

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005  # Increased from 0.001 for faster target network updates
        self.policy_noise = 0.1  # Reduced from 0.2 for more stable learning
        self.noise_clip = 0.3  # Reduced from 0.5 for more stable learning
        self.policy_freq = 2
        self.total_it = 0
        
        # Training parameters
        self.batch_size = 512  # Increased from 256 for better gradient estimates
        self.warmup_steps = 2000  # Increased from 1000 for better initial exploration
        self.total_steps = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size):
        if replay_buffer.size < batch_size:
            return
        
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.forward(state, self.actor(state))[0].mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, directory):
        """Save only the last run model files"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save only the last run files
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), f"{directory}/{filename}_actor_optimizer.pth")
        
        print(f"Saved model to {directory}/{filename}")

    def load(self, filename, directory):
        """Load the last run model files"""
        try:
            self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
            self.actor_optimizer.load_state_dict(torch.load(f"{directory}/{filename}_actor_optimizer.pth"))
            print(f"Loaded model from {directory}/{filename}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Starting with fresh model")

    def load_checkpoint(self, checkpoint_path):
        """Load model and training state from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path)
            
            # Load model and optimizer states
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"\nLoaded checkpoint from episode {checkpoint['episode']}")
            print(f"Best reward: {checkpoint['best_reward']:.2f}")
            print(f"Best moving average: {checkpoint['best_moving_avg']:.2f}")
            
            return checkpoint
            
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return None