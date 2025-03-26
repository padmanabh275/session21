import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TD3Agent:
    def __init__(self, state_dim, action_dim, device='cuda'):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # Critic networks
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)
        
        # Hyperparameters
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=1000000)
        
    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -1, 1)
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Convert lists to numpy arrays first for better performance
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # Convert numpy arrays to tensors
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.FloatTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # Select next action with noise
        noise = torch.randn_like(action_batch) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_action_batch = (self.actor_target(next_state_batch) + noise).clamp(-1, 1)
        
        # Compute target Q-value
        target_q1 = self.critic1_target(next_state_batch, next_action_batch)
        target_q2 = self.critic2_target(next_state_batch, next_action_batch)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * self.gamma * target_q
        
        # Update critics
        current_q1 = self.critic1(state_batch, action_batch)
        current_q2 = self.critic2(state_batch, action_batch)
        
        critic1_loss = F.mse_loss(current_q1, target_q.detach())
        critic2_loss = F.mse_loss(current_q2, target_q.detach())
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic1(state_batch, self.actor(state_batch)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            print(f"Saving model to {filename}")
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'actor_target_state_dict': self.actor_target.state_dict(),
                'critic1_state_dict': self.critic1.state_dict(),
                'critic2_state_dict': self.critic2.state_dict(),
                'critic1_target_state_dict': self.critic1_target.state_dict(),
                'critic2_target_state_dict': self.critic2_target.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
                'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            }, filename)
            print(f"Successfully saved model to {filename}")
        except Exception as e:
            print(f"Error saving model to {filename}: {str(e)}")
            raise
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict']) 