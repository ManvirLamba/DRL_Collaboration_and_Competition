
import torch

import random
from collections import namedtuple, deque
import numpy as np

from ddpg_agent import Agent

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
UPDATE_FREQ = 1

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random.seed(random_seed)
        
        # Created Agents
        self.agents = [Agent(state_size, action_size, num_agents, random_seed) for i in range(num_agents)]
        
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.step_count = 0
        
    def reset(self):
        for agent in self.agents:
            agent.reset()
            

    def act(self, state, i_episode, add_noise=True):
        actions = []
        for agent_state, agent in zip(state, self.agents):
            action = agent.act(agent_state, i_episode, add_noise)
            action = np.reshape(action, newshape=(-1))
            actions.append(action)
        actions = np.stack(actions)
        return actions

    def step(self, i_episode, state, action, reward, next_state, done):
        full_state = np.reshape(state, newshape=(-1))
        next_full_state = np.reshape(next_state, newshape=(-1))
        
        self.memory.add(state, full_state, action, reward, next_state, next_full_state, done)
        
        self.step_count = ( self.step_count + 1 ) % UPDATE_FREQ
        
        if len(self.memory) > BATCH_SIZE and i_episode > 500:
            for l_cnt in range(3):
                for agent in self.agents:
                    experiences = self.memory.sample()
                    self.learn(experiences, agent, GAMMA)

                for agent in self.agents:
                    agent.soft_update(agent.actor_local, agent.actor_target, TAU)
                    agent.soft_update(agent.critic_local, agent.critic_target, TAU)

    def learn(self, experiences, agent, GAMMA):
        states, full_states, actions, rewards, next_states, next_full_states, dones = experiences
                
      
        actor_target_actions = torch.zeros(actions.shape, dtype=torch.float, device=device)
        for agent_idx, agent_i in enumerate(self.agents):
            if agent == agent_i:
                agent_id = agent_idx
            agent_i_current_state = states[:,agent_idx]
            actor_target_actions[:,agent_idx,:] = agent_i.actor_target.forward(agent_i_current_state)
        actor_target_actions = actor_target_actions.view(BATCH_SIZE, -1)

        agent_state = states[:,agent_id,:]
        agent_action = actions[:,agent_id,:]
        agent_reward = rewards[:,agent_id].view(-1,1)
        agent_done = dones[:,agent_id].view(-1,1)
        

        actor_local_actions = actions.clone()
        actor_local_actions[:, agent_id, :] = agent.actor_local.forward(agent_state)
        actor_local_actions = actor_local_actions.view(BATCH_SIZE, -1)
        

        actions = actions.view(BATCH_SIZE, -1)
        

        
        agent_experience = (full_states, actions, actor_local_actions, actor_target_actions,
                            agent_state, agent_action, agent_reward, agent_done,
                            next_states, next_full_states)

        agent.learn(agent_experience, GAMMA)

    def save(self):
        for idx, agent in enumerate(self.agents):
            chk_actor_filename = 'checkpoint_agent{}_actor.pth'.format(idx)
            chk_critic_filename = 'checkpoint_critic{}_critic.pth'.format(idx)
            torch.save(agent.actor_local.state_dict(), chk_actor_filename)
            torch.save(agent.critic_local.state_dict(), chk_critic_filename)
        
class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=['state', 'full_state', 'action', 'reward', 'next_state', 'next_full_state','done'])
        self.seed = random.seed(seed)
    
    def add(self, state, full_state, action, reward, next_state, next_full_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, full_state, action, reward, next_state, next_full_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        full_states = torch.from_numpy(np.array([e.full_state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        next_full_states = torch.from_numpy(np.array([e.next_full_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, full_states, actions, rewards, next_states, next_full_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)    