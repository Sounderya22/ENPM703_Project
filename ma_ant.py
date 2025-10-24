import os
os.environ["LLVM_SUPPRESS_OPTION_REGISTRATION_ERRORS"] = "1"
os.environ["MUJOCO_GL"] = "egl"  # or "egl"

import mujoco  # Import this before torch
import glfw    # Optional, depending on renderer

# Only after this:
import torch


import numpy as np
from gymnasium_robotics import mamujoco_v1

class MultiAgentAntController:
    """Controller for multi-agent Ant environment"""
    
    def __init__(self):
        self.phase_offset = 0.0
        
    def get_trotting_gait(self, step, agent_id):
        """Generate trotting gait patterns for Ant"""
        t = step * 0.1
        phase = agent_id * np.pi  # 180 degree phase difference between agents
        
        # Trotting gait: diagonal legs move together
        if agent_id == 0:  # Controls front-left and front-right legs
            action = np.array([
                0.4 * np.sin(t + phase),           # front-left hip
                0.6 * np.sin(t + phase + np.pi/2), # front-left ankle  
                0.4 * np.sin(t + phase + np.pi),   # front-right hip
                0.6 * np.sin(t + phase + 3*np.pi/2) # front-right ankle
            ])
        else:  # Controls hind-left and hind-right legs
            action = np.array([
                0.4 * np.sin(t + phase + np.pi),   # hind-left hip (opposite phase)
                0.6 * np.sin(t + phase + 3*np.pi/2), # hind-left ankle
                0.4 * np.sin(t + phase),           # hind-right hip  
                0.6 * np.sin(t + phase + np.pi/2)  # hind-right ankle
            ])
        
        return np.clip(action, -1, 1)
    
    def get_walking_gait(self, step, agent_id):
        """Generate walking gait patterns for Ant"""
        t = step * 0.08
        phase = agent_id * np.pi / 2  # 90 degree phase difference
        
        if agent_id == 0:  # Front legs
            action = np.array([
                0.3 * np.sin(t + phase),           # front-left hip
                0.4 * np.sin(t + phase + 0.5),     # front-left ankle
                0.3 * np.sin(t + phase + 1.0),     # front-right hip  
                0.4 * np.sin(t + phase + 1.5)      # front-right ankle
            ])
        else:  # Hind legs
            action = np.array([
                0.3 * np.sin(t + phase + 1.0),     # hind-left hip
                0.4 * np.sin(t + phase + 1.5),     # hind-left ankle
                0.3 * np.sin(t + phase),           # hind-right hip
                0.4 * np.sin(t + phase + 0.5)      # hind-right ankle
            ])
        
        return np.clip(action, -1, 1)
    
    def get_random_exploration(self, step, agent_id, noise_level=0.3):
        """Random exploration with some structure"""
        t = step * 0.05
        base_action = 0.2 * np.sin(t + agent_id * np.pi)
        random_component = noise_level * np.random.uniform(-1, 1, 4)
        
        action = base_action + random_component
        return np.clip(action, -1, 1)

def main():
    print("🤖 Multi-Agent Ant Control - Advanced Demo")
    print("=" * 60)
    
    # Initialize controller
    controller = MultiAgentAntController()
    
    try:
        # Create environment
        env = mamujoco_v1.parallel_env(
            scenario="Ant",
            agent_conf="2x4",  # 2 agents, 4 joints each
            agent_obsk=1,
            render_mode="human"
        )
        
        observations, infos = env.reset()
        print("✅ Environment loaded successfully!")
        print(f"Agents: {env.agents}")
        print(f"Observation space per agent: {env.observation_space(env.agents[0]).shape}")
        print(f"Action space per agent: {env.action_space(env.agents[0]).shape}")
        
        # Test different control strategies
        strategies = [
            ("Trotting Gait", controller.get_trotting_gait),
            ("Walking Gait", controller.get_walking_gait), 
            ("Random Exploration", lambda step, agent_id: controller.get_random_exploration(step, agent_id, 0.5))
        ]
        
        for strategy_name, control_func in strategies:
            print(f"\n🏃 Testing: {strategy_name}")
            print("-" * 40)
            
            # Reset environment for new strategy
            observations, infos = env.reset()
            
            total_rewards = {agent: 0.0 for agent in env.agents}
            steps_per_strategy = 150
            
            for step in range(steps_per_strategy):
                actions = {}
                for agent in env.agents:
                    agent_id = int(agent.split('_')[-1])  # Extract agent number
                    actions[agent] = control_func(step, agent_id)
                
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Accumulate rewards
                for agent in rewards:
                    total_rewards[agent] += rewards[agent]
                
                # Print progress
                if step % 30 == 0:
                    print(f"   Step {step:3d}: Current rewards {rewards}")
                    # Print some observation stats
                    obs_norms = {agent: np.linalg.norm(observations[agent]) for agent in env.agents}
                    print(f"          Observation norms: {obs_norms}")
                
                # Reset if environment terminates
                if any(terminations.values()) or any(truncations.values()):
                    print("   Environment terminated, resetting...")
                    observations, infos = env.reset()
            
            # Print strategy results
            print(f"   ✅ {strategy_name} completed!")
            print(f"   Total rewards: {total_rewards}")
            avg_reward = sum(total_rewards.values()) / len(total_rewards)
            print(f"   Average reward per agent: {avg_reward:.3f}")
        
        # Final demonstration with mixed strategies
        print(f"\n🎯 Final Demo: Mixed Strategies")
        print("-" * 40)
        
        observations, infos = env.reset()
        mixed_rewards = {agent: 0.0 for agent in env.agents}
        
        for step in range(200):
            actions = {}
            for agent in env.agents:
                agent_id = int(agent.split('_')[-1])
                
                # Switch strategies based on step count
                if step < 50:
                    actions[agent] = controller.get_trotting_gait(step, agent_id)
                elif step < 100:
                    actions[agent] = controller.get_walking_gait(step, agent_id)
                elif step < 150:
                    actions[agent] = controller.get_random_exploration(step, agent_id, 0.2)
                else:
                    # Final phase: combine strategies
                    trot = controller.get_trotting_gait(step, agent_id)
                    walk = controller.get_walking_gait(step, agent_id)
                    actions[agent] = 0.7 * trot + 0.3 * walk
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            for agent in rewards:
                mixed_rewards[agent] += rewards[agent]
            
            if step % 40 == 0:
                print(f"   Step {step:3d}: Mode {'Trot' if step < 50 else 'Walk' if step < 100 else 'Explore' if step < 150 else 'Mixed'}")
                print(f"          Rewards: {rewards}")
        
        print(f"   ✅ Mixed strategies completed!")
        print(f"   Total rewards: {mixed_rewards}")
        avg_mixed = sum(mixed_rewards.values()) / len(mixed_rewards)
        print(f"   Average reward: {avg_mixed:.3f}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("\n📊 Summary:")
    print("  - Multi-agent control is working with 2 agents")
    print("  - Each agent controls 4 joints independently")
    print("  - Different gait patterns were tested")
    print("  - The system can handle strategy switching")
    print("\n🚀 Next steps:")
    print("  1. Enable rendering: Change render_mode=None to 'human'")
    print("  2. Add more sophisticated controllers (RL, PID, etc.)")
    print("  3. Experiment with different agent configurations")
    print("  4. Implement communication between agents")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
