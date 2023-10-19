import time
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import policy_gradient

start = time.time()

# create cartpole environment
env = gym.make("CartPole-v1", render_mode=None)

# set parameters for offline dataset
# such as number of trajectories
# and max length of trajectory
num_traj = 100
max_traj_length = 1000
dataset = np.zeros((1, 5))

# load trained neural network
INPUT_DIM = env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = env.action_space.n
policy = policy_gradient.MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
policy.load_state_dict(torch.load(r"C:\Users\jesse\Downloads\STAT 548\code_and_output\policy_inf.pth"))
policy.eval()

# gather trajectories
# and save to transition dataset
# (numpy array)
for i in range(num_traj):
   done = False
   state = env.reset()[0]
   traj_index = 1
   while not done:
      new_record = np.zeros((1, 5))
      state = torch.from_numpy(state)
      with torch.no_grad():
         action_pred = policy(state)
         action_prob = F.softmax(action_pred, dim = -1)
      action = torch.argmax(action_prob, dim = -1)
      state, reward, terminated, truncated, info = env.step(action.item())
      traj_index += 1
      if traj_index == max_traj_length:
         print("Trajectory max length hit!")
         new_record[0, 0:4] = state
         new_record[0, 4] = True
         dataset = np.append(dataset, new_record, axis=0)
         break
      done = terminated or truncated
      new_record[0, 0:4] = state
      new_record[0, 4] = done
      dataset = np.append(dataset, new_record, axis=0)

env.close()

dataset = np.delete(dataset, (0), axis=0)

# save transition dataset
np.save(r"C:\Users\jesse\Downloads\STAT 548\code_and_output\dataset.npy", dataset)

end = time.time()
print("Generated offline RL dataset in", end - start, "seconds.")