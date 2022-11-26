from stable_baselines3 import PPO
import os

from trecs.rl_envs import suppliers_env

rec_type = "content_based"
num_suppliers = 2
num_users = 100
num_items = num_suppliers
num_attributes = 100
attention_exp = 0
pretraining = 1000
simulation_steps = 100
steps_between_training = 10
max_preference_per_attribute = 5
train_between_steps = True
num_items_per_iter = 2
random_items_per_iter = 0
repeated_items = True
probabilistic_recommendations = False
vertically_differentiate = False
all_items_identical = False
price_into_observation = False
rs_knows_prices = False

learning_rate = 0.0003
gamma = 0.9999
training_steps = 5000000
DEBUG = True

savepath = "Results/RLSuppliers"
os.makedirs(savepath, exist_ok = True)

env = suppliers_env(
   rec_type = rec_type,
   num_suppliers = num_suppliers,
   num_users = num_users,
   num_items = num_items,
   num_attributes = num_attributes,
   attention_exp = attention_exp,
   pretraining = pretraining,
   simulation_steps = simulation_steps,
   steps_between_training = steps_between_training,
   max_preference_per_attribute = max_preference_per_attribute,
   train_between_steps = train_between_steps,
   num_items_per_iter = num_items_per_iter,
   random_items_per_iter = random_items_per_iter,
   repeated_items = repeated_items,
   probabilistic_recommendations = probabilistic_recommendations,
   vertically_differentiate = vertically_differentiate,
   all_items_identical = all_items_identical,
   price_into_observation = price_into_observation,
   rs_knows_prices = rs_knows_prices,
   savepath = savepath
)

model = PPO("MlpPolicy", env, learning_rate = learning_rate, gamma = gamma)
print("----------------- TRAINING -----------------")
model.learn(total_timesteps = training_steps)
model.save(savepath + "/suppliers_prices")
env.render(mode = "training")
env.close()

print("-------------- TRAINING DONE ---------------")
obs = env.reset()
done = False
while not done:
   action = model.predict(obs, deterministic = True)[0]
   obs, _, done, _ = env.step(action)
env.render(mode = "simulation")
env.close()

if DEBUG:
   print("------------------- DEBUG ------------------")
   import numpy as np

   if num_suppliers == num_items and not price_into_observation:
      all_possible_states = sum([[np.array([[i, j],]) / (steps_between_training * num_users) for i in range(steps_between_training * num_users + 1)] for j in range(steps_between_training * num_users + 1)], [])
      policy = np.array([model.predict(obs, deterministic = True)[0] for obs in all_possible_states]).flatten()

      import matplotlib.pyplot as plt
      plt.plot(np.arange(len(all_possible_states)), policy, color = "C0")
      plt.title("Policy representation for all possible states")
      plt.xlabel("State")
      plt.ylabel(r"Price ($\epsilon_i$)")
      plt.savefig(savepath + "/Policy.pdf", bbox_inches = "tight")
      plt.clf()
      plt.close("all")
