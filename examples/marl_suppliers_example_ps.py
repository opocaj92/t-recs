from stable_baselines3 import PPO
import supersuit as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from trecs.rl_envs import ma_suppliers_parallel_env

rec_type = "content_based"
num_suppliers = 5
num_users = 100
num_items = num_suppliers
num_attributes = 100
attention_exp = 0
pretraining = 1000
simulation_steps = 100
steps_between_training = 10
max_preference_per_attribute = 5
train_between_steps = True
num_items_per_iter = 3
random_items_per_iter = 0
repeated_items = True
probabilistic_recommendations = False

vertically_differentiate = False
all_items_identical = False
attributes_into_observation = False
price_into_observation = False
rs_knows_prices = False
discrete_actions = False

num_envs = 4
learning_rate = 0.0003
gamma = 0.9999
training_steps = 5000000
log_interval = 10
DEBUG = True

savepath = "Results/MARLSuppliers"
os.makedirs(savepath, exist_ok = True)
log_savepath = os.path.join(savepath, "logs")
os.makedirs(log_savepath, exist_ok = True)

env = ma_suppliers_parallel_env(
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
   attributes_into_observation = attributes_into_observation,
   price_into_observation = price_into_observation,
   rs_knows_prices = rs_knows_prices,
   discrete_actions = discrete_actions,
   savepath = savepath
)
env = ss.pad_observations_v0(env)
env = ss.pad_action_space_v0(env)
vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
vec_env = ss.concat_vec_envs_v1(vec_env, num_envs, num_cpus = 4, base_class = "stable_baselines3")

model = PPO("MlpPolicy", vec_env, learning_rate = learning_rate, gamma = gamma, verbose = 1, tensorboard_log = log_savepath)
print("----------------- TRAINING -----------------")
model.learn(total_timesteps = training_steps, log_interval = log_interval)
model.save(os.path.join(savepath, "suppliers_prices"))
vec_env.render(mode = "training")
vec_env.close()

print("---------------- SIMULATION ----------------")
observations = env.reset()
env_done = False
while not env_done:
   actions = {agent: model.predict(observations[agent], deterministic = True)[0] for agent in env.agents}
   observations, _, dones, _ = env.step(actions)
   env_done = list(dones.values())[0]
env.render(mode = "simulation")
env.close()

if num_suppliers == num_items and not price_into_observation and not attributes_into_observation:
   all_possible_states = sum([[np.array([[i, j],]) / (steps_between_training * num_users) for i in range(steps_between_training * num_users + 1)] for j in range(steps_between_training * num_users + 1)], [])
   policy = np.array([model.predict(obs, deterministic = True)[0] for obs in all_possible_states]).flatten()

   hm = sns.heatmap(policy.reshape((steps_between_training * num_users + 1, steps_between_training * num_users + 1)), linewidths = 0.2, square = True, cmap = "YlOrRd")
   fig = hm.get_figure()
   fig.savefig(os.path.join(savepath, "Policy_Heatmap.pdf"), bbox_inches = "tight")
   plt.clf()

   all_possible_states = [np.array([[i, steps_between_training * num_users],]) / (steps_between_training * num_users) for i in range(steps_between_training * num_users + 1)]
   policy = np.array([model.predict(obs, deterministic = True)[0] for obs in all_possible_states]).flatten()

   plt.plot(np.arange(len(all_possible_states)), policy, color = "C0")
   plt.title("Policy representation for all possible states")
   plt.xlabel("State")
   plt.ylabel(r"Price ($\epsilon_i$)")
   plt.savefig(os.path.join(savepath, "Policy.pdf"), bbox_inches = "tight")
   plt.clf()
   plt.close("all")

if DEBUG:
   print("------------------- DEBUG ------------------")
   debug_savepath = os.path.join(savepath, "DEBUG")
   os.makedirs(debug_savepath, exist_ok = True)

   env = ma_suppliers_parallel_env(
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
      attributes_into_observation = attributes_into_observation,
      price_into_observation = price_into_observation,
      rs_knows_prices = rs_knows_prices,
      discrete_actions = discrete_actions,
      savepath = debug_savepath
   )
   env = ss.pad_observations_v0(env)
   env = ss.pad_action_space_v0(env)

   policies = np.random.choice([0., 0.25, 0.5, 0.75, 1.], size = (num_suppliers))
   repetitions = num_items if type(num_items) == list else [num_items // num_suppliers for _ in range(num_suppliers)]

   _ = env.reset()
   env_done = False
   while not env_done:
      actions = {agent: np.array([policies[a] for _ in range(repetitions[a])]) for a, agent in enumerate(env.agents)}
      _, _, dones, _ = env.step(actions)
      env_done = list(dones.values())[0]
   env.render(mode = "simulation")
   env.close()