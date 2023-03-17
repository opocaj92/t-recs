from stable_baselines3 import PPO
import supersuit as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from trecs.rl_envs import suppliers_parallel_env

rec_type = "content_based"
num_suppliers = 5
num_fixed = 0
num_users = 100
num_items = num_suppliers + num_fixed
num_attributes = 100
attention_exp = 0.
drift = 0.
pretraining = 100
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
quality_into_observation = False
rs_knows_prices = False
users_know_prices = True
individual_rationality = False
discrete_actions = False

num_envs = 4
learning_rate = 0.0003
gamma = 0.9999
training_steps = 5000000
log_interval = 10
device = "cuda"
runs = 1

base_savepath = "Results/MARLSuppliers"
os.makedirs(base_savepath, exist_ok = True)

for r in range(runs):
   savepath = os.path.join(base_savepath, "run_" + str(r + 1)) if runs > 1 else base_savepath
   log_savepath = os.path.join(savepath, "logs")
   os.makedirs(log_savepath, exist_ok = True)

   env = suppliers_parallel_env(
      rec_type = rec_type,
      num_suppliers = num_suppliers,
      num_fixed = num_fixed,
      num_users = num_users,
      num_items = num_items,
      num_attributes = num_attributes,
      attention_exp = attention_exp,
      drift = drift,
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
      quality_into_observation = quality_into_observation,
      rs_knows_prices = rs_knows_prices,
      users_know_prices = users_know_prices,
      individual_rationality = individual_rationality,
      discrete_actions = discrete_actions,
      savepath = savepath
   )
   env = ss.pad_observations_v0(env)
   env = ss.pad_action_space_v0(env)
   vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
   vec_env = ss.concat_vec_envs_v1(vec_env, num_envs, num_cpus = 4, base_class = "stable_baselines3")

   model = PPO("MlpPolicy", vec_env, learning_rate = learning_rate, gamma = gamma, verbose = 1, device = device, tensorboard_log = log_savepath)
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

   if (num_suppliers + num_fixed) == num_items and not price_into_observation and not quality_into_observation and not attributes_into_observation:
      all_possible_states = np.stack(np.meshgrid(np.arange(0, steps_between_training * num_users + 1, steps_between_training), np.arange(0, steps_between_training * num_users + 1, steps_between_training), indexing = "ij"), axis = -1) / (steps_between_training * num_users)
      policy = np.array([model.predict(obs, deterministic = True)[0] for obs in all_possible_states]).flatten().reshape((num_users + 1, num_users + 1))

      if discrete_actions:
         policy = policy / 100
      for i in range(num_users + 1):
         for j in range(num_users + 1):
            if i > j:
               policy[i][j] = np.nan

      hm = sns.heatmap(policy, linewidths = 0.2, square = True, cmap = "YlOrRd")
      hm.set_xticks(range(0, steps_between_training * num_users + 1, 10))
      hm.set_xticklabels(f'{c:.1f}' for c in np.arange(0.0, 1.01, 0.1))
      hm.set_yticks(range(0, steps_between_training * num_users + 1, 10))
      hm.set_yticklabels(f'{c:.1f}' for c in np.arange(0.0, 1.01, 0.1))
      fig = hm.get_figure()
      fig.savefig(os.path.join(savepath, "Policy_Heatmap.pdf"), bbox_inches = "tight")
      plt.clf()

      plt.plot(np.arange(num_users + 1) / num_users, policy[:, -1], color = "C0")
      plt.xlabel("State")
      plt.ylabel(r"Price ($\epsilon_i$)")
      plt.savefig(os.path.join(savepath, "Policy.pdf"), bbox_inches = "tight")
      plt.clf()
      plt.close("all")