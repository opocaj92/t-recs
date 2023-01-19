import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from stable_baselines3 import PPO
from pettingzoo.utils import aec_to_parallel
from pantheonrl.common.agents import OnPolicyAgent
from pantheonrl.envs.pettingzoo import PettingZooAECWrapper

from trecs.rl_envs import ma_suppliers_env

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

learning_rate = 0.0003
gamma = 0.9999
training_steps = 5000000
log_interval = 10

savepath = "Results/MARLSuppliers"
os.makedirs(savepath, exist_ok = True)
log_savepath = os.path.join(savepath, "logs")
os.makedirs(log_savepath, exist_ok = True)
model_savepath = os.path.join(savepath, "models")
os.makedirs(model_savepath, exist_ok = True)

env = ma_suppliers_env(
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
env = PettingZooAECWrapper(env)

partners = []
for i in range(env.n_players - 1):
   print("Partner Agent " + str(i + 1))
   partners.append(OnPolicyAgent(PPO("MlpPolicy",
                                     env.getDummyEnv(i),
                                     learning_rate = learning_rate,
                                     gamma= gamma,
                                     verbose = 1
                                     )))
   env.add_partner_agent(partners[-1], player_num = i + 1)
print("Ego Agent")
ego = PPO("MlpPolicy",
          env,
          learning_rate = learning_rate,
          gamma = gamma,
          verbose = 1,
          tensorboard_log = log_savepath
          )

print("----------------- TRAINING -----------------")
ego.learn(total_timesteps = training_steps, log_interval = log_interval)
for i in range(env.n_players - 1):
   partners[i].model.save(os.path.join(model_savepath, "partner_" + str(i + 1) + "_model"))
ego.save(os.path.join(model_savepath, "ego_model"))
env.base_env.render(mode = "training")

print("---------------- SIMULATION ----------------")
env = aec_to_parallel(env.base_env)
obs = env.reset()
env_done = False
while not env_done:
   actions = {agent: ego.predict(obs[agent], deterministic=True)[0] if i == 0 else partners[i - 1].model.predict(obs[agent], deterministic=True)[0] for i, agent in enumerate(env.agents)}
   obs, _, dones, _ = env.step(actions)
   env_done = list(dones.values())[0]
env.render(mode = "simulation")
env.close()

if num_suppliers == num_items and not price_into_observation and not attributes_into_observation:
   for p in range(num_suppliers):
      if p == 0:
         model = ego
         name = "ego"
      else:
         model = partners[p - 1].model
         name = "partner_" + str(p)
      all_possible_states = sum([[np.array([[i, j],]) / (steps_between_training * num_users) for i in range(steps_between_training * num_users + 1)] for j in range(steps_between_training * num_users + 1)], [])
      policy = np.array([model.predict(obs, deterministic = True)[0] for obs in all_possible_states]).flatten().reshape((steps_between_training * num_users + 1, steps_between_training * num_users + 1))

      if discrete_actions:
         policy = policy / 100
      for i in range(steps_between_training * num_users + 1):
         for j in range(steps_between_training * num_users + 1):
            if i > j:
               policy[i][j] = np.nan

      hm = sns.heatmap(policy, linewidths = 0.2, square = True, cmap = "YlOrRd")
      hm.set_xticks(range(0, steps_between_training * num_users + 1, 10))
      hm.set_xticklabels(f'{c:.1f}' for c in np.arange(0.0, 1.01, 0.1))
      hm.set_yticks(range(0, steps_between_training * num_users + 1, 10))
      hm.set_yticklabels(f'{c:.1f}' for c in np.arange(0.0, 1.01, 0.1))
      fig = hm.get_figure()
      fig.savefig(os.path.join(savepath, "Policy_Heatmap_" + name + ".pdf"), bbox_inches = "tight")
      plt.clf()

      all_possible_states = [np.array([[i, steps_between_training * num_users],]) / (steps_between_training * num_users) for i in range(steps_between_training * num_users + 1)]
      policy = np.array([model.predict(obs, deterministic = True)[0] for obs in all_possible_states]).flatten()
      if discrete_actions:
         policy = policy / 100

      plt.plot(np.arange(steps_between_training * num_users + 1) / (steps_between_training * num_users), policy, color = "C0")
      plt.title("Policy representation for all possible states")
      plt.xlabel("State")
      plt.ylabel(r"Price ($\epsilon_i$)")
      plt.savefig(os.path.join(savepath, "Policy_" + name + ".pdf"), bbox_inches = "tight")
      plt.clf()
   plt.close("all")