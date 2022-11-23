from stable_baselines3 import PPO
import supersuit as ss
import os

from trecs.rl_envs import suppliers_parallel_env

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
price_into_observation = False
rs_knows_prices = False

training_steps = 5000000
gamma = 0.9999
num_envs = 4

savepath = "Results/SuppliersPrice"
os.makedirs(savepath, exist_ok = True)

env = suppliers_parallel_env(
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
   price_into_observation = price_into_observation,
   rs_knows_prices = rs_knows_prices,
   savepath = savepath
)
env = ss.pad_observations_v0(env)
env = ss.pad_action_space_v0(env)
vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
vec_env = ss.concat_vec_envs_v1(vec_env, num_envs, num_cpus = 4, base_class = "stable_baselines3")

model = PPO("MlpPolicy", vec_env, gamma = gamma)
print("----------------- TRAINING -----------------")
model.learn(total_timesteps = training_steps)
model.save(savepath + "/suppliers_prices")
vec_env.render(mode = "training")
vec_env.close()

print("-------------- TRAINING DONE ---------------")
observations = env.reset()
env_done = False
while not env_done:
   actions = {agent: model.predict(observations[agent], deterministic = True)[0] for agent in env.agents}
   observations, _, dones, _ = env.step(actions)
   env_done = list(dones.values())[0]
env.render(mode = "simulation")
env.close()
