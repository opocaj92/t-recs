import numpy as np
import gym
from gym.spaces import Box, MultiDiscrete
import matplotlib.pyplot as plt
import sys
import os
import pickle
from typing import Union

def blockTqdm():
    sys.stderr = open(os.devnull, "w")

def enableTqdm():
    sys.stderr = sys.__stderr__

from trecs.components import Users, Items
from trecs.models import PricedPopularityRecommender, PricedContentFiltering, PricedSocialFiltering, PricedImplicitMF, PricedRandomRecommender, PricedIdealRecommender
from trecs.random import Generator
from trecs.metrics import InteractionMeasurement, RecommendationMeasurement

models = {
  "popularity_recommender": PricedPopularityRecommender,
  "content_based": PricedContentFiltering,
  "social_filtering": PricedSocialFiltering,
  "collaborative_filtering": PricedImplicitMF,
  "random_recommender": PricedRandomRecommender,
  "ideal_recommender": PricedIdealRecommender
}

# WE ONLY CONTROL THE FIRST SUPPLIER!
class env(gym.Env):
  metadata = {"render_modes": ["simulation", "training"],
              "name": "rl_suppliers_v0"}

  def __init__(self,
               rec_type:str = "random_recommender",
               num_suppliers:int = 2,
               num_users:int = 100,
               num_items:Union[int, list] = 2,
               num_attributes:int = 100,
               attention_exp:int = 0,
               pretraining:int = 100,
               simulation_steps:int = 100,
               steps_between_training:int = 10,
               max_preference_per_attribute:int = 5,
               train_between_steps:bool = False,
               num_items_per_iter:int = 10,
               random_items_per_iter:int = 0,
               repeated_items:bool = True,
               probabilistic_recommendations:bool = True,
               vertically_differentiate:bool = False,
               all_items_identical:bool = False,
               attributes_into_observation:bool = True,
               price_into_observation:bool = False,
               quality_into_observation:bool = False,
               rs_knows_prices:bool = False,
               discrete_actions:bool = False,
               savepath:str = ""):
    super(env).__init__()

    self.rec_type = rec_type
    self.num_suppliers = num_suppliers
    self.num_items = num_items if type(num_items) == list else [num_items // self.num_suppliers for _ in range(self.num_suppliers)]
    self.tot_items = np.sum(self.num_items)
    self.num_users = num_users
    self.num_attributes = num_attributes
    self.attention_exp = attention_exp
    self.pretraining = pretraining
    self.simulation_steps = simulation_steps
    self.steps_between_training = steps_between_training
    self.max_preference_per_attribute = max_preference_per_attribute
    self.train_between_steps = train_between_steps
    self.num_items_per_iter = num_items_per_iter
    self.random_items_per_iter = random_items_per_iter
    self.repeated_items = repeated_items
    self.probabilistic_recommendations = probabilistic_recommendations

    self.all_items_identical = all_items_identical
    self.vertically_differentiate = vertically_differentiate and not self.all_items_identical
    self.attributes_into_observation = attributes_into_observation and not self.all_items_identical
    self.price_into_observation = price_into_observation
    self.quality_into_observation = quality_into_observation and not self.all_items_identical
    self.rs_knows_prices = rs_knows_prices
    self.discrete_actions = discrete_actions
    self.savepath = savepath
    os.makedirs(self.savepath, exist_ok = True)

    self.observation_space = Box(low = 0., high = 1., shape = (2 * self.num_items[0] + int(self.price_into_observation) * self.num_items[0] + int(self.quality_into_observation) * self.num_items[0] + int(self.attributes_into_observation) * (self.num_attributes + self.num_suppliers) * self.num_items[0],))
    self.action_space = MultiDiscrete([100 for _ in range(self.num_items[0])]) if self.discrete_actions else Box(low = 0., high = 1., shape = (self.num_items[0],))
    self.returns_history = []
    self.scaled_returns_history = []
    self.interactions_history = []
    self.recommendations_history = []
    self.prices_history = []
    self.other_policies = np.random.random(size = (self.tot_items - self.num_items[0]))
    self.others_history = []

  def step(self, action):
    if self.discrete_actions:
      action = action / 100
    epsilons = np.hstack([action, self.other_policies])
    self.episode_actions.append(action)
    self.prices_history[-1] = self.prices_history[-1] + action
    self.others_history[-1] = self.others_history[-1] + self.other_policies
    prices = self.costs + epsilons
    self.rec.set_items_price_for_users(prices)
    if self.rs_knows_prices:
      self.rec.set_items_price(prices)

    if self.rec_type == "collaborative_filtering":
      blockTqdm()
      self.rec.run(
        timesteps = self.steps_between_training,
        train_between_steps = self.train_between_steps,
        random_items_per_iter = self.random_items_per_iter,
        repeated_items = self.repeated_items,
        reset_interactions = False
      )
      enableTqdm()
    else:
      self.rec.run(
        timesteps = self.steps_between_training,
        train_between_steps = self.train_between_steps,
        random_items_per_iter = self.random_items_per_iter,
        repeated_items = self.repeated_items,
        disable_tqdm = True
      )

    self.measures = self.rec.get_measurements()
    period_interactions = np.sum(self.measures["interaction_histogram"][-self.steps_between_training:], axis = 0)
    individual_items_reward = np.multiply(period_interactions[:self.num_items[0]], action)
    reward = np.sum(individual_items_reward)
    self.scaled_returns_history[-1] += np.sum(np.divide(individual_items_reward, self.scales))

    period_interactions = period_interactions / (self.num_users * self.steps_between_training)
    self.interactions_history[-1] = self.interactions_history[-1] + period_interactions[:self.num_items[0]]
    period_recommendations = np.sum(self.measures["recommendation_histogram"][-self.steps_between_training:], axis = 0)
    period_recommendations = period_recommendations / (self.num_users * self.steps_between_training)
    self.recommendations_history[-1] = self.recommendations_history[-1] + period_recommendations[:self.num_items[0]]
    obs = np.concatenate([period_interactions[:self.num_items[0]], period_recommendations[:self.num_items[0]]])
    if self.price_into_observation:
      obs = np.concatenate([obs, prices[:self.num_items[0]]])
    if self.quality_into_observation:
      obs = np.concatenate([obs, self.qualities])
    if self.attributes_into_observation:
      obs = np.concatenate([obs, self.attr])

    self.returns_history[-1] += reward
    self.n_steps += 1
    env_done = self.n_steps >= self.simulation_steps
    done = env_done
    info = self.measures
    return obs, reward, done, info

  def reset(self):
    firm_scores = np.zeros((self.num_users, self.num_suppliers)) if self.tot_items == self.num_suppliers else np.random.randint(0, self.max_preference_per_attribute, size = (self.num_users, self.num_suppliers))
    self.actual_user_representation = Users(
      actual_user_profiles = np.concatenate([np.random.randint(0, self.max_preference_per_attribute, size = (self.num_users, self.num_attributes)), firm_scores], axis = 1),
      num_users = self.num_users,
      size = (self.num_users, self.num_attributes + self.num_suppliers),
      attention_exp = self.attention_exp
    )

    self.costs = np.random.random(self.tot_items) if self.vertically_differentiate else np.zeros(self.tot_items, dtype = float)
    if self.vertically_differentiate:
      items_attributes = np.array([Generator().binomial(n = 1, p = self.costs[i], size = (self.num_attributes)) for i in range(self.tot_items)]).T
    else:
      num_ones = np.random.randint(self.num_attributes)
      base_vector = np.concatenate([np.ones(num_ones), np.zeros(self.num_attributes - num_ones)])
      if not self.all_items_identical:
        items_attributes = np.array([np.random.permutation(base_vector) for _ in range(self.tot_items)]).T
      else:
        shared_attr = np.random.permutation(base_vector)
        items_attributes = np.array([shared_attr for _ in range(self.tot_items)]).T
    items_attributes = np.concatenate([items_attributes, np.repeat(np.eye(self.num_suppliers), self.num_items, axis = 1)], axis = 0)
    self.actual_item_representation = Items(
      item_attributes = items_attributes,
      size = (self.num_attributes + self.num_suppliers, self.tot_items)
    )

    if self.rec_type == "content_based" or self.rec_type == "ensemble_hybrid" or self.rec_type == "mixed_hybrid":
      self.rec = models[self.rec_type](num_attributes = self.num_attributes + self.num_suppliers,
                                       actual_user_representation = self.actual_user_representation,
                                       item_representation = self.actual_item_representation.get_component_state()["items"][0],
                                       actual_item_representation = self.actual_item_representation,
                                       prices = self.costs if self.rs_knows_prices else None,
                                       num_items_per_iter = self.num_items_per_iter,
                                       probabilistic_recommendations = self.probabilistic_recommendations
                                       )
    else:
      self.rec = models[self.rec_type](actual_user_representation = self.actual_user_representation,
                                       actual_item_representation = self.actual_item_representation,
                                       prices = self.costs if self.rs_knows_prices else None,
                                       num_items_per_iter = self.num_items_per_iter,
                                       probabilistic_recommendations = self.probabilistic_recommendations if self.rec_type != "random_recommender" else False
                                       )
    self.rec.set_items_price_for_users(self.costs)
    self.rec.add_metrics(InteractionMeasurement(), RecommendationMeasurement())

    self.scales = np.mean(self.rec.actual_user_item_scores, axis = 0)
    self.scales = self.scales[:self.num_items[0]] / np.max(self.scales)
    self.qualities = np.sum(items_attributes, axis = 1)
    self.qualities =  self.qualities[:self.num_items[0]] / (np.max(self.qualities) + 1e-32)

    if self.pretraining > 0:
      blockTqdm()
      self.rec.startup_and_train(timesteps = self.pretraining, no_new_items = True)
      enableTqdm()

    self.measures = self.rec.get_measurements()
    self.episode_actions = []
    self.returns_history.append(0.)
    self.scaled_returns_history.append(0.)
    self.interactions_history.append(np.zeros(self.num_items[0]))
    self.recommendations_history.append(np.zeros(self.num_items[0]))
    self.prices_history.append(np.zeros(self.num_items[0]))
    # self.other_policies = np.random.random(size = (self.tot_items - self.num_items[0]))7
    self.others_history.append(np.zeros(self.tot_items - self.num_items[0]))

    self.n_steps = 0
    obs = np.zeros(2 * self.num_items[0])
    if self.price_into_observation:
      obs = np.concatenate([obs, self.costs[:self.num_items[0]]])
    if self.quality_into_observation:
      obs = np.concatenate([obs, self.qualities])
    if self.attributes_into_observation:
      self.attr = items_attributes.T[:self.num_items[0]].flatten()
      obs = np.concatenate([obs, self.attr])
    return obs

  def render(self, mode = "simulation"):
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, self.tot_items))

    if mode == "training":
      if self.n_steps == 0:
        plt.plot(np.arange(len(self.returns_history) - 1), self.returns_history[:-1], color = colors[0], label = "Agent")
      else:
        plt.plot(np.arange(len(self.returns_history)), self.returns_history, color = colors[0], label = "Agent")
      plt.title("RL return over training episodes")
      plt.xlabel("Episode")
      plt.ylabel("Return")
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "History_Returns.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "History_Returns.pkl"), "wb") as f:
        pickle.dump(self.returns_history, f)

      if not self.all_items_identical:
        if self.n_steps == 0:
          plt.plot(np.arange(len(self.scaled_returns_history) - 1), self.scaled_returns_history[:-1], color = colors[0], label = "Agent")
        else:
          plt.plot(np.arange(len(self.scaled_returns_history)), self.scaled_returns_history, color = colors[0], label = "Agent")
        plt.title("RL scaled return over training episodes")
        plt.xlabel("Episode")
        plt.ylabel("Scaled Return")
        plt.legend()
        plt.savefig(os.path.join(self.savepath, "History_Scaled_Returns.pdf"), bbox_inches = "tight")
        plt.clf()
        with open(os.path.join(self.savepath, "History_Scaled_Returns.pkl"), "wb") as f:
          pickle.dump(self.returns_history, f)

      interactions_history = np.array(self.interactions_history)
      recommendations_history = np.array(self.recommendations_history)
      if self.n_steps != 0:
        interactions_history[:-1] = interactions_history[:-1] / self.simulation_steps
        interactions_history[-1] /= self.n_steps
        recommendations_history[:-1] = recommendations_history[:-1] / self.simulation_steps
        recommendations_history[-1] /= self.n_steps
      else:
        interactions_history = interactions_history[:-1] / self.simulation_steps
        recommendations_history = recommendations_history[:-1] / self.simulation_steps
      fig, ax1 = plt.subplots()
      ax2 = ax1.twinx()
      for i in range(self.num_items[0]):
        ax1.plot(np.arange(interactions_history.shape[0]), interactions_history[:, i], color = colors[i], label = ("Item " + str(i +1)) if self.num_items[0] > 1 else "Agent")
        ax2.plot(np.arange(recommendations_history.shape[0]), recommendations_history[:, i], color = colors[i], linestyle = "dashed")
      plt.title("Average RL observations over training episodes")
      plt.xlabel("Episode")
      ax1.set_ylabel("Avg. Interactions %")
      ax2.set_ylabel("Avg. Recommendations %")
      ax1.legend()
      plt.savefig(os.path.join(self.savepath, "History_Observations.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "History_Interactions.pkl"), "wb") as f:
        pickle.dump(self.interactions_history, f)
      with open(os.path.join(self.savepath, "History_Recommendations.pkl"), "wb") as f:
        pickle.dump(self.recommendations_history, f)

      prices_history = np.array(self.prices_history)
      others_history = np.array(self.others_history)
      if self.n_steps != 0:
        prices_history[:-1] = prices_history[:-1] / self.simulation_steps
        prices_history[-1] /= self.n_steps
        others_history[:-1] = others_history[:-1] / self.simulation_steps
        others_history[-1] /= self.n_steps
      else:
        prices_history = prices_history[:-1] / self.simulation_steps
        others_history = others_history[:-1] / self.simulation_steps
      for i in range(self.tot_items):
        if i < self.num_items[0]:
          plt.plot(np.arange(prices_history.shape[0]), prices_history[:, i], color = colors[i], label = ("Item " + str(i + 1)) if self.num_items[0] > 1 else "Agent")
        else:
          plt.plot(np.arange(others_history.shape[0]), others_history[:, i - self.num_items[0]], color = colors[i])
      plt.title("Average RL price over training episodes")
      plt.xlabel("Episode")
      plt.ylabel("Avg. Price")
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "History_Prices.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "History_Prices.pkl"), "wb") as f:
        pickle.dump(self.prices_history, f)
      with open(os.path.join(self.savepath, "History_Others.pkl"), "wb") as f:
        pickle.dump(self.others_history, f)

    else:
      tot_steps = self.simulation_steps * self.steps_between_training
      episode_actions = np.array(self.episode_actions)
      ah = np.repeat(self.costs[:self.num_items[0]] + episode_actions, self.steps_between_training, axis = 0)
      for i in range(self.tot_items):
        if i < self.num_items[0]:
          plt.plot(np.arange(1, tot_steps + 1), ah[:, i], color = colors[i], label = ("Item " + str(i +1)) if self.num_items[0] > 1 else "Agent")
        else:
          plt.hlines(self.other_policies[i - self.num_items[0]], xmin = 1, xmax = tot_steps,  color = colors[i])
      plt.title("Suppliers prices over simulation steps")
      plt.xlabel("Timestep")
      plt.ylabel(r"Price (cost + $\epsilon_i$)")
      plt.xticks([1] + list(range(0, tot_steps, 20))[1:])
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "Prices.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "Prices.pkl"), "wb") as f:
        pickle.dump(self.episode_actions, f)

      interactions = self.measures["interaction_histogram"][-tot_steps:]
      modified_ih = np.cumsum(interactions, axis = 0)
      modified_ih[0] = modified_ih[0] + 1e-32
      windowed_modified_ih = np.array([modified_ih[t] - modified_ih[t - 10] if t - 10 > 0 else modified_ih[t] for t in range(modified_ih.shape[0])])
      percentages = windowed_modified_ih / np.sum(windowed_modified_ih, axis = 1, keepdims = True)
      for i in range(self.num_items[0]):
        plt.plot(np.arange(1, tot_steps + 1), percentages[:, i], color = colors[i], label = ("Item " + str(i +1)) if self.num_items[0] > 1 else "Agent")
      plt.title("Suppliers interactions share over simulation steps")
      plt.xlabel("Timestep")
      plt.ylabel("Interactions share %")
      plt.xticks([1] + list(range(0, tot_steps, 20))[1:])
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "Interactions.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "Interactions.pkl"), "wb") as f:
        pickle.dump(percentages, f)

      recommendations = self.measures["recommendation_histogram"][-tot_steps:]
      modified_rh = np.cumsum(recommendations, axis = 0)
      modified_rh[0] = modified_rh[0] + 1e-32
      windowed_modified_rh = np.array([modified_rh[t] - modified_rh[t - 10] if t - 10 > 0 else modified_rh[t] for t in range(modified_rh.shape[0])])
      percentages = windowed_modified_rh / (np.sum(windowed_modified_rh, axis = 1, keepdims = True) / self.num_items_per_iter)
      for i in range(self.num_items[0]):
        plt.plot(np.arange(1, tot_steps + 1), percentages[:, i], color = colors[i], label = ("Item " + str(i +1)) if self.num_items[0] > 1 else "Agent")
      plt.title("Suppliers recommendations share over simulation steps")
      plt.xlabel("Timestep")
      plt.ylabel("Recommendations share %")
      plt.xticks([1] + list(range(0, tot_steps, 20))[1:])
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "Recommendations.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "Recommendations.pkl"), "wb") as f:
        pickle.dump(percentages, f)

      if self.vertically_differentiate:
        avg_prices = np.mean(episode_actions, axis = 0)
        std_prices = np.std(episode_actions, axis = 0)
        for i in range(self.num_items[0]):
          plt.errorbar(self.costs[i], avg_prices[i], yerr = std_prices[i], fmt = "o", color = colors[i], alpha = 0.5, capsize = 5, elinewidth = 1, linestyle = "", label = ("Item " + str(i +1)) if self.num_items[0] > 1 else "Agent")
        plt.title("Items quality-average price ratio")
        plt.xlabel("Cost (proportional to quality)")
        plt.ylabel("Avg. Price")
        plt.legend()
        plt.savefig(os.path.join(self.savepath, "Quality.pdf"), bbox_inches = "tight")
    plt.close("all")

  def close(self):
    debug_savepath = os.path.join(self.savepath, "debug")
    os.makedirs(debug_savepath, exist_ok = True)
    with open(os.path.join(debug_savepath, "predicted_user_profiles.pkl"), "wb") as f:
      pickle.dump(self.rec.predicted_user_profiles, f)
    with open(os.path.join(debug_savepath, "predicted_item_attributes.pkl"), "wb") as f:
      pickle.dump(self.rec.predicted_item_attributes, f)
    with open(os.path.join(debug_savepath, "actual_user_profiles.pkl"), "wb") as f:
      pickle.dump(self.rec.actual_user_profiles, f)
    with open(os.path.join(debug_savepath, "actual_item_attributes.pkl"), "wb") as f:
      pickle.dump(self.rec.actual_item_attributes, f)
    with open(os.path.join(debug_savepath, "actual_user_item_scores.pkl"), "wb") as f:
      pickle.dump(self.rec.actual_user_item_scores, f)
    with open(os.path.join(debug_savepath, "predicted_user_item_scoress.pkl"), "wb") as f:
      pickle.dump(self.rec.predicted_user_item_scores, f)