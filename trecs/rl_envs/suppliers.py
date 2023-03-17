import numpy as np
from gym.spaces import Box, MultiDiscrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers, parallel_to_aec
import functools
import matplotlib.pyplot as plt
import sys
import os
import pickle
import math
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

def env(**kwargs):
  env = parallel_env(**kwargs)
  env = parallel_to_aec(env)
  env = wrappers.CaptureStdoutWrapper(env)
  if "discrete_actions" in kwargs and not kwargs["discrete_actions"]:
    env = wrappers.ClipOutOfBoundsWrapper(env)
  env = wrappers.OrderEnforcingWrapper(env)
  return env

class parallel_env(ParallelEnv):
  metadata = {"render_modes": ["simulation", "training"],
              "name": "marl_suppliers_v0"}

  def __init__(self,
               rec_type:str = "random_recommender",
               num_suppliers:int = 2,
               num_fixed:int = 0,
               num_users:int = 100,
               num_items:Union[int, list] = 2,
               num_attributes:int = 100,
               attention_exp:float = 0.,
               drift:float = 0.,
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
               users_know_prices:bool = True,
               individual_rationality:bool = False,
               discrete_actions:bool = False,
               savepath:str = ""):
    super(parallel_env).__init__()

    self.rec_type = rec_type
    self.num_suppliers = num_suppliers
    self.num_fixed = num_fixed
    self.tot_agents = self.num_suppliers + self.num_fixed
    self.num_items = num_items if type(num_items) == list else [num_items // self.tot_agents for _ in range(self.tot_agents)]
    self.tot_items = np.sum(self.num_items)
    self.tot_supp_items = np.sum(self.num_items[:self.num_suppliers])
    self.cum_items = np.insert(np.cumsum(self.num_items), 0, 0)
    self.num_users = num_users
    self.num_attributes = num_attributes
    self.attention_exp = attention_exp
    self.drift = drift
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
    self.users_know_prices = users_know_prices
    self.individual_rationality = individual_rationality
    self.discrete_actions = discrete_actions
    self.savepath = savepath
    os.makedirs(self.savepath, exist_ok = True)

    self.possible_agents = ["Supplier " + str(r + 1) for r in range(self.num_suppliers)]
    self.agent_name_mapping = dict(zip(self.possible_agents, list(range(self.num_suppliers))))
    self.returns_history = []
    self.scaled_returns_history = []
    self.interactions_history = []
    self.recommendations_history = []
    self.prices_history = []
    if self.num_fixed > 0:
      self.fixed_policies = np.random.random(size = (self.tot_items - self.tot_supp_items))
      self.fixed_history = []

  @functools.lru_cache(maxsize = None)
  def observation_space(self, agent):
    return Box(low = 0., high = 1., shape = (2 * self.num_items[self.agent_name_mapping[agent]] + int(self.price_into_observation) * self.num_items[self.agent_name_mapping[agent]] + int(self.quality_into_observation) * self.num_items[self.agent_name_mapping[agent]] + int(self.attributes_into_observation) * (self.num_attributes + self.num_suppliers) * self.num_items[self.agent_name_mapping[agent]],))

  @functools.lru_cache(maxsize = None)
  def action_space(self, agent):
    if self.discrete_actions:
      return MultiDiscrete([100 for _ in range(self.num_items[self.agent_name_mapping[agent]])])
    else:
      return Box(low = 0., high = 1., shape = (self.num_items[self.agent_name_mapping[agent]],))

  def step(self, actions):
    if not actions:
      self.agents = []
      return {}, {}, {}, {}

    if self.discrete_actions:
      actions = {k: actions[k] / 100 for k in actions.keys()}
    epsilons = np.hstack(list(actions.values()))
    if self.num_fixed > 0:
      tot_epsilons = np.hstack([epsilons, self.fixed_policies])
    else:
      tot_epsilons = epsilons
    self.episode_actions.append(epsilons)
    self.prices_history[-1] = self.prices_history[-1] + epsilons
    if self.num_fixed > 0:
      self.fixed_history[-1] = self.fixed_history[-1] + self.fixed_policies
    prices = self.costs + tot_epsilons

    if self.users_know_prices:
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
    period_interactions = np.sum(self.measures["interaction_histogram"][-self.steps_between_training:], axis = 0)[:self.tot_supp_items]
    individual_items_reward = [np.multiply(period_interactions[self.cum_items[i]:self.cum_items[i + 1]], epsilons[self.cum_items[i]:self.cum_items[i + 1]]) for i in range(self.num_suppliers)]
    tmp_rewards = [np.sum(individual_items_reward[i]) for i in range(self.num_suppliers)]
    self.scaled_returns_history[-1] = self.scaled_returns_history[-1] + [np.sum(np.divide(individual_items_reward[i], (self.scales[self.cum_items[i]:self.cum_items[i + 1]] + 1e-32))) for i in range(self.num_suppliers)]

    period_interactions = period_interactions / (self.num_users * self.steps_between_training)
    self.interactions_history[-1] = self.interactions_history[-1] + period_interactions
    period_recommendations = np.sum(self.measures["recommendation_histogram"][-self.steps_between_training:], axis = 0)[:self.tot_supp_items]
    period_recommendations = period_recommendations / (self.num_users * self.steps_between_training)
    self.recommendations_history[-1] = self.recommendations_history[-1] + period_recommendations
    tmp_observations = [np.concatenate([period_interactions[self.cum_items[i]:self.cum_items[i + 1]], period_recommendations[self.cum_items[i]:self.cum_items[i + 1]]]) for i in range(self.num_suppliers)]
    if self.price_into_observation:
      tmp_observations = [np.concatenate([tmp_observations[i], prices[self.cum_items[i]:self.cum_items[i + 1]]]) for i in range(self.num_suppliers)]
    if self.quality_into_observation:
      tmp_observations = [np.concatenate([tmp_observations[i], self.qualities[self.cum_items[i]:self.cum_items[i + 1]]]) for i in range(self.num_suppliers)]
    if self.attributes_into_observation:
      tmp_observations = [np.concatenate([tmp_observations[i], self.attr[i]]) for i in range(self.num_suppliers)]

    rewards = {agent: tmp_rewards[i] for i, agent in enumerate(self.agents)}
    self.returns_history[-1] = self.returns_history[-1] + tmp_rewards
    observations = {agent: tmp_observations[i] for i, agent in enumerate(self.agents)}
    self.n_steps += 1

    env_done = self.n_steps >= self.simulation_steps
    dones = {agent: env_done for agent in self.agents}
    infos = {agent: self.measures for agent in self.agents}
    if env_done:
      self.agents = []
    return observations, rewards, dones, infos

  def reset(self):
    firm_scores = np.zeros((self.num_users, self.tot_agents)) if self.tot_items == self.tot_agents else np.random.randint(0, self.max_preference_per_attribute, size = (self.num_users, self.tot_agents))
    self.actual_user_representation = Users(
      actual_user_profiles = np.concatenate([np.random.randint(0, self.max_preference_per_attribute, size = (self.num_users, self.num_attributes)), firm_scores], axis = 1),
      num_users = self.num_users,
      size = (self.num_users, self.num_attributes + self.tot_agents),
      attention_exp = self.attention_exp,
      drift = self.drift,
      individual_rationality = self.individual_rationality
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
    items_attributes = np.concatenate([items_attributes, np.repeat(np.eye(self.tot_agents), self.num_items, axis = 1)], axis = 0)
    self.actual_item_representation = Items(
      item_attributes = items_attributes,
      size = (self.num_attributes + self.tot_agents, self.tot_items)
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
    if self.users_know_prices:
      self.rec.set_items_price_for_users(self.costs)
    self.rec.add_metrics(InteractionMeasurement(), RecommendationMeasurement())

    self.scales = np.mean(self.rec.actual_user_item_scores, axis = 0)
    self.scales = self.scales / (np.max(self.scales) + 1e-32)
    self.qualities = np.sum(items_attributes, axis = 1)
    self.qualities =  self.qualities / self.num_attributes

    if self.pretraining > 0:
      blockTqdm()
      self.rec.startup_and_train(timesteps = self.pretraining, no_new_items = True)
      enableTqdm()

    self.measures = self.rec.get_measurements()
    self.episode_actions = []
    self.returns_history.append(np.zeros(self.num_suppliers))
    self.scaled_returns_history.append(np.zeros(self.num_suppliers))
    self.interactions_history.append(np.zeros(self.tot_supp_items))
    self.recommendations_history.append(np.zeros(self.tot_supp_items))
    self.prices_history.append(np.zeros(self.tot_supp_items))
    if self.num_fixed > 0:
      # self.fixed_policies = np.random.random(size = (self.tot_items - self.tot_supp_items))
      self.fixed_history.append(np.zeros(self.tot_items - self.tot_supp_items))

    self.agents = self.possible_agents[:]
    self.n_steps = 0
    tmp_observations = [np.zeros(2 * self.num_items[i]) for i in range(self.num_suppliers)]
    if self.price_into_observation:
      tmp_observations = [np.concatenate([tmp_observations[i], self.costs[self.cum_items[i]:self.cum_items[i + 1]]]) for i in range(self.num_suppliers)]
    if self.quality_into_observation:
      tmp_observations = [np.concatenate([tmp_observations[i], self.qualities[self.cum_items[i]:self.cum_items[i + 1]]]) for i in range(self.num_suppliers)]
    if self.attributes_into_observation:
      self.attr = [items_attributes.T[self.cum_items[i]:self.cum_items[i + 1]].flatten() for i in range(self.num_suppliers)]
      tmp_observations = [np.concatenate([tmp_observations[i], self.attr[i]]) for i in range(self.num_suppliers)]
    return {agent: tmp_observations[i] for i, agent in enumerate(self.agents)}

  def render(self, mode = "simulation"):
    agents_colors = plt.get_cmap("tab20c")(np.linspace(0, 1, self.tot_agents))
    items_colors = plt.get_cmap("tab20c")(np.linspace(0, 1, self.tot_items))

    if mode == "training":
      returns_history = np.array(self.returns_history)
      for i, a in enumerate(self.possible_agents):
        if self.n_steps == 0:
          plt.plot(np.arange(returns_history.shape[0] - 1), returns_history[:-1, i], color = agents_colors[i], label = a)
        else:
          plt.plot(np.arange(returns_history.shape[0]), returns_history[:, i], color = agents_colors[i], label = a)
      plt.xlabel("Episode")
      plt.ylabel("Return")
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "History_Returns.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "History_Returns.pkl"), "wb") as f:
        pickle.dump(returns_history, f)

      if self.num_suppliers > 1:
        grid = plt.GridSpec(int(math.ceil(self.num_suppliers / 2)), 4, wspace = 0.8, hspace = 0.9)
        plots = sum([[plt.subplot(grid[i, :2]), plt.subplot(grid[i, 2:])] for i in range(int(math.ceil(self.num_suppliers / 2)) - 1)], [])
        if self.num_suppliers % 2 == 0:
          plots = plots + [plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, :2]), plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, 2:])]
        else:
          plots = plots + [plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, 1:3])]

        for i, a in enumerate(self.possible_agents):
          if self.n_steps == 0:
            plots[i].plot(np.arange(returns_history.shape[0] - 1), returns_history[:-1, i], color = agents_colors[i], label = a)
          else:
            plots[i].plot(np.arange(returns_history.shape[0]), returns_history[:, i], color = agents_colors[i], label = a)
          plots[i].legend()
        if self.num_suppliers % 2 == 0:
          plots[-1].set_xlabel("Episode")
          plots[-2].set_xlabel("Episode")
        else:
          plots[-1].set_xlabel("Episode")
        for i in range(int(math.ceil(self.num_suppliers / 2))):
          plots[2 * i].set_ylabel("Return")
        plt.savefig(os.path.join(self.savepath, "History_Returns_Individual.pdf"), bbox_inches = "tight")
        plt.clf()

      if not self.all_items_identical:
        scaled_returns_history = np.array(self.scaled_returns_history)
        for i, a in enumerate(self.possible_agents):
          if self.n_steps == 0:
            plt.plot(np.arange(scaled_returns_history.shape[0] - 1), scaled_returns_history[:-1, i], color = agents_colors[i], label = a)
          else:
            plt.plot(np.arange(scaled_returns_history.shape[0]), scaled_returns_history[:, i], color = agents_colors[i], label = a)
        plt.xlabel("Episode")
        plt.ylabel("Scaled Return")
        plt.legend()
        plt.savefig(os.path.join(self.savepath, "History_Scaled_Returns.pdf"), bbox_inches = "tight")
        plt.clf()
        with open(os.path.join(self.savepath, "History_Scaled_Returns.pkl"), "wb") as f:
          pickle.dump(returns_history, f)

        if self.num_suppliers > 1:
          grid = plt.GridSpec(int(math.ceil(self.num_suppliers / 2)), 4, wspace = 0.8, hspace = 0.9)
          plots = sum([[plt.subplot(grid[i, :2]), plt.subplot(grid[i, 2:])] for i in range(int(math.ceil(self.num_suppliers / 2)) - 1)], [])
          if self.num_suppliers % 2 == 0:
            plots = plots + [plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, :2]), plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, 2:])]
          else:
            plots = plots + [plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, 1:3])]

          for i, a in enumerate(self.possible_agents):
            if self.n_steps == 0:
              plots[i].plot(np.arange(scaled_returns_history.shape[0] - 1), scaled_returns_history[:-1, i], color = agents_colors[i], label = a)
            else:
              plots[i].plot(np.arange(scaled_returns_history.shape[0]), scaled_returns_history[:, i], color = agents_colors[i], label = a)
            plots[i].legend()
          if self.num_suppliers % 2 == 0:
            plots[-1].set_xlabel("Episode")
            plots[-2].set_xlabel("Episode")
          else:
            plots[-1].set_xlabel("Episode")
          for i in range(int(math.ceil(self.num_suppliers / 2))):
            plots[2 * i].set_ylabel("Scaled Return")
          plt.savefig(os.path.join(self.savepath, "History_Scaled_Returns_Individual.pdf"), bbox_inches = "tight")
          plt.clf()

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
      count = 0
      for i, a in enumerate(self.possible_agents):
        for j in range(self.num_items[i]):
          ax1.plot(np.arange(interactions_history.shape[0]), interactions_history[:, count], color = items_colors[count], label = a + ((" Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
          ax2.plot(np.arange(recommendations_history.shape[0]), recommendations_history[:, count], color = items_colors[count], linestyle = "dashed")
          count += 1
      ax1.set_xlabel("Episode")
      ax1.set_ylabel("Avg. Interactions %")
      ax2.set_ylabel("Avg. Recommendations %")
      ax1.legend()
      plt.savefig(os.path.join(self.savepath, "History_Observations.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "History_Interactions.pkl"), "wb") as f:
        pickle.dump(interactions_history, f)
      with open(os.path.join(self.savepath, "History_Recommendations.pkl"), "wb") as f:
        pickle.dump(recommendations_history, f)

      if self.num_suppliers > 1:
        grid = plt.GridSpec(int(math.ceil(self.num_suppliers / 2)), 4, wspace = 0.8, hspace = 0.9)
        plots = sum([[plt.subplot(grid[i, :2]), plt.subplot(grid[i, 2:])] for i in range(int(math.ceil(self.num_suppliers / 2)) - 1)], [])
        if self.num_suppliers % 2 == 0:
          plots = plots + [plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, :2]), plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, 2:])]
        else:
          plots = plots + [plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, 1:3])]

        count = 0
        for i, a in enumerate(self.possible_agents):
          for j in range(self.num_items[i]):
            plots[i].plot(np.arange(interactions_history.shape[0]), interactions_history[:, count], color = items_colors[count], label = a + ((" Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
            count += 1
          plots[i].legend()
        if self.num_suppliers % 2 == 0:
          plots[-1].set_xlabel("Episode")
          plots[-2].set_xlabel("Episode")
        else:
          plots[-1].set_xlabel("Episode")
        for i in range(int(math.ceil(self.num_suppliers / 2))):
          plots[2 * i].set_ylabel("Avg. Interactions %")
        plt.savefig(os.path.join(self.savepath, "History_Interactions_Individual.pdf"), bbox_inches = "tight")
        plt.clf()

        grid = plt.GridSpec(int(math.ceil(self.num_suppliers / 2)), 4, wspace = 0.8, hspace = 0.9)
        plots = sum([[plt.subplot(grid[i, :2]), plt.subplot(grid[i, 2:])] for i in range(int(math.ceil(self.num_suppliers / 2)) - 1)], [])
        if self.num_suppliers % 2 == 0:
          plots = plots + [plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, :2]), plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, 2:])]
        else:
          plots = plots + [plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, 1:3])]

        count = 0
        for i, a in enumerate(self.possible_agents):
          for j in range(self.num_items[i]):
            plots[i].plot(np.arange(recommendations_history.shape[0]), recommendations_history[:, count], color = items_colors[count], label = a + ((" Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
            count += 1
          plots[i].legend()
        if self.num_suppliers % 2 == 0:
          plots[-1].set_xlabel("Episode")
          plots[-2].set_xlabel("Episode")
        else:
          plots[-1].set_xlabel("Episode")
        for i in range(int(math.ceil(self.num_suppliers / 2))):
          plots[2 * i].set_ylabel("Avg. Recommendations %")
        plt.savefig(os.path.join(self.savepath, "History_Recommendations_Individual.pdf"), bbox_inches = "tight")
        plt.clf()

      prices_history = np.array(self.prices_history)
      if self.num_fixed > 0:
        fixed_history = np.array(self.fixed_history)
      if self.n_steps != 0:
        prices_history[:-1] = prices_history[:-1] / self.simulation_steps
        prices_history[-1] /= self.n_steps
        if self.num_fixed > 0:
          fixed_history[:-1] = fixed_history[:-1] / self.simulation_steps
          fixed_history[-1] /= self.n_steps
      else:
        prices_history = prices_history[:-1] / self.simulation_steps
        if self.num_fixed > 0:
          fixed_history[:-1] = fixed_history[:-1] / self.simulation_steps
      count = 0
      for i, a in enumerate(self.possible_agents):
        for j in range(self.num_items[i]):
          plt.plot(np.arange(prices_history.shape[0]), prices_history[:, count], color = items_colors[count], label = a + (("Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
          count += 1
      if self.num_fixed > 0:
        new_count = 0
        for i in range(self.num_fixed):
          for j in range(self.num_items[self.num_suppliers + i]):
            plt.plot(np.arange(fixed_history.shape[0]), fixed_history[:, new_count], color = items_colors[count], label = "Fixed " + str(i + 1) + (("Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
            count += 1
            new_count += 1
      plt.xlabel("Episode")
      plt.ylabel("Avg. Price")
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "History_Prices.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "History_Prices.pkl"), "wb") as f:
        pickle.dump(self.prices_history, f)
      if self.num_fixed > 0:
        with open(os.path.join(self.savepath, "History_Fixed.pkl"), "wb") as f:
          pickle.dump(self.fixed_history, f)

      if self.num_suppliers > 1:
        grid = plt.GridSpec(int(math.ceil(self.num_suppliers / 2)), 4, wspace = 0.8, hspace = 0.9)
        plots = sum([[plt.subplot(grid[i, :2]), plt.subplot(grid[i, 2:])] for i in range(int(math.ceil(self.num_suppliers / 2)) - 1)], [])
        if self.num_suppliers % 2 == 0:
          plots = plots + [plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, :2]), plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, 2:])]
        else:
          plots = plots + [plt.subplot(grid[int(math.ceil(self.num_suppliers / 2)) - 1, 1:3])]

        count = 0
        for i, a in enumerate(self.possible_agents):
          for j in range(self.num_items[i]):
            plots[i].plot(np.arange(prices_history.shape[0]), prices_history[:, count], color = items_colors[count], label = a + (("Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
            count += 1
          plots[i].legend()
        if self.num_suppliers % 2 == 0:
          plots[-1].set_xlabel("Episode")
          plots[-2].set_xlabel("Episode")
        else:
          plots[-1].set_xlabel("Episode")
        for i in range(int(math.ceil(self.num_suppliers / 2))):
          plots[2 * i].set_ylabel("Avg. Price")
        plt.savefig(os.path.join(self.savepath, "History_Prices_Individual.pdf"), bbox_inches = "tight")
        plt.clf()

    else:
      tot_steps = self.simulation_steps * self.steps_between_training
      episode_actions = np.array(self.episode_actions)
      ah = np.repeat(self.costs[:self.tot_supp_items] + episode_actions, self.steps_between_training, axis = 0)
      count = 0
      for i, a in enumerate(self.possible_agents):
        for j in range(self.num_items[i]):
          plt.plot(np.arange(1, tot_steps + 1), ah[:, count], color = items_colors[count], label = a + ((" Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
          count += 1
      if self.num_fixed > 0:
        new_count = 0
        for i in range(self.num_fixed):
          for j in range(self.num_items[self.num_suppliers + i]):
            plt.hlines(self.fixed_policies[new_count], xmin = 1, xmax = tot_steps, color = items_colors[count], label = "Fixed " + str(i + 1) + (("Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
            count += 1
            new_count += 1
      plt.xlabel("Timestep")
      plt.ylabel(r"Price (cost + $\epsilon_i$)")
      plt.xticks([1] + list(range(0, tot_steps, 20))[1:])
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "Prices.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "Prices.pkl"), "wb") as f:
        pickle.dump(episode_actions, f)

      interactions = self.measures["interaction_histogram"][-tot_steps:]
      modified_ih = np.cumsum(interactions, axis = 0)
      windowed_modified_ih = np.array([modified_ih[t] - modified_ih[t - 10] if t - 10 > 0 else modified_ih[t] for t in range(modified_ih.shape[0])])
      percentages = windowed_modified_ih / np.sum(windowed_modified_ih, axis = 1, keepdims = True)
      count = 0
      for i, a in enumerate(self.possible_agents):
        for j in range(self.num_items[i]):
          plt.plot(np.arange(1, tot_steps + 1), percentages[:, count], color = items_colors[count], label = a + ((" Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
          count += 1
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
      windowed_modified_rh = np.array([modified_rh[t] - modified_rh[t - 10] if t - 10 > 0 else modified_rh[t] for t in range(modified_rh.shape[0])])
      percentages = windowed_modified_rh / (np.sum(windowed_modified_rh, axis = 1, keepdims = True) / self.num_items_per_iter)
      count = 0
      for i, a in enumerate(self.possible_agents):
        for j in range(self.num_items[i]):
          plt.plot(np.arange(1, tot_steps + 1), percentages[:, count], color = items_colors[count], label = a + ((" Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
          count += 1
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
        count = 0
        for i, a in enumerate(self.possible_agents):
          for j in range(self.num_items[i]):
            plt.errorbar(self.costs[count], avg_prices[count], yerr = std_prices[count], fmt = "o", color = items_colors[count], alpha = 0.5, capsize = 5, elinewidth = 1, label = a + ((" Item " + str(j + 1)) if self.tot_items == self.num_suppliers else ""))
            count += 1
        plt.xlabel("Initial cost (proportional to quality)")
        plt.ylabel(r"Average price")
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