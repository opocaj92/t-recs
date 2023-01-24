import numpy as np
from gym.spaces import Box, MultiDiscrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers, parallel_to_aec
import functools
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
               num_suppliers:int = 20,
               num_users:int = 100,
               num_items:Union[int, list] = 500,
               num_attributes:int = 100,
               attention_exp:int = 0,
               pretraining:int = 1000,
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
               rs_knows_prices:bool = False,
               discrete_actions:bool = False,
               savepath:str = ""):
    super(parallel_env).__init__()

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

    self.vertically_differentiate = vertically_differentiate
    self.costs = np.random.random(self.tot_items) if self.vertically_differentiate else np.zeros(self.tot_items, dtype = float)
    self.all_items_identical = all_items_identical
    self.attributes_into_observation = attributes_into_observation and not self.all_items_identical
    self.price_into_observation = price_into_observation
    self.rs_knows_prices = rs_knows_prices
    self.discrete_actions = discrete_actions
    self.savepath = savepath

    self.possible_agents = ["Supplier " + str(r) for r in range(self.num_suppliers)]
    self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
    self.returns_history = []
    self.scaled_returns_history = []
    self.interactions_history = []
    self.recommendations_history = []
    self.prices_history = []
    self.cum_items = np.insert(np.cumsum(self.num_items), 0, 0)

  @functools.lru_cache(maxsize = None)
  def observation_space(self, agent):
    # FOR EACH SUPPLIER, WE STORE THE NUMBER OF RECOMMENDATIONS AND INTERACTIONS FOR EACH OF ITS ITEMS OVER THE LAST PERIOD
    return Box(low = 0., high = 1., shape = (2 * self.num_items[self.agent_name_mapping[agent]] + int(self.price_into_observation) * self.num_items[self.agent_name_mapping[agent]] + int(self.attributes_into_observation) * (self.num_attributes + self.num_suppliers) * self.num_items[self.agent_name_mapping[agent]],))

  @functools.lru_cache(maxsize = None)
  def action_space(self, agent):
    # FOR EACH SUPPLIER, ONE CONTINUOUS ACTION FOR EACH OF ITS ITEMS THAT IS THE PRICE INCREASE OVER THE COST FOR THE NEXT PERIOD
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
    self.episode_actions.append(epsilons)
    self.prices_history[-1] = self.prices_history[-1] + epsilons
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
    self.measures["interaction_histogram"][0] = np.zeros(self.tot_items)
    self.measures["recommendation_histogram"][0] = np.zeros(self.tot_items)

    period_interactions = np.sum(self.measures["interaction_histogram"][-self.steps_between_training:], axis = 0)
    # REWARD IS HOW MUCH EACH SUPPLIER GAINED OVER THE COST
    individual_items_reward = [np.multiply(period_interactions[self.cum_items[i]:self.cum_items[i + 1]], epsilons[self.cum_items[i]:self.cum_items[i + 1]]) for i in range(self.num_suppliers)]
    tmp_rewards = [np.sum(individual_items_reward[i]) for i in range(self.num_suppliers)]
    self.scaled_returns_history[-1] = self.scaled_returns_history[-1] + [np.sum(np.divide(individual_items_reward[i], (self.scales[self.cum_items[i]:self.cum_items[i + 1]] + 1e-32))) for i in range(self.num_suppliers)]

    # OBSERVATION FOR EACH SUPPLIER IS THE NUMBER OF RECOMMENDATIONS AND INTERACTIONS FOR ITS ITEMS IN THE LAST PERIOD
    period_interactions = period_interactions / (self.num_users * self.steps_between_training)
    self.interactions_history[-1] = self.interactions_history[-1] + period_interactions
    period_recommendations = np.sum(self.measures["recommendation_histogram"][-self.steps_between_training:], axis = 0)
    period_recommendations = period_recommendations / (self.num_users * self.steps_between_training)
    self.recommendations_history[-1] = self.recommendations_history[-1] + period_recommendations
    tmp_observations = [np.concatenate([period_interactions[self.cum_items[i]:self.cum_items[i + 1]], period_recommendations[self.cum_items[i]:self.cum_items[i + 1]]]) for i in range(self.num_suppliers)]
    if self.price_into_observation:
      tmp_observations = [np.concatenate([tmp_observations[i], prices[self.cum_items[i]:self.cum_items[i + 1]]]) for i in range(self.num_suppliers)]
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
    # IF WE WANT MULTIPLE ITEMS, WE GIVE DIFFERENT SCORES TO FIRMS TOO
    firm_scores = np.zeros((self.num_users, self.num_suppliers)) if self.tot_items == self.num_suppliers else np.random.randint(0, self.max_preference_per_attribute, size = (self.num_users, self.num_suppliers))
    self.actual_user_representation = Users(
      actual_user_profiles = np.concatenate([np.random.randint(0, self.max_preference_per_attribute, size = (self.num_users, self.num_attributes)), firm_scores], axis = 1),
      num_users = self.num_users,
      size = (self.num_users, self.num_attributes + self.num_suppliers),
      attention_exp = self.attention_exp
    )

    # IF WE WANT VERTICAL DIFFERENTIATION, NUMBER OF 1s DEPENDS ON THE COST
    if self.vertically_differentiate:
      items_attributes = np.array([Generator().binomial(n = 1, p = self.costs[i], size = (self.num_attributes)) for i in range(self.tot_items)]).T
    # IF WE WANT ITEMS TO BE ONLY HORIZONTALLY DIFFERENTIATED, NUMBER OF 1s IS THE SAME
    else:
      num_ones = np.random.randint(self.num_attributes)
      base_vector = np.concatenate([np.ones(num_ones), np.zeros(self.num_attributes - num_ones)])
      if not self.all_items_identical:
        items_attributes = np.array([np.random.permutation(base_vector) for _ in range(self.tot_items)]).T
      else:
        shared_attr = np.random.permutation(base_vector)
        items_attributes = np.array([shared_attr for _ in range(self.tot_items)]).T
    items_attributes = np.concatenate([items_attributes, np.repeat(np.eye(self.num_suppliers), self.num_items, axis = 1)], axis = 0)
    # WE HAD A ONE-HOT ENCODING FOR THE SUPPLIER ID
    self.actual_item_representation = Items(
      item_attributes = items_attributes,
      size = (self.num_attributes + self.num_suppliers, self.tot_items)
    )

    # RS INITIALIZATION AND INITIAL TRAINING
    # WE START WITH PRICE-TAKER SUPPLIERS: p=c
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
    self.scales = self.scales / (np.max(self.scales) + 1e-32)

    if self.pretraining > 0:
      blockTqdm()
      self.rec.startup_and_train(timesteps = self.pretraining, no_new_items = True)
      enableTqdm()

    self.measures = self.rec.get_measurements()
    self.measures["interaction_histogram"][0] = np.zeros(self.tot_items)
    self.measures["recommendation_histogram"][0] = np.zeros(self.tot_items)
    self.episode_actions = []
    self.returns_history.append(np.zeros(len(self.possible_agents[:])))
    self.scaled_returns_history.append(np.zeros(len(self.possible_agents[:])))
    self.interactions_history.append(np.zeros(self.tot_items))
    self.recommendations_history.append(np.zeros(self.tot_items))
    self.prices_history.append(np.zeros(self.tot_items))

    self.agents = self.possible_agents[:]
    self.n_steps = 0
    tmp_observations = [np.zeros(2 * self.num_items[i]) for i in range(self.num_suppliers)]
    if self.price_into_observation:
      tmp_observations = [np.concatenate([tmp_observations[i], self.costs[self.cum_items[i]:self.cum_items[i + 1]]]) for i in range(self.num_suppliers)]
    if self.attributes_into_observation:
      self.attr = [items_attributes.T[self.cum_items[i]:self.cum_items[i + 1]].flatten() for i in range(self.num_suppliers)]
      tmp_observations = [np.concatenate([tmp_observations[i], self.attr[i]]) for i in range(self.num_suppliers)]
    return {agent: tmp_observations[i] for i, agent in enumerate(self.agents)}

  def render(self, mode = "simulation"):
    agents_colors = plt.get_cmap("tab20c")(np.linspace(0, 1, len(self.possible_agents)))
    items_colors = plt.get_cmap("tab20c")(np.linspace(0, 1, self.tot_items))

    if mode == "training":
      returns_history = np.array(self.returns_history)
      for i, a in enumerate(self.possible_agents):
        if self.n_steps == 0:
          plt.plot(np.arange(returns_history.shape[0] - 1), returns_history[:-1, i], color = agents_colors[i], label = a)
        else:
          plt.plot(np.arange(returns_history.shape[0]), returns_history[:, i], color = agents_colors[i], label = a)
      plt.title("RL return over training episodes")
      plt.xlabel("Episode")
      plt.ylabel("Return")
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "History_Returns.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "History_Returns.pkl"), "wb") as f:
        pickle.dump(returns_history, f)

      if not self.all_items_identical:
        scaled_returns_history = np.array(self.scaled_returns_history)
        for i, a in enumerate(self.possible_agents):
          if self.n_steps == 0:
            plt.plot(np.arange(scaled_returns_history.shape[0] - 1), scaled_returns_history[:-1, i], color = agents_colors[i], label = a)
          else:
            plt.plot(np.arange(scaled_returns_history.shape[0]), scaled_returns_history[:, i], color = agents_colors[i], label = a)
        plt.title("RL scaledreturn over training episodes")
        plt.xlabel("Episode")
        plt.ylabel("Scaled Return")
        plt.legend()
        plt.savefig(os.path.join(self.savepath, "History_Scaled_Returns.pdf"), bbox_inches = "tight")
        plt.clf()
        with open(os.path.join(self.savepath, "History_Scaled_Returns.pkl"), "wb") as f:
          pickle.dump(returns_history, f)

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
      plt.title("Average RL observations over training episodes")
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

      prices_history = np.array(self.prices_history)
      if self.n_steps != 0:
        prices_history[:-1] = prices_history[:-1] / self.simulation_steps
        prices_history[-1] /= self.n_steps
      else:
        prices_history = prices_history[:-1] / self.simulation_steps
      count = 0
      for i, a in enumerate(self.possible_agents):
        for j in range(self.num_items[i]):
          plt.plot(np.arange(prices_history.shape[0]), prices_history[:, count], color = items_colors[count], label = a + (("Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
          count += 1
      plt.title("Average RL price over training episodes")
      plt.xlabel("Episode")
      plt.ylabel("Avg. Price")
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "History_Prices.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "History_Prices.pkl"), "wb") as f:
        pickle.dump(self.prices_history, f)

    else:
      tot_steps = self.pretraining + 1 + self.simulation_steps * self.steps_between_training
      episode_actions = np.array(self.episode_actions)
      ah = self.costs + episode_actions
      ah = np.concatenate([np.repeat(np.expand_dims(self.costs, 0), self.pretraining + 1, axis = 0), np.repeat(ah, self.steps_between_training, axis = 0)], axis = 0)
      count = 0
      for i, a in enumerate(self.possible_agents):
        for j in range(self.num_items[i]):
          plt.plot(np.arange(tot_steps), ah[:, count], color = items_colors[count], label = a + ((" Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
          count += 1
      plt.title("Suppliers prices over simulation steps")
      plt.xlabel("Timestep")
      plt.ylabel(r"Price (cost + $\epsilon_i$)")
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "Prices.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "Prices.pkl"), "wb") as f:
        pickle.dump(episode_actions, f)

      interactions = self.measures["interaction_histogram"]
      interactions[0] = np.zeros(self.tot_items)
      modified_ih = np.cumsum(interactions, axis = 0)
      modified_ih[0] = modified_ih[0] + 1e-32
      windowed_modified_ih = np.array([modified_ih[t] - modified_ih[t - 10] if t - 10 >= self.pretraining else modified_ih[t] for t in range(modified_ih.shape[0])])
      percentages = windowed_modified_ih / np.sum(windowed_modified_ih, axis = 1, keepdims = True)
      count = 0
      for i, a in enumerate(self.possible_agents):
        for j in range(self.num_items[i]):
          plt.plot(np.arange(tot_steps), percentages[:, count], color = items_colors[count], label = a + ((" Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
          count += 1
      plt.axvline(self.pretraining, color = "k", ls = ":", lw = .5)
      plt.title("Suppliers interactions share over simulation steps")
      plt.xlabel("Timestep")
      plt.ylabel(r"Interactions share %")
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "Interactions.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "Interactions.pkl"), "wb") as f:
        pickle.dump(percentages, f)

      recommendations = self.measures["recommendation_histogram"]
      recommendations[0] = np.zeros(self.tot_items)
      modified_rh = np.cumsum(recommendations, axis = 0)
      modified_rh[0] = modified_rh[0] + 1e-32
      windowed_modified_rh = np.array([modified_rh[t] - modified_rh[t - 10] if t - 10 >= self.pretraining else modified_rh[t] for t in range(modified_rh.shape[0])])
      percentages = windowed_modified_rh / (np.sum(windowed_modified_rh, axis = 1, keepdims = True) / self.num_items_per_iter)
      count = 0
      for i, a in enumerate(self.possible_agents):
        for j in range(self.num_items[i]):
          plt.plot(np.arange(tot_steps), percentages[:, count], color = items_colors[count], label = a + ((" Item " + str(j + 1)) if self.num_items[i] > 1 else ""))
          count += 1
      plt.axvline(self.pretraining, color = "k", ls = ":", lw = .5)
      plt.title("Suppliers recommendations share over simulation steps")
      plt.xlabel("Timestep")
      plt.ylabel(r"Recommendations share %")
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
        plt.title("Items quality-average price ratio")
        plt.xlabel("Initial cost (proportional to quality)")
        plt.ylabel(r"Average price")
        plt.legend()
        plt.savefig(os.path.join(self.savepath, "Quality.pdf"), bbox_inches = "tight")
    plt.close("all")

  def close(self):
    del self.rec