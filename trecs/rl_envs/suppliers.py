from gym.spaces import Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers, parallel_to_aec
import functools
import matplotlib.pyplot as plt
import sys
import os
import pickle

def blockTqdm():
    sys.stderr = open(os.devnull, "w")

def enableTqdm():
    sys.stderr = sys.__stderr__

from trecs.components.users import Users
from trecs.components.items import Items
from trecs.models.popularity import PopularityRecommender
from trecs.models.content import ContentFiltering
from trecs.models.social import SocialFiltering
from trecs.models.mf import ImplicitMF
from trecs.models.random import RandomRecommender
from trecs.random import Generator
from trecs.metrics import InteractionMeasurement

from trecs_plus.metrics import RecommendationMetric
from trecs_plus.matrix_ops import *

models = {
  "PR": PopularityRecommender,
  "CF": ContentFiltering,
  "SF": SocialFiltering,
  "IMF": ImplicitMF,
  "Random": RandomRecommender
}

scores_fn = {
    "inner_product": inner_product,
    "cosine_similarity": cos_similarity,
    "euclidean_distance": euclidean_distance,
    "pearson_correlation": pearson_correlation
}

def env(**kwargs):
  env = parallel_env(**kwargs)
  env = wrappers.CaptureStdoutWrapper(env)
  env = wrappers.ClipOutOfBoundsWrapper(env)
  env = wrappers.OrderEnforcingWrapper(env)
  env = parallel_to_aec(env)
  return env

class parallel_env(ParallelEnv):
  metadata = {"render_modes": ["human"], "name": "suppliers_price_v0"}

  def __init__(self,
               rec_type = "random_recommender",
               num_suppliers = 20,
               num_users = 100,
               num_items = 500,
               num_attributes = 100,
               pretraining = 1000,
               simulation_steps = 100,
               steps_between_training = 10,
               max_preference_per_attribute = 5,
               train_between_steps = False,
               costs = None,
               score_fn_name = "inner_product",
               vertically_differentiate = False,
               price_into_observation = False,
               savepath = ""):
    super(parallel_env).__init__()

    self.rec_type = rec_type
    self.num_suppliers = num_suppliers
    if type(num_items) == list:
      self.num_items = num_items
    else:
      self.num_items = [num_items // self.num_suppliers for _ in range(self.num_suppliers)]
    self.tot_items = np.sum(self.num_items)
    self.num_users = num_users
    self.num_attributes = num_attributes
    self.pretraining = pretraining
    self.simulation_steps = simulation_steps
    self.steps_between_training = steps_between_training
    self.max_preference_per_attribute = max_preference_per_attribute
    self.train_between_steps = train_between_steps
    if costs is None:
      if vertically_differentiate:
        self.costs = np.random.random(self.tot_items)
      else:
        self.costs = np.zeros(self.tot_items, dtype = float)
    else:
      self.costs = costs
    self.score_fn = scores_fn[score_fn_name]
    self.vertically_differentiate = vertically_differentiate
    self.price_into_observation = price_into_observation
    self.savepath = savepath

    self.possible_agents = ["supplier_" + str(r) for r in range(self.num_suppliers)]
    self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
    self.returns = []

  @functools.lru_cache(maxsize = None)
  def observation_space(self, agent):
    # FOR EACH SUPPLIER, WE STORE THE NUMBER OF RECOMMENDATIONS AND INTERACTIONS FOR EACH OF ITS ITEMS OVER THE LAST PERIOD
    return Box(low = 0., high = self.steps_between_training * self.num_users, shape = (2 * (self.num_items[self.agent_name_mapping[agent]]) + int(self.price_into_observation) * (self.num_items[self.agent_name_mapping[agent]]),))

  @functools.lru_cache(maxsize = None)
  def action_space(self, agent):
    # FOR EACH SUPPLIER, ONE CONTINUOUS ACTION FOR EACH OF ITS ITEMS THAT IS THE PRICE INCREASE OVER THE COST FOR THE NEXT PERIOD
    return Box(low = 0., high = 1., shape = (self.num_items[self.agent_name_mapping[agent]],))

  def step(self, actions):
    if not actions:
      self.agents = []
      return {}, {}, {}, {}

    epsilons = np.hstack((list(actions.values())))
    nonrect_epsilons = self.__make_nonrect(epsilons)
    self.actions_hist.append(nonrect_epsilons)
    prices = self.costs + epsilons
    fn_with_costs = functools.partial(scores_with_cost, scores_fn = self.score_fn, item_costs = prices)
    self.rec.users.set_score_function(fn_with_costs)

    if self.rec_type == "IMF":
      blockTqdm()
      self.rec.run(timesteps = self.steps_between_training, train_between_steps = self.train_between_steps, reset_interactions = False)
      enableTqdm()
    else:
      self.rec.run(timesteps = self.steps_between_training, train_between_steps = self.train_between_steps, disable_tqdm = True)
    self.measures = self.rec.get_measurements()

    period_interactions = np.sum(self.measures["interaction_histogram"][-self.steps_between_training:], axis = 0)
    nonrect_interactions = self.__make_nonrect(period_interactions)
    # REWARD IS THE HOW MUCH EACH SUPPLIER GAINED OVER THE COST
    tmp_rewards = [np.sum(np.multiply(nonrect_interactions[i], epsilons[i])) for i in range(self.num_suppliers)]
    # OBSERVATION FOR EACH SUPPLIER IS THE NUMBER OF RECOMMENDATIONS AND INTERACTIONS FOR ITS ITEMS IN THE LAST PERIOD
    period_interactions = period_interactions / np.sum(period_interactions)
    nonrect_interactions = self.__make_nonrect(period_interactions)
    period_recommendations = np.sum(self.measures["recommendation_histogram"][-self.steps_between_training:], axis = 0)
    period_recommendations = period_recommendations / np.sum(period_recommendations)
    nonrect_recommendations = self.__make_nonrect(period_recommendations)
    tmp_observations = [np.concatenate([nonrect_interactions[i], nonrect_recommendations[i]]) for i in range(self.num_suppliers)]
    if self.price_into_observation:
      nonrect_prices = self.__make_nonrect(prices)
      tmp_observations = [np.concatenate([tmp_observations[i], nonrect_prices[i]]) for i in range(self.num_suppliers)]

    rewards = {agent: tmp_rewards[i] for i, agent in enumerate(self.agents)}
    self.returns[-1] = self.returns[-1] + tmp_rewards
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
    if self.tot_items == self.num_suppliers:
      firm_scores = np.zeros((self.num_users, self.num_suppliers))
    else:
      firm_scores = np.random.randint(0, self.max_preference_per_attribute, size = (self.num_users, self.num_suppliers))
    self.actual_user_representation = Users(
      actual_user_profiles = np.concatenate([np.random.randint(0, self.max_preference_per_attribute, size = (self.num_users, self.num_attributes)),
                                             firm_scores], axis = 1),
      num_users = self.num_users,
      size = (self.num_users, self.num_attributes + self.num_suppliers))
    # IF WE WANT VERTICAL DIFFERENTIATION, NUMBER OF 1s DEPENDS ON THE COST
    if self.vertically_differentiate:
      items_attributes = np.array([Generator().binomial(n = 1, p = self.costs[i], size = (self.num_attributes)) for i in range(self.tot_items)]).T
    # IF WE WANT ITEMS TO BE ONLY HORIZONTALLY DIFFERENTIATED, NUMBER OF 1s IS THE SAME
    else:
      num_ones = np.random.randint(self.num_attributes)
      base_vector = np.concatenate([np.ones(num_ones), np.zeros(self.num_attributes - num_ones)])
      items_attributes = np.array([np.random.permutation(base_vector) for _ in range(self.tot_items)]).T
    items_attributes = np.concatenate([items_attributes, np.repeat(np.eye(self.num_suppliers), self.num_items, axis = 1)], axis = 0)
    # WE HAD A ONE-HOT ENCODING FOR THE SUPPLIER ID
    self.actual_item_representation = Items(
      item_attributes = items_attributes,
      size = (self.num_attributes + self.num_suppliers, self.tot_items))

    # RS INITIALIZATION AND INITIAL TRAINING
    if self.rec_type == "content_filtering":
      self.rec = models[self.rec_type](num_attributes = self.num_attributes + self.num_suppliers,
                                       actual_user_representation = self.actual_user_representation,
                                       item_representation = self.actual_item_representation.get_component_state()["items"][0])
    else:
      self.rec = models[self.rec_type](actual_user_representation = self.actual_user_representation,
                                       actual_item_representation = self.actual_item_representation)

    # WE START WITH PRICE-TAKER SUPPLIERS: p=c
    fn_with_costs = functools.partial(scores_with_cost, scores_fn = self.score_fn, item_costs = self.costs)
    self.rec.users.set_score_function(fn_with_costs)

    self.rec.add_metrics(InteractionMeasurement(), RecommendationMetric())
    blockTqdm()
    self.rec.startup_and_train(timesteps = self.pretraining)
    enableTqdm()
    self.measures = self.rec.get_measurements()
    self.measures["interaction_histogram"][0] = np.zeros(self.tot_items) + 1e-32
    self.measures["recommendation_histogram"][0] = np.zeros(self.tot_items) + 1e-32
    self.actions_hist = []
    self.returns.append(np.zeros(len(self.possible_agents[:])))

    self.agents = self.possible_agents[:]
    self.n_steps = 0
    return {agent: np.zeros(2 * (self.num_items[self.agent_name_mapping[agent]]) + int(self.price_into_observation) * (self.num_items[self.agent_name_mapping[agent]])) for agent in self.agents}

  def render(self, mode = "human"):
    colors = plt.get_cmap("YlGnBu")(np.linspace(0, 1, len(self.possible_agents)))
    self.actions_hist = np.array(self.actions_hist, dtype = object)
    nonrect_costs = self.__make_nonrect(self.costs)
    for i, a in enumerate(self.possible_agents):
      ah = nonrect_costs[i] + np.reshape(np.stack(self.actions_hist[:, i]), (self.simulation_steps, self.num_items[self.agent_name_mapping[a]]))
      plt.plot(np.arange(self.simulation_steps), np.mean(ah, axis = -1), color = colors[i], label = a)
      #plt.fill_between(np.arange(self.simulation_steps), np.mean(ah, axis = -1) - np.std(ah, axis = -1), np.mean(ah, axis = -1) + np.std(ah, axis = -1), color = colors[i], alpha = 0.3)
    plt.title("Suppliers prices over simulation steps")
    plt.xlabel("Timestep")
    plt.ylabel(r"Price (cost + $\epsilon_i$)")
    if len(self.possible_agents) <= 5:
      plt.legend()
    plt.savefig(self.savepath + "/Prices_" + self.rec_type + "_" + ("No" if not self.train_between_steps else "") + "Retrain.pdf", bbox_inches = "tight")
    plt.clf()
    with open(self.savepath + "/Prices_" + self.rec_type + "_" + ("No" if not self.train_between_steps else "") + "Retrain.pkl", "wb") as f:
      pickle.dump(self.actions_hist, f)

    self.returns = np.array(self.returns)
    for i, a in enumerate(self.possible_agents):
      plt.plot(np.arange(self.returns.shape[0]), self.returns[:, i], color = colors[i], label = a)
    plt.title("RL return over training steps")
    plt.xlabel("Timestep")
    plt.ylabel("Return")
    if len(self.possible_agents) <= 5:
      plt.legend()
    plt.savefig(self.savepath + "/Returns_" + self.rec_type + "_" + ("No" if not self.train_between_steps else "") + "Retrain.pdf", bbox_inches = "tight")
    plt.clf()
    with open(self.savepath + "/Returns_" + self.rec_type + "_" + ("No" if not self.train_between_steps else "") + "Retrain.pkl", "wb") as f:
      pickle.dump(self.returns, f)

    interactions = self.measures["interaction_histogram"][-self.simulation_steps:]
    interactions[0] = np.zeros(self.tot_items) + 1e-32
    modified_ih = np.cumsum(interactions, axis = 0)
    percentages = np.reshape(modified_ih / np.sum(modified_ih, axis = 1)[:, None], (self.simulation_steps, self.tot_items))
    percentages = np.array([self.__make_nonrect(percentages[i]) for i in range(self.simulation_steps)], dtype = object)
    for i, a in enumerate(self.possible_agents):
      pctg = np.reshape(np.stack(percentages[:, i]), (self.simulation_steps, self.num_items[self.agent_name_mapping[a]]))
      plt.plot(np.arange(self.simulation_steps), np.sum(pctg, axis = -1), color = colors[i], label = a)
      #plt.fill_between(np.arange(self.simulation_steps), np.mean(pctg, axis = -1) - np.std(pctg, axis = -1), np.mean(pctg, axis = -1) + np.std(pctg, axis = -1), color = colors[i], alpha = 0.3)
    plt.axvline(self.pretraining, color = "k", ls = ":", lw = .5)
    plt.title("Suppliers shares over simulation steps")
    plt.xlabel("Timestep")
    plt.ylabel(r"Market shares %")
    if len(self.possible_agents) <= 5:
      plt.legend()
    plt.savefig(self.savepath + "/Shares_" + self.rec_type + "_" + ("No" if not self.train_between_steps else "") + "Retrain.pdf", bbox_inches = "tight")
    plt.clf()
    with open(self.savepath + "/Shares_" + self.rec_type + "_" + ("No" if not self.train_between_steps else "") + "Retrain.pkl", "wb") as f:
      pickle.dump(percentages, f)

    recommendations = self.measures["recommendation_histogram"][-self.simulation_steps:]
    recommendations[0] = np.zeros(self.tot_items) + 1e-32
    modified_rh = np.cumsum(recommendations, axis = 0)
    percentages = np.reshape(modified_rh / np.sum(modified_rh, axis = 1)[:, None], (self.simulation_steps, self.tot_items))
    percentages = np.array([self.__make_nonrect(percentages[i]) for i in range(self.simulation_steps)], dtype = object)
    for i, a in enumerate(self.possible_agents):
      pctg = np.reshape(np.stack(percentages[:, i]), (self.simulation_steps, self.num_items[self.agent_name_mapping[a]]))
      plt.plot(np.arange(self.simulation_steps), np.mean(pctg, axis = -1), color = colors[i], label = a)
      #plt.fill_between(np.arange(self.simulation_steps), np.mean(pctg, axis = -1) - np.std(pctg, axis = -1), np.mean(pctg, axis = -1) + np.std(pctg, axis = -1), color = colors[i], alpha = 0.3)
    plt.axvline(self.pretraining, color = "k", ls = ":", lw = .5)
    plt.title("Suppliers recommendations over simulation steps")
    plt.xlabel("Timestep")
    plt.ylabel(r"Recommendations %")
    if len(self.possible_agents) <= 5:
      plt.legend()
    plt.savefig(self.savepath + "/Recommendations_" + self.rec_type + "_" + ("No" if not self.train_between_steps else "") + "Retrain.pdf", bbox_inches = "tight")
    plt.clf()
    with open(self.savepath + "/Recommendations_" + self.rec_type + "_" + ("No" if not self.train_between_steps else "") + "Retrain.pkl", "wb") as f:
      pickle.dump(percentages, f)

    if self.vertically_differentiate:
      avg_prices = np.mean(np.hstack([np.reshape(np.stack(self.actions_hist[:, self.agent_name_mapping[a]]), (self.simulation_steps, self.num_items[self.agent_name_mapping[a]])) for a in self.possible_agents]), axis = 0)
      plt.scatter(self.costs, avg_prices, color = "C0", alpha = 0.5)
      plt.title("Items quality-average price ratio")
      plt.xlabel("Initial cost (proportional to quality)")
      plt.ylabel(r"Average price")

  def close(self):
    del self.rec
    plt.close("all")

  def __make_nonrect(self, arr):
    cum_items = np.insert(np.cumsum(self.num_items), 0, 0)
    nonrect_arr = []
    for i in range(self.num_suppliers):
      nonrect_arr.append(arr[cum_items[i]:cum_items[i + 1]])
    return nonrect_arr