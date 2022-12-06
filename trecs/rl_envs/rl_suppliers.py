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

    self.vertically_differentiate = vertically_differentiate
    self.costs = np.random.random(self.tot_items) if self.vertically_differentiate else np.zeros(self.tot_items, dtype = float)
    self.all_items_identical = all_items_identical
    self.attributes_into_observation = attributes_into_observation and not self.all_items_identical
    self.price_into_observation = price_into_observation
    self.rs_knows_prices = rs_knows_prices
    self.discrete_actions = discrete_actions
    self.savepath = savepath

    self.observation_space = Box(low = 0., high = 1., shape = (2 * self.num_items[0] + int(self.price_into_observation) * self.num_items[0] + int(self.attributes_into_observation) * (self.num_attributes + self.num_suppliers) * self.num_items[0],))
    self.action_space = MultiDiscrete([100 for _ in range(self.num_items[0])]) if self.discrete_actions else Box(low = 0., high = 1., shape = (self.num_items[0],))
    self.episodes_return = []
    self.episodes_interaction = []
    self.episodes_recommendation = []

  def step(self, action):
    epsilons = np.hstack([action, self.other_policies])
    nonrect_epsilons = self.__make_nonrect(epsilons)
    self.actions_hist.append(nonrect_epsilons)
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
    self.episodes_interaction[-1] += period_interactions[0]
    nonrect_interactions = self.__make_nonrect(period_interactions)
    # REWARD IS HOW MUCH EACH SUPPLIER GAINED OVER THE COST
    reward = np.sum(np.multiply(nonrect_interactions[0], epsilons[0]))

    # OBSERVATION FOR EACH SUPPLIER IS THE NUMBER OF RECOMMENDATIONS AND INTERACTIONS FOR ITS ITEMS IN THE LAST PERIOD
    period_interactions = period_interactions / np.sum(period_interactions)
    nonrect_interactions = self.__make_nonrect(period_interactions)
    period_recommendations = np.sum(self.measures["recommendation_histogram"][-self.steps_between_training:], axis = 0)
    self.episodes_recommendation[-1] += period_recommendations[0]
    period_recommendations = period_recommendations / np.sum(period_recommendations)
    nonrect_recommendations = self.__make_nonrect(period_recommendations)
    obs = np.concatenate([nonrect_interactions[0], nonrect_recommendations[0]])
    if self.price_into_observation:
      nonrect_prices = self.__make_nonrect(prices)[0]
      obs = np.concatenate([obs, nonrect_prices])
    if self.attributes_into_observation:
      obs = np.concatenate([obs, self.nonrect_attr])

    self.episodes_return[-1] += reward
    self.n_steps += 1
    env_done = self.n_steps >= self.simulation_steps
    done = env_done
    info = self.measures
    return obs, reward, done, info

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

    self.rec.add_metrics(
      InteractionMeasurement(),
      RecommendationMeasurement()
    )

    if self.pretraining > 0:
      blockTqdm()
      self.rec.startup_and_train(timesteps = self.pretraining, no_new_items = True)
      enableTqdm()

    self.measures = self.rec.get_measurements()
    self.measures["interaction_histogram"][0] = np.zeros(self.tot_items)
    self.measures["recommendation_histogram"][0] = np.zeros(self.tot_items)
    self.actions_hist = []
    self.episodes_return.append(0)
    self.episodes_interaction.append(0)
    self.episodes_recommendation.append(0)
    self.other_policies = np.random.random(size = (self.tot_items - self.num_items[0]))

    self.n_steps = 0
    obs = np.zeros(2 * self.num_items[0])
    if self.price_into_observation:
      nonrect_costs = self.__make_nonrect(self.costs)[0]
      obs = np.concatenate([obs, nonrect_costs])
    if self.attributes_into_observation:
      self.nonrect_attr = self.__make_nonrect(items_attributes.T)[0].flatten()
      obs = np.concatenate([obs, self.nonrect_attr])
    return obs

  def render(self, mode = "simulation"):
    if mode == "training":
      plt.plot(np.arange(len(self.episodes_return)), self.episodes_return, color = "C0", label = "Agent")
      plt.title("RL return over training episodes")
      plt.xlabel("Timestep")
      plt.ylabel("Episode return")
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "Returns.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "Returns.pkl"), "wb") as f:
        pickle.dump(self.episodes_return, f)

      fig, ax1 = plt.subplots()
      ax2 = ax1.twinx()
      ax1.plot(np.arange(len(self.episodes_interaction)), self.episodes_interaction, color = "C0", label = "Agent")
      ax2.plot(np.arange(len(self.episodes_recommendation)), self.episodes_recommendation, color = "C0", linestyle = "dashed")
      plt.title("RL observations over training episodes")
      ax1.set_xlabel("Timestep")
      ax1.set_ylabel("Episode interactions")
      ax2.set_ylabel("Episode recommendations")
      ax1.legend()
      plt.savefig(os.path.join(self.savepath, "Observations.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "Obs_Interactions.pkl"), "wb") as f:
        pickle.dump(self.episodes_interaction, f)
      with open(os.path.join(self.savepath, "Obs_Recommendations.pkl"), "wb") as f:
        pickle.dump(self.episodes_recommendation, f)

    else:
      self.actions_hist = np.array(self.actions_hist, dtype = object)
      nonrect_costs = self.__make_nonrect(self.costs)
      ah = nonrect_costs[0] + np.reshape(np.stack(self.actions_hist[:, 0]), (self.simulation_steps, self.num_items[0]))
      ah = np.concatenate([np.repeat(np.expand_dims(nonrect_costs[0], 0), self.pretraining + 1, axis = 0), np.repeat(ah, self.steps_between_training, axis = 0)], axis = 0)
      plt.plot(np.arange(self.pretraining + 1 + self.simulation_steps * self.steps_between_training), np.mean(ah, axis = -1), color = "C0", label = "Agent")
      # plt.fill_between(np.arange(self.pretraining + 1 + self.simulation_steps * self.steps_between_training), np.mean(ah, axis = -1) - np.std(ah, axis = -1), np.mean(ah, axis = -1) + np.std(ah, axis = -1), color = "C0", alpha = 0.3)
      plt.title("Suppliers prices over simulation steps")
      plt.xlabel("Timestep")
      plt.ylabel(r"Price (cost + $\epsilon_i$)")
      plt.legend()
      plt.savefig(os.path.join(self.savepath, "Prices.pdf"), bbox_inches = "tight")
      plt.clf()
      with open(os.path.join(self.savepath, "Prices.pkl"), "wb") as f:
        pickle.dump(self.actions_hist, f)

      interactions = self.measures["interaction_histogram"]
      interactions[0] = np.zeros(self.tot_items)
      modified_ih = np.cumsum(interactions, axis = 0)
      modified_ih[0] = modified_ih[0] + 1e-32
      windowed_modified_ih = np.array([modified_ih[t] - modified_ih[t - 10] if t - 10 >= 0 else modified_ih[t] for t in range(modified_ih.shape[0])])
      percentages = np.reshape(windowed_modified_ih / np.sum(windowed_modified_ih, axis = 1)[:, None], (self.pretraining + 1 + self.simulation_steps * self.steps_between_training, self.tot_items))
      percentages = np.array([self.__make_nonrect(percentages[i]) for i in range(self.pretraining + 1 + self.simulation_steps * self.steps_between_training)], dtype = object)
      pctg = np.reshape(np.stack(percentages[:, 0]), (self.pretraining + 1 + self.simulation_steps * self.steps_between_training, self.num_items[0]))
      plt.plot(np.arange(self.pretraining + 1 + self.simulation_steps * self.steps_between_training), np.mean(pctg, axis = -1), color = "C0", label = "Agent")
      # plt.fill_between(np.arange(self.pretraining + 1 + self.simulation_steps * self.steps_between_training), np.mean(pctg, axis = -1) - np.std(pctg, axis = -1), np.mean(pctg, axis = -1) + np.std(pctg, axis = -1), color = "C0", alpha = 0.3)
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
      windowed_modified_rh = np.array([modified_rh[t] - modified_rh[t - 10] if t - 10 >= 0 else modified_rh[t] for t in range(modified_rh.shape[0])])
      percentages = np.reshape(windowed_modified_rh / np.sum(windowed_modified_rh, axis = 1)[:, None], (self.pretraining + 1 + self.simulation_steps * self.steps_between_training, self.tot_items))
      percentages = np.array([self.__make_nonrect(percentages[i]) for i in range(self.pretraining + 1 + self.simulation_steps * self.steps_between_training)], dtype = object)
      pctg = np.reshape(np.stack(percentages[:, 0]), (self.pretraining + 1 + self.simulation_steps * self.steps_between_training, self.num_items[0]))
      plt.plot(np.arange(self.pretraining + 1 + self.simulation_steps * self.steps_between_training), np.mean(pctg, axis = -1), color = "C0", label = "Agent")
      # plt.fill_between(np.arange(self.pretraining + 1 + self.simulation_steps * self.steps_between_training), np.mean(pctg, axis = -1) - np.std(pctg, axis = -1), np.mean(pctg, axis = -1) + np.std(pctg, axis = -1), color = "C0"", alpha = 0.3)
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
        avg_prices = np.mean(np.reshape(np.stack(self.actions_hist[:, 0]), (self.simulation_steps * self.steps_between_training, self.num_items[0])), axis = 0)
        if self.num_items[0] != 1:
          std_prices = np.std(np.reshape(np.stack(self.actions_hist[:, 0]), (self.simulation_steps * self.steps_between_training, self.num_items[0])), axis = 0)
        else:
          std_prices = None
        plt.errorbar(self.costs[:self.num_items[0]], avg_prices, yerr = std_prices, fmt = "o", color = "C0", alpha = 0.5, capsize = 5, elinewidth = 1)
        plt.title("Items quality-average price ratio")
        plt.xlabel("Initial cost (proportional to quality)")
        plt.ylabel(r"Average price")
        plt.savefig(os.path.join(self.savepath, "Quality.pdf"), bbox_inches = "tight")
    plt.close("all")

  def close(self):
    del self.rec

  def __make_nonrect(self, arr):
    cum_items = np.insert(np.cumsum(self.num_items), 0, 0)
    nonrect_arr = []
    for i in range(self.num_suppliers):
      nonrect_arr.append(arr[cum_items[i]:cum_items[i + 1]])
    return nonrect_arr