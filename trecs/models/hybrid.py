import numpy as np

from .recommender import BaseRecommender
from .content import ContentFiltering
from .mf import ImplicitMF
from trecs.validate import validate_user_item_inputs
from trecs.random import Generator

class HybridRecommender(BaseRecommender):
    def __init__(
        self,
        num_users = None,
        num_items = None,
        num_attributes = None,
        num_latent_factors = None,
        user_representation = None,
        item_representation = None,
        actual_user_representation = None,
        actual_item_representation = None,
        probabilistic_recommendations = False,
        seed = None,
        num_items_per_iter = 10,
        model_params = None,
        **kwargs
    ):
        num_users, num_items, num_attributes = validate_user_item_inputs(
            num_users,
            num_items,
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            100,
            1250,
            1000,
            num_attributes,
        )

        if user_representation is None:
            user_representation = np.zeros((num_users, num_attributes))
        if item_representation is None:
            item_representation = Generator(seed = seed).binomial(n = 1, p = 0.5, size = (num_attributes, num_items))
        if actual_item_representation is None:
            actual_item_representation = item_representation.copy()

        self.content_based = ContentFiltering(
            num_users,
            num_items,
            num_latent_factors,
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            probabilistic_recommendations,
            seed,
            num_items_per_iter,
            **kwargs
        )

        self.collaborative_filtering = ImplicitMF(
            num_users,
            num_items,
            num_attributes,
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            seed,
            num_items_per_iter,
            model_params,
            probabilistic_recommendations = probabilistic_recommendations,
            **kwargs
        )

        super().__init__(
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            num_users,
            num_items,
            num_items_per_iter,
            probabilistic_recommendations = probabilistic_recommendations,
            seed = seed,
            **kwargs
        )

    def initialize_user_scores(self):
        self.content_based.initialize_user_scores()
        self.collaborative_filtering.initialize_user_scores()
        super().initialize_user_scores()

    def train(self):
        self.content_based.train()
        self.collaborative_filtering.train()
        super().train()

    def generate_recommendations(self, k = 1, item_indices = None):
        return super().generate_recommendations(k, item_indices)

    def _update_internal_state(self, interactions):
        self.content_based._update_internal_state(interactions)
        self.collaborative_filtering._update_internal_state(interactions)

    def process_new_items(self, new_items):
        cb = self.content_based.process_new_items(new_items)
        cf = self.collaborative_filtering.process_new_items(new_items)
        return cb

    def process_new_users(self, new_users, **kwargs):
        cb = self.content_based.process_new_users(new_users, **kwargs)
        cf =  self.collaborative_filtering.process_new_users(new_users, **kwargs)
        return cb

    def create_and_process_items(self):
        new_items = self.creators.generate_items()

        self.num_items += new_items.shape[1]
        self.content_based.num_items += new_items.shape[1]
        self.collaborative_filtering.num_items += new_items.shape[1]

        self.items.append_new_items(new_items)
        self.content_based.items.append_new_items(new_items)
        self.collaborative_filtering.items.append_new_items(new_items)

        new_items_hat = self.process_new_items(new_items)

        self.items_hat.append_new_items(new_items_hat)
        self.content_based.items_hat.append_new_items(new_items_hat)
        self.collaborative_filtering.items_hat.append_new_items(new_items_hat)

        self.add_new_item_indices(new_items.shape[1])
        new_item_pred_score = self.score_fn(self.users_hat.value, new_items_hat)

        self.predicted_scores.append_item_scores(new_item_pred_score)
        self.content_based.predicted_scores.append_item_scores(new_item_pred_score)
        self.collaborative_filtering.predicted_scores.append_item_scores(new_item_pred_score)

        self.users.score_new_items(new_items)
        self.content_based.users.score_new_items(new_items)
        self.collaborative_filtering.users.score_new_items(new_items)

    def add_users(self, new_users, **kwargs):
        self.num_users += new_users.shape[0]

        self.users.append_new_users(new_users, self.items.value)
        self.content_based.users.append_new_users(new_users, self.items.value)
        self.collaborative_filtering.users.append_new_users(new_users, self.items.value)

        new_users_hat = self.process_new_users(new_users, **kwargs)

        self.users_hat.append_new_users(new_users_hat)
        self.content_based.users_hat.append_new_users(new_users_hat)
        self.collaborative_filtering.users_hat.append_new_users(new_users_hat)

        self.add_new_user_indices(new_users.shape[0])
        new_item_pred_score = self.score_fn(new_users_hat, self.items_hat.value)

        self.predicted_scores.append_user_scores(new_item_pred_score)
        self.content_based.predicted_scores.append_user_scores(new_item_pred_score)
        self.collaborative_filtering.predicted_scores.append_user_scores(new_item_pred_score)

    def set_num_items_per_iter(self, num_items_per_iter):
        self.content_based.set_num_items_per_iter(num_items_per_iter)
        self.collaborative_filtering.set_num_items_per_iter(num_items_per_iter)
        super().set_num_items_per_iter(num_items_per_iter)

    def add_new_item_indices(self, num_new_items):
        self.content_based.add_new_item_indices(num_new_items)
        self.collaborative_filtering.add_new_item_indices(num_new_items)
        super().add_new_item_indices(num_new_items)

    def add_new_user_indices(self, num_new_users):
        self.content_based.add_new_user_indices(num_new_users)
        self.collaborative_filtering.add_new_user_indices(num_new_users)
        super().add_new_user_indices(num_new_users)


class MixedHybrid(HybridRecommender):
    def __init__(
        self,
        num_users = None,
        num_items = None,
        num_attributes = None,
        num_latent_factors = None,
        user_representation = None,
        item_representation = None,
        actual_user_representation = None,
        actual_item_representation = None,
        probabilistic_recommendations = False,
        seed = None,
        num_items_per_iter = 10,
        model_params = None,
        **kwargs
    ):
        super().__init__(
            num_users,
            num_items,
            num_attributes,
            num_latent_factors,
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            probabilistic_recommendations,
            seed,
            num_items_per_iter,
            model_params,
            **kwargs
        )

    def generate_recommendations(self, k = 1, item_indices = None):
        cb_rec = self.content_based.generate_recommendations(k, item_indices)
        cf_rec = self.collaborative_filtering.generate_recommendations(k, item_indices)
        rec = []
        for u in range(self.num_users):
            rec.append(list(dict.fromkeys(np.ravel(np.column_stack((cb_rec[u], cf_rec[u]))).tolist()).keys())[:k // 2])
        return np.array(rec)


class EnsembleHybrid(HybridRecommender):
    def __init__(
        self,
        num_users = None,
        num_items = None,
        num_attributes = None,
        num_latent_factors = None,
        user_representation = None,
        item_representation = None,
        actual_user_representation = None,
        actual_item_representation = None,
        probabilistic_recommendations = False,
        seed = None,
        num_items_per_iter = 10,
        model_params = None,
        w_cb = 0.5,
        **kwargs
    ):
        self.w_cb = w_cb

        super().__init__(
            num_users,
            num_items,
            num_attributes,
            num_latent_factors,
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            probabilistic_recommendations,
            seed,
            num_items_per_iter,
            model_params,
            **kwargs
        )

    def train(self):
        super().train()
        self.predicted_scores.value = self.w_cb * self.content_based.predicted_scores.value + (1. - self.w_cb) * self.collaborative_filtering.predicted_scores.value
