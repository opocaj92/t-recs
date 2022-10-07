import numpy as np
from scipy.optimize import nnls
from lenskit.algorithms import als

from .recommender import BaseRecommender
from .popularity import PopularityRecommender
from .content import ContentFiltering
from .social import SocialFiltering
from .mf import ImplicitMF
from .random import RandomRecommender
import trecs.matrix_ops as mo
from trecs.validate import validate_user_item_inputs

class PricedBaseRecommender(BaseRecommender):
    def __init__(
        self,
        users_hat,
        items_hat,
        users,
        items,
        num_users,
        num_items,
        num_items_per_iter,
        prices = None,
        creators = None,
        probabilistic_recommendations = False,
        measurements = None,
        record_base_state = False,
        system_state = None,
        score_fn = mo.inner_product,
        interleaving_fn = None,
        verbose = False,
        seed = None,
    ):
        self.num_items = num_items
        self.set_items_price(prices)
        super().__init__(
            users_hat,
            items_hat,
            users,
            items,
            num_users,
            num_items,
            num_items_per_iter,
            creators,
            probabilistic_recommendations,
            measurements,
            record_base_state,
            system_state,
            score_fn,
            interleaving_fn,
            verbose,
            seed,
        )

    def set_items_price(self, prices):
        if prices is not None:
            if not isinstance(prices, (list, np.ndarray)):
                raise TypeError("prices must be a list or numpy.ndarray")
            prices = np.array(prices)
            if prices.shape[0] != self.num_items:
                raise TypeError("prices must the same number of items")
        else:
            prices = np.zeros(self.num_items, dtype = float)
        self.prices = prices

    def train(self):
        super().train()
        ps_min = np.min(self.predicted_scores)
        self.predicted_scores.value = ((self.predicted_scores.value - ps_min) / (np.max(self.predicted_scores.value - ps_min))) - self.prices

class PricedContentFiltering(ContentFiltering, PricedBaseRecommender):
    def __init__(
        self,
        num_users = None,
        num_items = None,
        num_attributes = None,
        user_representation = None,
        item_representation = None,
        actual_user_representation = None,
        actual_item_representation = None,
        prices = None,
        probabilistic_recommendations = False,
        seed = None,
        num_items_per_iter = 10,
        **kwargs
    ):
        _, num_items, _ = validate_user_item_inputs(
            num_users,
            num_items,
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            None,
            num_attributes = num_users,
            default_num_items = 1250,
            default_num_attributes = None,
        )
        self.num_items = num_items
        self.set_items_price(prices)
        ContentFiltering.__init__(
            self,
            num_users = num_users,
            num_items = num_items,
            num_attributes = num_attributes,
            user_representation = user_representation,
            item_representation = item_representation,
            actual_user_representation = actual_user_representation,
            actual_item_representation = actual_item_representation,
            probabilistic_recommendations = probabilistic_recommendations,
            seed = seed,
            num_items_per_iter = num_items_per_iter,
            **kwargs
        )

    def train(self):
        if self.all_interactions is not None and self.all_interactions.sum() > 0:
            for i in range(self.num_users):
                item_attr = np.array(mo.to_dense(self.predicted_item_attributes.T))
                user_interactions = self.all_interactions[i, :].toarray()[0, :]
                self.users_hat.value[i, :] = nnls(item_attr, user_interactions)[0]
        PricedBaseRecommender.train(self)


class PricedImplicitMF(ImplicitMF, PricedBaseRecommender):
    def __init__(
        self,
        num_users = None,
        num_items = None,
        num_latent_factors = None,
        user_representation = None,
        item_representation = None,
        actual_user_representation = None,
        actual_item_representation = None,
        prices = None,
        seed = None,
        num_items_per_iter = 10,
        model_params = None,
        **kwargs
    ):
        super().__init__(
            num_users = num_users,
            num_items = num_items,
            num_latent_factors = num_latent_factors,
            user_representation = user_representation,
            item_representation = item_representation,
            actual_user_representation = actual_user_representation,
            actual_item_representation = actual_item_representation,
            prices = prices,
            seed = seed,
            num_items_per_iter = num_items_per_iter,
            model_params = model_params,
            **kwargs
        )

    def train(self):
        if self.all_interactions.size > 0:
            self.model_params["features"] = self.num_latent_factors
            model = als.ImplicitMF(**self.model_params)
            model.fit(self.all_interactions)
            self.als_model = model
            user_index, item_index = list(set(model.user_index_)), list(set(model.item_index_))
            self.users_hat.value[user_index, :] = model.user_features_
            self.items_hat.value[:, item_index] = model.item_features_.T
        PricedBaseRecommender.train(self)

class PricedPopularityRecommender(PopularityRecommender, PricedBaseRecommender):
    def __init__(
            self,
            num_users = None,
            num_items = None,
            user_representation = None,
            item_representation = None,
            actual_user_representation = None,
            actual_item_representation = None,
            prices = None,
            verbose = False,
            num_items_per_iter = 10,
            **kwargs
    ):
        super().__init__(
            num_users = num_users,
            num_items = num_items,
            user_representation = user_representation,
            item_representation = item_representation,
            actual_user_representation = actual_user_representation,
            actual_item_representation = actual_item_representation,
            prices = prices,
            verbose = verbose,
            num_items_per_iter = num_items_per_iter,
            **kwargs
        )

class PricedRandomRecommender(RandomRecommender, PricedBaseRecommender):
    def __init__(
        self,
        num_users = None,
        num_items = None,
        actual_user_representation = None,
        actual_item_representation = None,
        prices = None,
        num_items_per_iter = 10,
        **kwargs
    ):
        super().__init__(
            num_users = num_users,
            num_items = num_items,
            actual_user_representation = actual_user_representation,
            actual_item_representation = actual_item_representation,
            prices = prices,
            num_items_per_iter = num_items_per_iter,
            **kwargs
        )

class PricedSocialFiltering(SocialFiltering, PricedBaseRecommender):
    def __init__(
            self,
            num_users = None,
            num_items = None,
            user_representation = None,
            item_representation = None,
            actual_user_representation = None,
            actual_item_representation = None,
            prices = None,
            probabilistic_recommendations = False,
            num_items_per_iter = 10,
            seed = None,
            **kwargs
    ):
        num_users, num_items, num_attributes = validate_user_item_inputs(
            num_users,
            num_items,
            user_representation,
            item_representation,
            actual_user_representation,
            actual_item_representation,
            None,
            num_attributes = num_users,
            default_num_items = 1250,
            default_num_attributes = None,
        )
        self.num_items = num_items
        self.set_items_price(prices)
        SocialFiltering.__init__(
            self,
            num_users = num_users,
            num_items = num_items,
            user_representation = user_representation,
            item_representation = item_representation,
            actual_user_representation = actual_user_representation,
            actual_item_representation = actual_item_representation,
            probabilistic_recommendations = probabilistic_recommendations,
            num_items_per_iter = num_items_per_iter,
            seed = seed,
            **kwargs
        )
