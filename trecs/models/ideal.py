from trecs.models import RandomRecommender
from trecs.components import PredictedScores

class IdealRecommender(RandomRecommender):
    def __init__(
        self,
        num_users = None,
        num_items = None,
        actual_user_representation = None,
        actual_item_representation = None,
        num_items_per_iter = 10,
        **kwargs
    ):
        super().__init__(
            num_users,
            num_items,
            actual_user_representation,
            actual_item_representation,
            num_items_per_iter,
            **kwargs
        )

    def train(self):
        predicted_scores = self.score_fn(
            self.predicted_user_profiles, self.predicted_item_attributes
        )
        if self.is_verbose():
            self.log(
                "System updates predicted scores given by users (rows) "
                "to items (columns):\n"
                f"{str(predicted_scores)}"
            )
        if self.predicted_scores is None:
            self.predicted_scores = PredictedScores(predicted_scores)
        else:
            self.predicted_scores.value = self.users.actual_user_scores.value
