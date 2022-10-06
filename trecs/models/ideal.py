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
        self.train()

    def train(self):
        if hasattr(self, 'users'):
            if self.is_verbose():
                self.log(
                    "System updates predicted scores given by users (rows) "
                    "to items (columns):\n"
                    f"{str(self.actual_user_item_scores)}"
                )
            if self.predicted_scores is None:
                self.predicted_scores = PredictedScores(self.actual_user_item_scores)
            else:
                self.predicted_scores.value = self.actual_user_item_scores
