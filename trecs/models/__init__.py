""" Various algorithms for recommender systems that use the same base """
from .recommender import BaseRecommender
from .bass import BassModel
from .content import ContentFiltering
from .mf import ImplicitMF
from .social import SocialFiltering
from .popularity import PopularityRecommender
from .random import RandomRecommender
from .hybrid import HybridRecommender, MixedHybrid, EnsembleHybrid
from .ideal import IdealRecommender
from .priced import (
	PricedBaseRecommender,
	PricedContentFiltering,
	PricedImplicitMF,
	PricedSocialFiltering,
	PricedPopularityRecommender,
	PricedRandomRecommender,
	PricedIdealRecommender
)
