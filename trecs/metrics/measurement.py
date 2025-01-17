"""
Set of various measurements that can be used to track outcomes of interest
throughout a simulation. Diagnostics may optionally be included to measure
ancillary information for each measurement, such as variance or
kurtosis.
"""
from abc import ABC, abstractmethod
import networkx as nx
from networkx import wiener_index
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau, rankdata, skew, kurtosis, shapiro
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from trecs.logging import VerboseMode
from trecs.base import (
    BaseObservable,
    register_observables,
)
from trecs.matrix_ops import cos_similarity


class Diagnostics:
    """
    Class to generate diagnostics on measurements.

    Attributes
    -----------

        measurement_diagnostics: pandas dataframe
            Dataframe containing diagnostics statistics at each timestep.

        last_observation: `None` or :obj:`numpy.ndarray`
            1-D numpy array containing the values for the specified metric
            at the most recent timestep.

        columns: list
            List of strings containing the column titles.

    """

    def __init__(
        self,
        columns=None,
    ):
        if columns is None:
            columns = [
                "mean",
                "std",
                "median",
                "min",
                "max",
                "skew",
                "kurtosis",
                "sw_stat",
                "sw_p",
                "n",
            ]
        self.columns = columns
        self.measurement_diagnostics = pd.DataFrame(columns=columns)

        self.last_observation = None

    def diagnose(self, observation):
        """
        Calculates diagnostic measurements on the latest observation
        from the recommender system. Also stores the current observation for
        later reference.

        Parameters
        -----------

            observation: :obj:`numpy.ndarray`
                1-D numpy array containing the values for the specified metric
                at this timestep.
        """

        # rudimentary type-checks
        if not isinstance(observation, np.ndarray):
            raise TypeError("Diagnostics can only be performed on numpy arrays")

        if observation.ndim != 1:
            raise ValueError("Diagnostics can only be performed on 1-d numpy arrays")

        self.last_observation = observation

        values = []
        sw_test = None
        col_to_fn = {
            "mean": np.mean,
            "std": np.std,
            "median": np.median,
            "min": np.min,
            "max": np.max,
            "skew": skew,
            "kurtosis": kurtosis,
        }
        for col in self.columns:
            if col in col_to_fn:
                values.append(col_to_fn[col](observation))
            elif col == "sw_stat":
                if sw_test is None:
                    sw_test = shapiro(observation)
                values.append(sw_test.statistic)
            elif col == "sw_p":
                if sw_test is None:
                    sw_test = shapiro(observation)
                if observation.size >= 5000:
                    sw_p = np.nan
                else:
                    sw_p = sw_test.pvalue
                values.append(sw_p)
            elif col == "n":
                values.append(observation.size)
        diagnostics = pd.Series(
            values,
            index=self.measurement_diagnostics.columns,
        )

        self.measurement_diagnostics = self.measurement_diagnostics.append(
            diagnostics, ignore_index=True
        )

    def hist(self, split_indices=None):
        """
        Draws a histogram of the most recent observation values.

        Parameters
        -----------
            split_indices: list or None
                Contains "splits" that determine which values
                to use for distinct histograms. For example,
                if there are 100 observation values and the
                split index is 50, then two separate histograms are
                created from the first 50 values and the second 50
                values.
        """
        if len(split_indices) > 4:
            raise RuntimeError("Too many split indices")
        colors = ["blue", "orange", "red", "yellow", "green"]
        if split_indices is not None and len(split_indices) > 0:
            splits = [0] + split_indices + [self.last_observation.size]
            for i in range(len(splits) - 1):
                values = self.last_observation[splits[i] : splits[i + 1]]
                plt.hist(values, alpha=0.7, color=colors[i])
        else:
            plt.hist(self.last_observation, bins="auto")
        plt.ylabel("observation count (total n={})".format(self.last_observation.size))

    def get_diagnostics(self):
        """
        Returns
        --------
        `pd.DataFrame`:
            Dataframe containing diagnostics statistics at each timestep.
        """
        return self.measurement_diagnostics


class Measurement(BaseObservable, VerboseMode, ABC):
    """
    Abstract observable class to store measurements.

    Parameters
    -----------

        verbose: bool, default False
            If ``True``, enables verbose mode. Disabled by default.

    Attributes
    -----------

        measurement_history: list
            List of measurements. A new element is added at each timestep.

        name: str
            Name of the measurement quantity.
    """

    def __init__(self, name, verbose=False):
        self.name = name
        VerboseMode.__init__(self, __name__.upper(), verbose)
        self.measurement_history = list()

    def get_measurement(self):
        """
        Returns measurements. See
        :func:`~base.base_components.BaseObservable.get_observable`
        for more details.

        Returns
        --------
        dict:
            Measurements
        """
        return self.get_observable(data=self.measurement_history)

    def observe(self, observation, copy=True):  # pylint: disable=arguments-differ
        """
        Stores measurements. It can be called by implementations to ensure
        consistency when storing different measurements.

        Parameters
        -----------

        observation: array_like or int or float or None
            Element that will be stored

        copy: bool, default True
            If ``True``, the function stores a copy of observation. Useful for
            :obj:`numpy.ndarray`.

        """
        # avoid in-place modification issues by copying lists and
        # numpy arrays
        if isinstance(observation, (list, np.ndarray)) and copy:
            to_append = np.copy(observation)
        else:
            to_append = observation
        self.measurement_history.append(to_append)

        # print(self.measurement_diagnostics.head())

    @abstractmethod
    def measure(self, recommender):
        """
        Function that should calculate some outcome of interest of the system
        at the current timestep
        """

    def get_timesteps(self):
        """
        Returns the number of measurements stored (which is equivalent to the
        number of timesteps that the system has been measuring).

        Returns
        --------
        int:
            Length of ``measurement_history``
        """
        return len(self.measurement_history)


class MeasurementModule:  # pylint: disable=too-few-public-methods
    """
    Mixin for observers of :class:`Measurement` observables. Implements the
    `Observer design pattern`_.

    .. _`Observer design pattern`: https://en.wikipedia.org/wiki/Observer_pattern

    This mixin allows the system to monitor metrics. That is, at each timestep,
    an element will be added to the
    :attr:`~metrics.measurement.Measurement.measurement_history` lists of each
    metric that the system is monitoring.

    Attributes
    ------------

        metrics: list
            List of metrics that the system will monitor.

    """

    def __init__(self):
        self.metrics = list()

    def add_metrics(self, *args):
        """
        Adds metrics to the :attr:`metrics` list. This allows the system to
        monitor these metrics.

        Parameters
        -----------

            args: :class:`~metrics.measurement.Measurement`
                Accepts a variable number of metrics that inherits from
                :class:`~metrics.measurement.Measurement`
        """
        register_observables(
            observer=self.metrics, observables=list(args), observable_type=Measurement
        )
        # after adding a new metric, we always perform an initial measurement
        for metric in args:
            metric.measure(self)

    def measure_content(self):
        """
        Calls method in the :class:`Measurements` module to record metrics.
        For more details, see the :class:`Measurements` class and its measure
        method.
        """
        for metric in self.metrics:
            metric.measure(self)


class InteractionMeasurement(Measurement):
    """
    Keeps track of the interactions between users and items.

    Specifically, at each timestep, it stores a histogram of length
    :math:`|I|`, where element :math:`i` is the number of interactions
    received by item :math:`i`.

    Parameters
    -----------

        verbose: bool, default False
            If ``True``, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str, default ``"interaction_histogram"``
            Name of the measurement component.
    """

    def __init__(self, name="interaction_histogram", verbose=False):
        Measurement.__init__(self, name, verbose)

    @staticmethod
    def _generate_interaction_histogram(interactions, num_users, num_items):
        """
        Generates a histogram of the number of interactions per item at the
        given timestep.

        Parameters
        -----------
            interactions : :obj:`numpy.ndarray`
                Array of user interactions.

            num_users : int
                Number of users in the system

            num_items : int
                Number of items in the system

        Returns
        ---------
            :obj:`numpy.ndarray`:
                Histogram of the number of interactions aggregated by items at the given timestep.
        """
        histogram = np.zeros(num_items)
        interactions_idx = np.argwhere(interactions != -1).flatten()
        np.add.at(histogram, interactions[interactions_idx], 1)
        # Check that there's one interaction per user
        if histogram.sum() != interactions_idx.shape[0]:
            raise ValueError("The sum of interactions must be equal to the number of users")
        return histogram

    def measure(self, recommender):
        """
        Measures and stores a histogram of the number of interactions per
        item at the given timestep.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from :class:`~models.recommender.BaseRecommender`.
        """
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return

        histogram = self._generate_interaction_histogram(
            recommender.interactions, recommender.num_users, recommender.num_items
        )
        self.observe(histogram, copy=True)


class InteractionSimilarity(Measurement, Diagnostics):
    """
    Keeps track of the average Jaccard similarity between interactions with items
    between pairs of users at each timestep. The pairs of users must be passed
    in by the user.

    Parameters
    -----------
        pairs: iterable of tuples
            Contains tuples representing each pair of users. Each user should
            be represented as an index into the user profiles matrix.

        verbose: bool, default False
            If ``True``, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str, default ``"interaction_similarity"``
            Name of the measurement component.
    """

    def __init__(
        self, pairs, name="interaction_similarity", verbose=False, diagnostics=False, **kwargs
    ):
        self.pairs = pairs
        # will eventually be a matrix where each row corresponds to 1 user
        self.interaction_hist = None
        self.diagnostics = diagnostics
        Measurement.__init__(self, name, verbose)

        if diagnostics:
            Diagnostics.__init__(self, **kwargs)

    def measure(self, recommender):
        """
        Measures the average Jaccard index of items that pairs of users have interacted
        with in the system. Intuitively, a higher average Jaccard index corresponds to
        increasing "homogenization" in that user behavior is becoming more and more
        similar (i.e., users have all interacted with the same items).

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        similarity = 0
        interactions = recommender.interactions
        if interactions.size == 0:
            self.observe(None)  # no interactions yet
            return

        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack(
                [self.interaction_hist, interactions.reshape((-1, 1))]
            )

        pair_sim = []
        for pair in self.pairs:
            itemset_1 = set(self.interaction_hist[pair[0], :])
            itemset_2 = set(self.interaction_hist[pair[1], :])
            common = len(itemset_1.intersection(itemset_2))
            union = len(itemset_1.union(itemset_2))
            similarity += common / union / len(self.pairs)

            if self.diagnostics:
                pair_sim.append(common / union)

        self.observe(similarity)

        if self.diagnostics:
            self.diagnose(np.array(pair_sim))


class RecSimilarity(Measurement):
    """
    Keeps track of the average Jaccard similarity between items seen by pairs
    of users at each timestep. The pairs of users must be passed in by the
    user.

    Parameters
    -----------
        pairs: iterable of tuples
            Contains tuples representing each pair of users. Each user should
            be represented as an index into the user profiles matrix.

        verbose: bool, default False
            If ``True``, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str, default ``"rec_similarity"``
            Name of the measurement component.
    """

    def __init__(self, pairs, name="rec_similarity", verbose=False):
        self.pairs = pairs
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        """
        Measures the average Jaccard index of items shown to pairs of users in
        the system. Intuitively, a higher average Jaccard index corresponds to
        increasing "homogenization" in that the recommender system is starting
        to treat each user the same way (i.e., show them the same items).

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        similarity = 0
        items_shown = recommender.items_shown
        if items_shown.size == 0:
            # at the beginning of the simulation, there are no recommendations yet
            self.observe(None)
            return

        for pair in self.pairs:
            itemset_1 = set(items_shown[pair[0], :])
            itemset_2 = set(items_shown[pair[1], :])
            common = len(itemset_1.intersection(itemset_2))
            union = len(itemset_1.union(itemset_2))
            similarity += common / union / len(self.pairs)
        self.observe(similarity)


class InteractionSpread(InteractionMeasurement):
    """
    Measures the diversity of the interactions between users and items.

    Specifically, at each timestep, it measures whether interactions are spread
    among many items or only a few items.

    This class inherits from :class:`.InteractionMeasurement`.

    Parameters
    -----------

        verbose: bool, default False
            If ``True``, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by InteractionMeasurement: :class:`.InteractionMeasurement`

        name: str, default ``"interaction_spread"``
            Name of the measurement component.

        _old_histogram: None, list, array_like
            A copy of the histogram at the previous timestep.
    """

    def __init__(self, verbose=False):
        self.histogram = None
        self._old_histogram = None
        InteractionMeasurement.__init__(self, name="interaction_spread", verbose=verbose)

    def measure(self, recommender):
        """
        Measures the diversity of user interactions -- that is, whether
        interactions are spread among many items or only a few items.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        interactions = recommender.interactions
        if interactions.size == 0:  # initially, there are no interactions
            self.observe(None)
            return
        histogram = self._generate_interaction_histogram(
            interactions, recommender.num_users, recommender.num_items
        )
        histogram[::-1].sort()
        if self._old_histogram is None:
            self._old_histogram = np.zeros(recommender.num_items)
        self.observe(np.trapz(self._old_histogram, dx=1) - np.trapz(histogram, dx=1), copy=False)
        self._old_histogram = np.copy(histogram)
        self.histogram = histogram


class RecallMeasurement(Measurement):
    """
    Measures the proportion of relevant items (i.e., those users interacted with) falling
    within the top k ranked items shown.

    Parameters
    -----------
        k: int
            The rank at which recall should be evaluated.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str, default ``"recall_at_k"``
            Name of the measurement component.
    """

    # Note: RecallMeasurement evalutes recall for the top-k (i.e., highest predicted value)
    # items regardless of whether these items derive from the recommender or from randomly
    # interleaved items. Currently, this metric will only be correct for
    # cases in which users iteract with one item per timestep

    def __init__(self, k=5, name="recall_at_k", verbose=False):
        self.k = k

        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        """
        Measures the proportion of relevant items (i.e., those users interacted with) falling
        within the top k ranked items shown..

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        if self.k >= recommender.num_items_per_iter:
            raise ValueError("k must be smaller than the number of items per iteration")

        interactions = recommender.interactions
        if interactions.size == 0:
            self.observe(None)  # no interactions yet
            return

        else:
            shown_item_scores = np.take(recommender.predicted_scores.value, recommender.items_shown)
            shown_item_ranks = np.argsort(shown_item_scores, axis=1)
            top_k_items = np.take(recommender.items_shown, shown_item_ranks[:, self.k :])
            recall = (
                len(np.where(np.isin(recommender.interactions, top_k_items))[0])
                / recommender.num_users
            )

        self.observe(recall)


class MSEMeasurement(Measurement, Diagnostics):
    """
    Measures the mean squared error (MSE) between real and predicted user scores.

    It can be used to evaluate how accurate the model predictions are.

    This class inherits from :class:`.Measurement`.

    Parameters
    -----------

        verbose: bool, default False
            If ``True``, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str (optional, default: "mse")
            Name of the measurement component.
    """

    def __init__(self, verbose=False, diagnostics=False, **kwargs):

        self.diagnostics = diagnostics

        Measurement.__init__(self, "mse", verbose=verbose)

        if diagnostics:
            Diagnostics.__init__(self, **kwargs)

    def measure(self, recommender):
        """
        Measures and records the mean squared error between the user preferences
        predicted by the system and the users' actual preferences.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from :class:`~models.recommender.BaseRecommender`.
        """
        diff = recommender.predicted_scores.value - recommender.users.actual_user_scores.value
        self.observe((diff ** 2).mean(), copy=False)

        if self.diagnostics:
            self.diagnose(
                (
                    recommender.predicted_scores.value.mean(axis=1)
                    - recommender.users.actual_user_scores.value.mean(axis=1)
                )
                ** 2
            )


class RMSEMeasurement(Measurement):
    """
    Measures the root mean squared error (RMSE) between real and predicted user scores.

    It can be used to evaluate how accurate the model predictions are.

    This class inherits from :class:`.Measurement`.

    Parameters
    -----------

        verbose: bool, default False
            If ``True``, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str, default ``"mse"``
            Name of the measurement component.
    """

    def __init__(self, verbose=False):
        Measurement.__init__(self, "rmse", verbose=verbose)

    def measure(self, recommender):
        """
        Measures and records the mean squared error between the user preferences
        predicted by the system and the users' actual preferences.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from :class:`~models.recommender.BaseRecommender`.
        """
        diff = recommender.predicted_scores.value - recommender.users.actual_user_scores.value
        self.observe((diff ** 2).mean() ** 0.5, copy=False)


class DiffusionTreeMeasurement(Measurement):
    """
    Class that implements an information diffusion tree. The current
    implementation assumes that agents using this class (i.e., a model)
    implement an :attr:`~models.bass.BassModel.infection_state` matrix that
    denotes the initial state of information.

    In this implementation, the nodes represent users and are labeled with the
    user indices. A branch between nodes `u` and `v` indicates that user `u`
    passed information onto user `v` -- that is, `u` "infected" `v`.

    Trees are implemented using the `Networkx library`_. Please refer to
    Networkx's `documentation`_ for more details.

    .. _Networkx library: http://networkx.github.io
    .. _documentation: https://networkx.github.io/documentation/stable/

    Parameters
    -----------

        infection_state: :class:`~models.bass.InfectionState`
            The initial "infection state" of all users

        verbose: bool, default False
            If ``True``, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str, default ``"num_infected"``
            Name of the metric that is recorded at each time step. Note that,
            in this case, the metric stored in
            :attr:`~.Measurement.measurement_history` is actually the
            **number of infected users**. The diffusion tree itself is kept in
            the :attr:`.diffusion_tree` data structure.

        diffusion_tree: :obj:`networkx.Graph`
            Diffusion tree.

        _old_infection_state: array_like
            Infection state at the previous timestep.
    """

    def __init__(self, verbose=False):
        self._old_infection_state = None
        self.diffusion_tree = nx.Graph()
        Measurement.__init__(self, "num_infected", verbose=verbose)

    def _find_parents(self, user_profiles, new_infected_users):
        """Find the users who infected the newly infected users"""
        if (self._old_infection_state == 0).all():
            # Node is root
            return None
        # TODO: function is_following() based on code below:
        # candidates must have been previously infected
        prev_infected_users = np.where(self._old_infection_state > 0)[0]
        # candidates must be connected to newly infected users
        candidate_parents = user_profiles[:, prev_infected_users][new_infected_users]
        if not isinstance(candidate_parents, np.ndarray):
            candidate_parents = candidate_parents.toarray()  # convert sparse to numpy if needed
        # randomly select parent out of those who were infected, use random multiplication
        candidate_parents = candidate_parents * np.random.rand(*candidate_parents.shape)
        parents = prev_infected_users[np.argmax(candidate_parents, axis=1)]
        return parents

    def _add_to_graph(self, user_profiles, new_infected_users):
        """Add the newly infected users to the graph with edges to the users
        who infected them
        """
        self.diffusion_tree.add_nodes_from(new_infected_users)
        parents = self._find_parents(user_profiles, new_infected_users)
        # connect parent(s) and child(ren)
        if parents is not None:
            edges = np.vstack((parents, new_infected_users)).T
            self.diffusion_tree.add_edges_from(edges)

    def _manage_new_infections(self, user_profiles, current_infection_state):
        """Add new infected users to graph and return number of newly infected
        users

        Parameters
        ------------
            user_profiles: :obj:`numpy.ndarray`
                :math:`|U|\\times|A|` numpy adjacency matrix.

            current_infection_state: :class:`~models.bass.InfectionState`
                Matrix that contains state about recovered,
                infected, and susceptible individuals.
        """
        if self._old_infection_state is None:
            self._old_infection_state = np.zeros(current_infection_state.value.shape)
        new_infections = current_infection_state.infected_users()[0]  # only extract user indices
        if len(new_infections) == 0:
            # no new infections
            return 0
        self._add_to_graph(user_profiles, new_infections)
        # return number of new infections
        return len(new_infections)

    def measure(self, recommender):
        """
        Updates tree with new infections and stores information about new
        infections. In :attr:`~.Measurement.measurement_history`, it stores the
        total number of infected users in the system -- that is, the number of
        nodes in the tree.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from :class:`~models.recommender.BaseRecommender`.
        """
        self._manage_new_infections(recommender.users_hat.value, recommender.infection_state)
        self.observe(self.diffusion_tree.number_of_nodes(), copy=False)
        self._old_infection_state = np.copy(recommender.infection_state.value)

    def draw_tree(self):
        """
        Plots the tree using the Networkx library API.
        """
        nx.draw(self.diffusion_tree, with_labels=True)


class StructuralVirality(DiffusionTreeMeasurement):
    """
    This class extends :class:`DiffusionTreeMeasurement` with the concept of
    structural virality developed by Goel, Anderson, Hofman, and Watts in
    `The Structural Virality of Online Diffusion`_. It is used in
    :class:`~models.bass.BassModel`.

    .. _The Structural Virality of Online Diffusion: https://5harad.com/papers/twiral.pdf
    """

    def __init__(self, verbose=False):
        DiffusionTreeMeasurement.__init__(self, verbose)

    def get_structural_virality(self):
        """
        Returns a measure of structural virality.

        Returns
        --------
            Structural virality: float
        """
        num_nodes = self.diffusion_tree.number_of_nodes()
        return wiener_index(self.diffusion_tree) / (num_nodes * (num_nodes - 1))


class AverageFeatureScoreRange(Measurement):
    """
    Measures the average range (across users) of item attributes for items
    users were recommended at a time step.

    This metric is based on the item diversity measure used in :

        Willemsen, M. C., Graus, M. P.,
        & Knijnenburg, B. P. (2016). Understanding the role of latent feature
        diversification on choice difficulty and satisfaction. User Modeling
        and User-Adapted Interaction, 26(4), 347-389.

    This class inherits from :class:`.Measurement`.

    Parameters
    -----------

        verbose: bool, default False
            If ``True``, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str, default ``"afsr"``
            Name of the measurement component.
    """

    def __init__(self, name="afsr", verbose=False):
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        """
        Measures the average range (across users) of item attributes for items
        users were recommended at a time step. Used as a measure of within
        list recommendation diversity

        This metric is based on the item diversity measure used in :
        Willemsen, M. C., Graus, M. P.,
        & Knijnenburg, B. P. (2016). Understanding the role of latent feature
        diversification on choice difficulty and satisfaction. User Modeling
        and User-Adapted Interaction, 26(4), 347-389.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        items_shown = recommender.items_shown
        if items_shown.size == 0:
            # at beginning of simulation, there are no recommendations,
            # so we log a `None` value
            self.observe(None)
            return

        recommended_item_attr = recommender.items_hat.value[:, items_shown]

        afsr = np.mean(
            recommended_item_attr.max(axis=(0, 2)) - recommended_item_attr.min(axis=(0, 2))
        )

        self.observe(afsr)


class DisutilityMetric(Measurement):
    def __init__(self, name = "disutility", verbose = False):
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        items_shown = recommender.items_shown
        if items_shown.size == 0:
            self.observe(0.)
            return

        items_shown_attrs = recommender.actual_item_attributes[:, items_shown]
        user_profs = recommender.actual_user_profiles
        sim_vals = np.array([cos_similarity(user_profs[i], items_shown_attrs[:, i]) for i in range(recommender.num_users)])
        self.observe(sim_vals.mean())


class RecommendationMeasurement(Measurement):
    def __init__(self, name = "recommendation_histogram", verbose = False):
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        if recommender.items_shown.size == 0:
            self.observe(None)
            return

        histogram = self.__generate_items_shown_histogram(
            recommender.items_shown, recommender.num_users, recommender.num_items, recommender.num_items_per_iter
        )
        self.observe(histogram, copy = True)

    @staticmethod
    def __generate_items_shown_histogram(items_shown, num_users, num_items, num_items_per_iter):
        """
        Generates a histogram of the number of item_shown per item at the
        given timestep.
        Parameters
        -----------
            items_shown : :obj:`numpy.ndarray`
                Array of user items_shown.
            num_users : int
                Number of users in the system
            num_items : int
                Number of items in the system
            num_items_per_iter : int
                Number of items shown at each timestep
        Returns
        ---------
            :obj:`numpy.ndarray`:
                Histogram of the number of items_shown aggregated by items at the given timestep.
        """
        histogram = np.zeros(num_items)
        for u in range(num_users):
            np.add.at(histogram, items_shown[u], 1)
        # Check that there're num_items_per_iter items_shown per user
        if histogram.sum() != num_users * num_items_per_iter:
            raise ValueError("The sum of items shown must be equal to the number of users times the number of recommender items per timestep")
        return histogram


# Instead of histograms, we simply return the list of interactions
class InteractionMetric(Measurement):
    def __init__(self, name = "interaction_history", verbose = False, user = None):
        Measurement.__init__(self, name, verbose)
        if type(user) is not int and user is not None:
            raise ValueError("Parameter user can be either a user index (int) or None.")
        self.user = user

    def measure(self, recommender):
        if recommender.interactions.size == 0:
            self.observe(None)
            return

        if self.user is not None:
            history = np.expand_dims(recommender.interactions[self.user], 0)
        else:
            history = recommender.interactions
        self.observe(history, copy = True)


# Also for recommendations
class RecommendationMetric(Measurement):
    def __init__(self, name = "recommendation_history", verbose = False, user = None):
        Measurement.__init__(self, name, verbose)
        if type(user) is not int and user is not None:
            raise ValueError("Parameter user can be either a user index (int) or None.")
        self.user = user

    def measure(self, recommender):
        if recommender.items_shown.size == 0:
            self.observe(None)
            return

        if self.user is not None:
            history = np.expand_dims(recommender.items_shown[self.user], 0)
        else:
            history = recommender.items_shown
        self.observe(history, copy = True)


class ScoreMetric(Measurement):
    def __init__(self, name = "score", user = None, verbose = False):
        Measurement.__init__(self, name, verbose)
        self.user = user

    def measure(self, recommender):
        interactions = recommender.interactions
        if interactions.size == 0:
            self.observe(None)
            return

        min_score = np.min(recommender.actual_user_item_scores, axis = 1)
        sim_vals = (recommender.users.actual_user_scores.get_item_scores(np.expand_dims(interactions, 1)).flatten() - min_score) / (np.max(recommender.actual_user_item_scores, axis = 1) - min_score)
        interactions_idx = np.argwhere(interactions != -1).flatten()
        sim_vals = sim_vals[interactions_idx] if interactions_idx.shape[0] != 0 else np.zeros_like(sim_vals)
        if self.user is None:
            self.observe(sim_vals.mean())
        else:
            self.observe(sim_vals[self.user])


class CorrelationMeasurement(Measurement, Diagnostics):
    """
    Measures the correlation coefficient between real and predicted user scores.
    It can be used to evaluate how accurate the model predictions are.
    This class inherits from :class:`.Measurement`.
    Parameters
    -----------
        verbose: bool, default False
            If ``True``, enables verbose mode. Disabled by default.
    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`
        name: str (optional, default: "mse")
            Name of the measurement component.
    """
    def __init__(self, name = "correlation", verbose = False, diagnostics = False, **kwargs):
        self.diagnostics = diagnostics
        Measurement.__init__(self, name, verbose = verbose)
        if diagnostics:
            Diagnostics.__init__(self, **kwargs)
        import warnings
        warnings.filterwarnings('ignore')

    def measure(self, recommender):
        """
        Measures and records the correlation coefficient between the user preferences
        predicted by the system and the users' actual preferences.
        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from :class:`~models.recommender.BaseRecommender`.
        """
        correlations = np.array([pearsonr(recommender.predicted_user_item_scores[i],
                                          recommender.actual_user_item_scores[i])[0] for i in range(recommender.num_users)])
        self.observe(correlations.mean(), copy = False)
        if self.diagnostics:
            self.diagnose(
                pearsonr(recommender.predicted_user_item_scores.mean(axis = 1),
                         recommender.actual_user_item_scores.mean(axis = 1))[0]
            )


class RankingMetric(Measurement):
    def __init__(self, user_profiles, name = "ranking", user = None, verbose = False):
        Measurement.__init__(self, name, verbose)
        self.user_profiles = user_profiles
        self.num_items = self.user_profiles.actual_user_scores.num_items
        if type(user) is not int and user is not None:
            raise ValueError("Parameter user can be either a user index (int) or None.")
        self.user = user
        if self.user == None:
            self.true_rank = [rankdata(self.user_profiles.actual_user_scores.value[u], method = "min") for u in range(self.user_profiles.num_users)]
        else:
            self.true_rank = rankdata(self.user_profiles.actual_user_scores.value[self.user], method = "min")

    def measure(self, recommender):
        if recommender.num_items != self.num_items or recommender.users.drift > 0:
            if self.user == None:
                self.true_rank = np.array([rankdata(-self.user_profiles.actual_user_scores.value[u], method = "min") for u in range(self.user_profiles.num_users)])
            else:
                self.true_rank = rankdata(-self.user_profiles.actual_user_scores.value[self.user], method = "min")
            self.num_items = recommender.num_items

        if self.user is not None:
            correct, rank = self.__correctly_ranked(self.user_profiles.actual_user_scores.value[self.user], recommender.predicted_user_item_scores[self.user])
            tau = kendalltau(self.true_rank, np.array(rank).flatten())[0]
        else:
            correct, rank = self.__correctly_ranked(self.user_profiles.actual_user_scores.value, recommender.predicted_user_item_scores)
            tau = np.mean([kendalltau(self.true_rank[u], rank[u], method='min')[0] for u in range(recommender.num_users)])
        self.observe(tau)

    def __correctly_ranked(self, true_scores, predicted_scores):
        num_users = true_scores.shape[0]
        num_items = true_scores.shape[1]
        sort = np.sort(true_scores, axis = 1)
        all_correct = []
        all_rank = []
        for user in range(num_users):
            pos = [[] for _ in range(num_items)]
            for e in sort[user]:
                for j, v in enumerate(sort[user]):
                    if e == v:
                        for i, u in enumerate(true_scores[user]):
                            if v == u:
                                pos[i].append(j)
            sort_rec = sorted(enumerate(predicted_scores[user]), key = lambda x: x[1])
            rank = [0 for _ in range(num_items)]
            correct = 0
            for i, e in enumerate(sort_rec):
                if i in pos[e[0]]:
                    correct += 1
                    rank[e[0]] = min(pos[e[0]]) + 1
                else:
                    for j in range(num_items):
                        if i in pos[j]:
                            rank[e[0]] = min(pos[j]) + 1
                            break
            all_correct.append(correct)
            all_rank.append(rank)
        return (all_correct, all_rank)


class RecommendationRankingMetric(Measurement):
    def __init__(self, user_profiles, name = "recommendation_ranking", user = None, verbose = False):
        Measurement.__init__(self, name, verbose)
        self.user_profiles = user_profiles
        self.num_items = self.user_profiles.actual_user_scores.num_items
        if type(user) is not int and user is not None:
            raise ValueError("Parameter user can be either a user index (int) or None.")
        self.user = user
        if self.user == None:
            self.true_rank = np.array([rankdata(-self.user_profiles.actual_user_scores.value[u], method = "min") for u in range(self.user_profiles.num_users)])
        else:
            self.true_rank = rankdata(-self.user_profiles.actual_user_scores.value[self.user], method = "min")

    def measure(self, recommender):
        if recommender.num_items != self.num_items or recommender.users.drift > 0:
            if self.user == None:
                self.true_rank = np.array([rankdata(-self.user_profiles.actual_user_scores.value[u], method = "min") for u in range(self.user_profiles.num_users)])
            else:
                self.true_rank = rankdata(-self.user_profiles.actual_user_scores.value[self.user], method = "min")
            self.num_items = recommender.num_items
        if recommender.items_shown.size == 0:
            self.observe(0.)
            return

        if self.user is not None:
            rank = np.sum(self.true_rank[recommender.items_shown[self.user]] <= recommender.num_items_per_iter) / float(recommender.num_items_per_iter)
        else:
            rank = np.mean(np.sum(np.take_along_axis(self.true_rank, recommender.items_shown, axis = 1) <= recommender.num_items_per_iter, axis = 1) / float(recommender.num_items_per_iter))
        self.observe(rank)


class InteractionRankingMetric(Measurement):
    def __init__(self, user_profiles, name = "interaction_ranking", user = None, verbose = False):
        Measurement.__init__(self, name, verbose)
        self.user_profiles = user_profiles
        self.num_items = self.user_profiles.actual_user_scores.num_items
        if type(user) is not int and user is not None:
            raise ValueError("Parameter user can be either a user index (int) or None.")
        self.user = user
        if self.user == None:
            self.true_rank = np.array([rankdata(-self.user_profiles.actual_user_scores.value[u], method = "min") for u in range(self.user_profiles.num_users)])
        else:
            self.true_rank = rankdata(-self.user_profiles.actual_user_scores.value[self.user], method = "min")

    def measure(self, recommender):
        if recommender.num_items != self.num_items or recommender.users.drift > 0:
            if self.user == None:
                self.true_rank = np.array([rankdata(-self.user_profiles.actual_user_scores.value[u], method = "min") for u in range(self.user_profiles.num_users)])
            else:
                self.true_rank = rankdata(-self.user_profiles.actual_user_scores.value[self.user], method = "min")
            self.num_items = recommender.num_items
        if recommender.interactions.size == 0:
            self.observe(0.)
            return

        if self.user is not None:
            rank = 1. / self.true_rank[recommender.interactions[self.user]] if recommender.interactions[self.user] != -1 else 0.
        else:
            tmp = 1. / np.take_along_axis(self.true_rank, np.expand_dims(recommender.interactions, 1), axis = 1)
            tmp[recommender.interactions == -1] = 0
            rank = np.mean(tmp)
        self.observe(rank)


class InteractionAttributesSimilarity(Measurement):
    def __init__(self, pairs, name = "interaction_attr_similarity", verbose = False):
        self.pairs = pairs
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        similarity = 0
        interactions = recommender.interactions
        if interactions.size == 0:
            self.observe(None) # no interactions yet
            return

        for pair in self.pairs:
            if interactions[pair[0]] != -1 and interactions[pair[1]] != -1:
                similarity += cosine_similarity(recommender.actual_item_attributes.T[interactions[pair[0]]].reshape(1, -1),
                                                recommender.actual_item_attributes.T[interactions[pair[1]]].reshape(1, -1))[0, 0]
        self.observe(similarity / len(self.pairs))


class InteractionAttrJaccard(Measurement):
    def __init__(
        self, pairs, name = "interaction_attr_jaccard", verbose = False):
        self.pairs = pairs
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        similarity = 0
        interactions = recommender.interactions
        if interactions.size == 0:
            self.observe(None)
            return

        pair_sim = []
        for pair in self.pairs:
            if interactions[pair[0]] != -1 and interactions[pair[1]] != -1:
                common = np.sum(recommender.actual_item_attributes.T[interactions[pair[0]]] == recommender.actual_item_attributes.T[interactions[pair[1]]])
            else:
                common = 0
            similarity += common / recommender.actual_item_attributes.shape[0] / len(self.pairs)
        self.observe(similarity)


class RecAttributesSimilarity(Measurement):
    def __init__(self, pairs, name = "rec_attr_similarity", verbose = False):
        self.pairs = pairs
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        similarity = 0
        items_shown = recommender.items_shown
        if items_shown.size == 0:
            # at the beginning of the simulation, there are no recommendations yet
            self.observe(None)
            return

        for pair in self.pairs:
            similarity += np.mean(cosine_similarity(recommender.actual_item_attributes.T[items_shown[pair[0]]],
                                                    recommender.actual_item_attributes.T[items_shown[pair[1]]]))
        self.observe(similarity / len(self.pairs))


class RecSummedAttributesSimilarity(Measurement):
    def __init__(self, pairs, name = "rec_summed_attr_similarity", verbose = False):
        self.pairs = pairs
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        similarity = 0
        items_shown = recommender.items_shown
        if items_shown.size == 0:
            # at the beginning of the simulation, there are no recommendations yet
            self.observe(None)
            return

        for pair in self.pairs:
            similarity += cosine_similarity(np.sum(recommender.actual_item_attributes.T[items_shown[pair[0]]], axis = 0, keepdims = True),
                                            np.sum(recommender.actual_item_attributes.T[items_shown[pair[1]]], axis = 0, keepdims = True))[0, 0]
        self.observe(similarity / len(self.pairs))


# Wrapper classes to allow dynamic pairs based on a given function for similarity metrics
class VaryingInteractionSimilarity(InteractionSimilarity):
    def __init__(self, pairs_fn, user_profiles, name = "interaction_similarity", verbose = False):
        self.pairs_fn = pairs_fn
        self.num_items = user_profiles.actual_user_scores.num_items
        pairs = self.pairs_fn(user_profiles.actual_user_profiles.value)
        super().__init__(pairs, name, verbose)

    def measure(self, recommender):
        if recommender.num_items != self.num_items or recommender.users.drift > 0:
            self.pairs = self.pairs_fn(recommender.actual_user_profiles)
            self.num_items = recommender.num_items
        super().measure(recommender)


class VaryingInteractionAttributesSimilarity(InteractionAttributesSimilarity):
    def __init__(self, pairs_fn, user_profiles, name = "interaction_attr_similarity", verbose = False):
        self.pairs_fn = pairs_fn
        self.num_items = user_profiles.actual_user_scores.num_items
        pairs = self.pairs_fn(user_profiles.actual_user_profiles.value)
        super().__init__(pairs, name, verbose)

    def measure(self, recommender):
        if recommender.num_items != self.num_items or recommender.users.drift > 0:
            self.pairs = self.pairs_fn(recommender.actual_user_profiles)
            self.num_items = recommender.num_items
        super().measure(recommender)


class VaryingInteractionAttrJaccard(InteractionAttrJaccard):
    def __init__(self, pairs_fn, user_profiles, name = "interaction_attr_jaccard", verbose = False):
        self.pairs_fn = pairs_fn
        self.num_items = user_profiles.actual_user_scores.num_items
        pairs = self.pairs_fn(user_profiles.actual_user_profiles.value)
        super().__init__(pairs, name, verbose)

    def measure(self, recommender):
        if recommender.num_items != self.num_items or recommender.users.drift > 0:
            self.pairs = self.pairs_fn(recommender.actual_user_profiles)
            self.num_items = recommender.num_items
        super().measure(recommender)


class VaryingRecSimilarity(RecSimilarity):
    def __init__(self, pairs_fn, user_profiles, name = "rec_similarity", verbose = False):
        self.pairs_fn = pairs_fn
        self.num_items = user_profiles.actual_user_scores.num_items
        pairs = self.pairs_fn(user_profiles.actual_user_profiles.value)
        super().__init__(pairs, name, verbose)

    def measure(self, recommender):
        if recommender.num_items != self.num_items or recommender.users.drift > 0:
            self.pairs = self.pairs_fn(recommender.actual_user_profiles)
            self.num_items = recommender.num_items
        super().measure(recommender)


class VaryingRecAttributesSimilarity(RecAttributesSimilarity):
    def __init__(self, pairs_fn, user_profiles, name = "rec_attr_similarity", verbose = False):
        self.pairs_fn = pairs_fn
        self.num_items = user_profiles.actual_user_scores.num_items
        pairs = self.pairs_fn(user_profiles.actual_user_profiles.value)
        super().__init__(pairs, name, verbose)

    def measure(self, recommender):
        if recommender.num_items != self.num_items or recommender.users.drift > 0:
            self.pairs = self.pairs_fn(recommender.actual_user_profiles)
            self.num_items = recommender.num_items
        super().measure(recommender)


class VaryingRecSummedAttributesSimilarity(RecSummedAttributesSimilarity):
    def __init__(self, pairs_fn, user_profiles, name = "rec_summed_attr_similarity", verbose = False):
        self.pairs_fn = pairs_fn
        self.num_items = user_profiles.actual_user_scores.num_items
        pairs = self.pairs_fn(user_profiles.actual_user_profiles.value)
        super().__init__(pairs, name, verbose)

    def measure(self, recommender):
        if recommender.num_items != self.num_items or recommender.users.drift > 0:
            self.pairs = self.pairs_fn(recommender.actual_user_profiles)
            self.num_items = recommender.num_items
        super().measure(recommender)


def most_similar_users_pair(users):
    matrix = np.multiply(cosine_similarity(users, users), np.ones((users.shape[0], users.shape[0])) - np.eye(users.shape[0])) - np.eye(users.shape[0])
    idx = np.argmax(matrix)
    return [(idx // users.shape[0], idx % users.shape[0])]


def least_similar_users_pair(users):
    matrix = cosine_similarity(users, users) + np.eye(users.shape[0])
    idx = np.argmin(matrix)
    return [(idx // users.shape[0], idx % users.shape[0])]


def all_users_pairs(users):
    pairs = []
    for i in range(users.shape[0]):
        for j in range(users.shape[0]):
            if i != j:
                pairs.append((i, j))
    return pairs


# FOR DEBUGGING ONLY
class NNLSCoefficientsxMetric(Measurement):
    def __init__(self, name = "nnls_coefficients_x", verbose = False):
        Measurement.__init__(self, name, verbose = verbose)

    def measure(self, recommender):
        if recommender.all_interactions is None:
            self.observe(None)
        users_hat = recommender.users_hat.value
        self.observe(users_hat)


class NNLSCoefficientsAMetric(Measurement):
    def __init__(self, name = "nnls_coefficients_A", verbose = False):
        Measurement.__init__(self, name, verbose = verbose)

    def measure(self, recommender):
        if recommender.all_interactions is None:
            self.observe(None)
        items_attr = recommender.predicted_item_attributes.T
        self.observe(items_attr)


class NNLSCoefficientsbMetric(Measurement):
    def __init__(self, name = "nnls_coefficients_b", verbose = False):
        Measurement.__init__(self, name, verbose = verbose)

    def measure(self, recommender):
        if recommender.all_interactions is None:
            self.observe(None)
        all_interactions = recommender.all_interactions.toarray()
        self.observe(all_interactions)