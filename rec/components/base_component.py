import numpy as np
from abc import ABC, abstractmethod
from rec.utils import VerboseMode
import inspect

# subclass Numpy ndarray
class FromNdArray(np.ndarray, VerboseMode):
    def __new__(cls, input_array, num_items=None, verbose=False):
        obj = np.asarray(input_array).view(cls)
        obj.verbose = verbose
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.verbose = getattr(obj, 'verbose', False)


class BaseObserver(ABC):
    """Observer class for the observer design pattern
    """
    def register_observables(self, **kwargs):
        observables = kwargs.pop("observables", None)

        observer = kwargs.pop("observer", None)
        if observer is None:
            raise ValueError("Argument `observer` cannot be None")
        elif not isinstance(observer, list):
            raise TypeError("Argument `observer` must be a list")

        observable_type = kwargs.pop("observable_type", None)
        if not inspect.isclass(observable_type):
            raise TypeError("Argument `observable_type` must be a class")

        self._add_observable(observer=observer, observables=observables,
                            observable_type=observable_type)

    def _add_observable(self, observer, observables, observable_type):
        if len(observables) < 1:
            raise ValueError("Can't add fewer than one observable!")
        new_observables = list()
        for observable in observables:
            if isinstance(observable, observable_type):
                new_observables.append(observable)
            else:
                raise ValueError("Observables must be of type %s" % observable_type)
        observer.extend(new_observables)


class BaseObservable(ABC):
    """Observable class for the observer design patter
    """
    def get_observable(self, **kwargs):
        data = kwargs.pop("data", None)
        if data is None:
            raise ValueError("Argument `data` cannot be None")
        elif not isinstance(data, list):
            raise TypeError("Argument `data` must be a list")
        if len(data) > 0:
            name = getattr(self, 'name', 'Unnamed')
            return {name: data}
        else:
            return None

    def observe(self, *args, **kwargs):
        pass


class BaseComponent(BaseObservable, VerboseMode, ABC):
    def __init__(self, verbose=False, init_value=None):
        VerboseMode.__init__(self, __name__.upper(), verbose)
        self.component_data = list()
        self.component_data.append(init_value)

    def get_component_state(self):
        return get_observable(data=self.component_data)

    @abstractmethod
    def store_state(self):
        pass

    def get_timesteps(self):
        return len(self.component_data)