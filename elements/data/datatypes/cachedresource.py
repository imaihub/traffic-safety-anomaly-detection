import os
import pickle
import uuid
from typing import Any


def def_save(data: Any, fname: str):
    pickle.dump(data, open(fname, "wb"))


def def_load(fname: str) -> Any:
    return pickle.load(open(fname, "rb"))


class CachedResource:
    def __init__(self, data: Any, name: str = "", persist: bool = True, cache_dir: str = None, ext: str = "", load_cache_func=def_load, save_cache_func=def_save):
        """
        This is a convenient wrapper around a cached data resource.

        :param data: the data the object represents.
        :param persist: determines if the object is persistent (if this value is False, the memory is automatically deleted)
        :param load_cache_func: the function to use to load the resource (by default Pickle is used, but can be overridden to load images in known formats)
        :param save_cache_func: the function to use to save the resource (by default Pickle is used, but can be overridden to store images in known formats)
        """
        self.load_cache_func = load_cache_func
        self.save_cache_func = save_cache_func

        self._data = data
        self._name = name
        self._persist = persist

        if not persist and cache_dir is not None:
            self._cache_file = os.path.join(cache_dir, str(uuid.uuid4())) + ext
            self.save_cache_file()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: Any):
        self._data = data

    @property
    def persist(self):
        return self._persist

    @persist.setter
    def persist(self, persist: bool):
        self._persist = persist

    def save_cache_file(self):
        """
        Save data to temporary directory.
        """
        self.save_cache_func(data=self.data, fname=self._cache_file)

    def free(self):
        """
        Clear data to reduce memory usage.
        """
        self.data = None

    def get(self) -> Any:
        """
        Load data from cache file if data is None, otherwise return already loaded data.
        :return: data
        """
        if self.data is None:
            self.data = self.load_cache_func(fname=self._cache_file)
        return self.data

    def __call__(self) -> Any:
        """
        Load data from cache file if data is None, otherwise return already loaded data.
        :return: data
        """
        return self.get()


class PersistState:
    def __init__(self, state: bool = True):
        self._state = state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state
