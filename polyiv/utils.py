from collections import defaultdict
import warnings

__all__ = ['Pairs', 'PairKeyDict']


class Pairs(object):

    def __init__(self):
        self._dict = defaultdict(list)

    def add_pair(self, item1, item2):
        if (item1 in self._dict[item2]) or (item1 in self._dict[item2]):
            warnings.warn("({}, {}) pair already exists".format(str(item1), str(item2)))
        else:
            self._dict[item1].append(item2)

    def __iter__(self):
        for key, item in self._dict.items():
            for element in item:
                yield key, element


class PairKeyDict(object):

    def __init__(self):
        self._dict = defaultdict(dict)

    def set_pair_value(self, key1, key2, value):
        self._dict[key1][key2] = value
        self._dict[key2][key1] = value
        
    def get_pair_value(self, key1, key2):
        return self._dict[key1][key2]
