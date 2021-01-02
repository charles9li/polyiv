from collections import defaultdict


class PairKeyDict(object):

    def __init__(self):
        self._dict = defaultdict(dict)

    def set_pair_value(self, key1, key2, value):
        self._dict[key1][key2] = value
        self._dict[key2][key1] = value
        
    def get_pair_value(self, key1, key2):
        return self._dict[key1][key2]
