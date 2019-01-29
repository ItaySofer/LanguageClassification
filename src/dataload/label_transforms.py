import numpy as np


class ToOneHot(object):
    def __init__(self, language_names):
        self.language_names = np.array(language_names)

    def __call__(self, language_label):
        one_hot_func = np.vectorize(lambda lang: lang == language_label, otypes=[float])
        return one_hot_func(self.language_names)
