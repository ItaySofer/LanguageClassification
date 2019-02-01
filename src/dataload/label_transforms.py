import numpy as np
import torch


class ToOneHot(object):
    def __init__(self, language_names):
        self.language_names = np.array(language_names)

    def __call__(self, language_label):
        one_hot_func = np.vectorize(lambda lang: lang == language_label, otypes=[float])
        one_hot = one_hot_func(self.language_names)
        return torch.tensor(one_hot)


class ToLabelIdTensor(object):
    def __init__(self, language_names):
        self.languages_mapping = {lang: np.array(i) for i, lang in enumerate(language_names)}

    def __call__(self, language_label):
        label_id = self.languages_mapping[language_label]
        return torch.tensor(label_id)


class ToCuda(object):
    def __init__(self):
        pass

    def __call__(self, language_label):
        device = torch.device("cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
        return language_label.to(device=device, dtype=torch.int64)
