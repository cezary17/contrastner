import random
from setfit.dataset import filter_dataset, find_indices
import numpy as np

from flair.datasets import CONLL_03

corpus = CONLL_03()

indices = find_indices(corpus, 100)

print(np.diff(indices))
