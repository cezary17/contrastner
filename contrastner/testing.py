import random
from contrastner.dataset import filter_dataset, find_indices_old
import numpy as np

from flair.datasets import CONLL_03

corpus = CONLL_03()

indices = find_indices_old(corpus, 100)

print(np.diff(indices))
