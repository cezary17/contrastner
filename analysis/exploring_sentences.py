import flair

from contrastner.dataset import KShotCounter
from flair.datasets import CONLL_03


"""
Seeds to check:
    contrastive performance: 40
    baseline performance: 92
"""
def main():
    corpus = CONLL_03()

    k_shot_counter = KShotCounter(
        k=8,
        mode="simple",
        remove_dev=True,
        remove_test=True,
        shuffle=True,
        shuffle_seed=40
    )

    k_shot_counter(corpus)

    print("Done")

if __name__ == "__main__":
    main()