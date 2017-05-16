import numpy as np
import matplotlib.pyplot as plt


def prob_2_odds(p):
    return p/(1.0-p)


def odds_2_prob(o):
    return o/(o+1.0)


if __name__ == "__main__":
    prob = np.linspace(0.0, 1.0, 11)[:-1]
    odds = prob_2_odds(prob)
    print(odds)

    plt.scatter(prob, odds)
    plt.show()

    odds = np.linspace(0.0, 10.0, 100)[:-1]
    probs = odds_2_prob(odds)

    plt.scatter(odds, probs)
    plt.show()
