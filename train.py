"""
Train two models on the coffee analysis data and save them as pickle files.

model_1.pickle: linear regression, rating ~ 100g_USD
model_2.pickle: decision tree regressor, rating ~ 100g_USD + roast_cat
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

DATA_URL = ("https://raw.githubusercontent.com/leontoddjohnson/datasets/"
            "refs/heads/main/data/coffee_analysis.csv")

# roast levels from lightest to darkest, so the numbers keep their order
ROAST_ORDER = {
    "Light": 0,
    "Medium-Light": 1,
    "Medium": 2,
    "Medium-Dark": 3,
    "Dark": 4,
}


def roast_category(roast):
    """
    Turn a roast name into a number, e.g. 'Medium-Light' -> 1.

    Missing or unknown roast values become NaN, which the decision tree
    can handle.
    """
    return ROAST_ORDER.get(roast, np.nan)


def save_model(model, filename):
    """Save a trained model to a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def main():
    df = pd.read_csv(DATA_URL)

    y = df["rating"].values

    # Exercise 1: linear regression on price only
    X_1 = df[["100g_USD"]].values
    model_1 = LinearRegression()
    model_1.fit(X_1, y)
    save_model(model_1, "model_1.pickle")

    # Exercise 2: decision tree on price and roast
    df["roast_cat"] = df["roast"].map(roast_category)

    X_2 = df[["100g_USD", "roast_cat"]].values
    model_2 = DecisionTreeRegressor(random_state=42)
    model_2.fit(X_2, y)
    save_model(model_2, "model_2.pickle")

    print("Saved model_1.pickle and model_2.pickle")


# only train when the script is run, not when it is imported
if __name__ == "__main__":
    main()
