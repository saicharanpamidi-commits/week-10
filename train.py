"""
Train two models on the coffee data and save them as pickle files.

model_1.pickle: linear regression, rating ~ 100g_USD
model_2.pickle: decision tree, rating ~ 100g_USD + roast_cat
"""

import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

DATA_URL = ("https://raw.githubusercontent.com/leontoddjohnson/datasets/"
            "refs/heads/main/data/coffee_analysis.csv")

# lightest to darkest, so the numbers keep the order
ROAST_ORDER = {
    "Light": 0,
    "Medium-Light": 1,
    "Medium": 2,
    "Medium-Dark": 3,
    "Dark": 4,
}


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def main():
    df = pd.read_csv(DATA_URL)
    y = df["rating"]

    # Exercise 1: linear regression on price
    model_1 = LinearRegression()
    model_1.fit(df[["100g_USD"]], y)
    save_model(model_1, "model_1.pickle")

    # Exercise 2: decision tree on price and roast
    # roasts not in ROAST_ORDER (or missing) become NaN; the tree handles that
    df["roast_cat"] = df["roast"].map(ROAST_ORDER)

    model_2 = DecisionTreeRegressor(random_state=42)
    model_2.fit(df[["100g_USD", "roast_cat"]], y)
    save_model(model_2, "model_2.pickle")

    print("Saved model_1.pickle and model_2.pickle")


if __name__ == "__main__":
    main()
