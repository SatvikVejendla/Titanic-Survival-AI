import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import json


df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

y = df["Survived"].values
x = df.drop(["Survived"], axis=1).values

test_y = test_df["Survived"].values
test_x = test_df.drop(["Survived"], axis=1).values


x = x.astype("float32")
y = y.astype("float32")


scaler = StandardScaler()

x = scaler.fit_transform(x)

test_x = test_x.astype("float32")
test_y = test_y.astype("float32")

test_x = scaler.fit_transform(test_x)


model = Sequential()

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])


model.fit(x,y, epochs=20, batch_size=64, verbose=1)


model.save("model/optimal/model.h5")



results = model.evaluate(test_x, test_y)

stats = {
    "accuracy": results[1],
    "loss": results[0]
}
with open("model/optimal/results.json", "w") as handler:
    json.dump(stats, handler)