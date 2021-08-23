import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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



x, val_x, y, val_y = train_test_split(x,y, test_size=0.3, shuffle=True)

model = Sequential()

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])


history = model.fit(x,y, epochs=200, batch_size=64, validation_data = (val_x, val_y), verbose=1).history


model.save("model/overfit/model.h5")

pd.DataFrame.from_dict(history).to_csv("model/overfit/data.csv", index=False)
