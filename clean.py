import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import math


#Training Data
re = RandomOverSampler()


df = pd.read_csv("data/raw/train.csv")

y = df["Survived"]
x = df.drop(["Survived", "Cabin", "Name", "PassengerId", "Ticket"], axis=1)


embark = ["C", "Q", "S"]
genders = ["male", "female"]
for i, v in enumerate(x["Embarked"]):
    try:
        x.at[i, "Embarked"] = embark.index(v) + 1
    except ValueError as n:
        x.at[i, "Embarked"] = 0

for i, v in enumerate(x["Sex"]):
    x.at[i, "Sex"] = genders.index(v)


mean_age = df.describe()["Age"]["mean"]
for i, v in enumerate(x["Age"]):
    if(math.isnan(v)):
        x.at[i, "Age"] = mean_age

x,y = re.fit_resample(x,y)


df = pd.concat([x,y], axis=1)
df.to_csv("data/processed/train.csv", index=False)


#Test Data

test_df = pd.read_csv("data/raw/test.csv")

test_df = test_df.drop(["Cabin", "Name", "PassengerId", "Ticket"], axis=1)



for i, v in enumerate(test_df["Embarked"]):
    try:
        test_df.at[i, "Embarked"] = embark.index(v) + 1
    except ValueError as n:
        test_df.at[i, "Embarked"] = 0


for i, v in enumerate(test_df["Sex"]):
    test_df.at[i, "Sex"] = genders.index(v)

for i, v in enumerate(test_df["Age"]):
    if(math.isnan(v)):
        test_df.at[i, "Age"] = mean_age


for i, v in enumerate(test_df["Age"]):
    if(math.isnan(v)):
        test_df.at[i, "Age"] = mean_age

mean_fare = df.describe()["Fare"]["mean"]
for i, v in enumerate(test_df["Fare"]):
    if(math.isnan(v)):
        test_df.at[i, "Fare"] = mean_fare

test_y = pd.read_csv("data/raw/gender.csv")
test_y = test_y.drop(["PassengerId"], axis=1)

test_df = pd.concat([test_df, test_y], axis=1)
test_df.to_csv("data/processed/test.csv", index=False)