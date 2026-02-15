import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset.csv")   # or covtype.csv if that's your file
print(df.shape)
print(df.columns)
df = df.sample(n=50000, random_state=42)


TARGET = "Cover_Type"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.05,        # only 5% → small download file
    random_state=42,
    stratify=y
)

train_df = X_train.copy()
train_df[TARGET] = y_train

test_df = X_test.copy()
test_df[TARGET] = y_test

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("Files created:")
print("train_data.csv ->", train_df.shape)
print("test_data.csv  ->", test_df.shape)
