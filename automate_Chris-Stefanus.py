
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(
    input_path="/content/titanic.csv",
    output_path="/content/titanic_preprocessed.csv"
):

    df = pd.read_csv(input_path)

    df.drop(
        columns=["Name", "Ticket", "Cabin"],
        inplace=True
    )

    df["Age"] = df["Age"].fillna(
      df["Age"].median()
    )

    df["Embarked"] = df["Embarked"].fillna(
      df["Embarked"].mode()[0]
    )

    le = LabelEncoder()

    df["Sex"] = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])

    df.to_csv(
        output_path,
        index=False
    )

    return df


if __name__ == "__main__":
    preprocess_data()
    print("preprocessing selesai")
