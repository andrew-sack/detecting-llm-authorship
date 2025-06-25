r"""
Load the raw data and combine the training and test dataset together and any other dataset
"""
import numpy as np
import pandas as pd

if __name__ == "__main__":
    TASK = "one"   # task is either 'one' or 'two'
    assert TASK in ["one", "two"], "TASK should be one or two."

    df_train = pd.read_csv(
        f"../Data/AuTextification/raw/subtask_{TASK}_train.tsv",
        sep="\t",
        header=0,
        index_col=0
    )
    df_test = pd.read_csv(
        f"../Data/AuTextification/raw/subtask_{TASK}_test.tsv",
        sep="\t",
        header=0,
        index_col=0
    )

    print("Columns")
    print(df_train.columns.values)
    assert np.all(df_train.columns == df_test.columns), "Columns do not match"

    print(f"Number of Training Samples: {df_train.shape[0]}")
    print(f"Number of Test Samples:  {df_test.shape[0]}")

    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    print(df_combined.head())

    print(f"Full Number of Samples Together: {len(df_combined)}")

    df_combined.reset_index(drop=True, inplace=True)

    df_combined.to_csv(
        f"../Data/AuTextification/cleaned/subtask_{TASK}_grouped.tsv",
        sep="\t"
    )
    print("Done")