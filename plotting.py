import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(df, x_col, y_col, value_col,annot=False, fixed_params={}):
    filtered_df = df.copy()
    for k, v in fixed_params.items():
        filtered_df = filtered_df[filtered_df[k] == v]
    pivot_df = filtered_df.pivot(index=y_col, columns=x_col, values=value_col)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=annot, fmt=".1f", cmap="coolwarm")
    plt.title(f"{value_col} Heatmap ({', '.join([f'{k}={v}' for k,v in fixed_params.items()])})")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()
