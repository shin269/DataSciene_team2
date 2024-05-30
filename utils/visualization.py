import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import Tuple
def visDistOfData(df : pd.DataFrame,
                  row_len : int = 3,
                  fig_size : Tuple[int, int] = (15,12)
) -> None:
    """ visualization each columns distribution of data """
    num_rows = (len(df.columns) + 2) // row_len

    # Plot histograms for each column as subplots
    fig, axes = plt.subplots(num_rows, row_len, figsize= fig_size)
    for i, column in enumerate(df.columns):
        row = i // row_len
        col = i % row_len
        sns.histplot(df[column], kde=True, ax=axes[row, col])
        axes[row, col].set_title(f'Distribution of {column}')
        axes[row, col].set_xlabel(column)
        axes[row, col].set_ylabel('Frequency')
    
    # Remove any empty subplots
    for j in range(i + 1, num_rows * row_len):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()