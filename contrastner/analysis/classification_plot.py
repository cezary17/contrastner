import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from PIL import Image
import wandb

def generate_classification_heatmap(data: dict) -> Image.Image:
    data = {key.upper().replace(" ", "-"): val for key, val in data.items()}

    mpl.rcParams['figure.dpi'] = 600
    # Extract labels
    labels = list(data.keys())

    precision = [data[label]['precision'] for label in labels]
    recall = [data[label]['recall'] for label in labels]
    f1_score = [data[label]['f1-score'] for label in labels]

    # Create a numpy array for the heatmap
    data_array = np.array([precision, recall, f1_score])

    # Create the heatmap
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 7)
    heatmap = ax.pcolor(
        data_array,
        cmap=sns.color_palette("magma", as_cmap=True),
        vmin=np.min(data_array) - 0.15,
        vmax=np.max(data_array) + 0.15
    )

    # Add labels and title
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.set_yticks(np.arange(3) + 0.5)
    ax.set_yticklabels(["Precision", "Recall", "F1-Score"])

    try:
        run_name = wandb.run.name
    except AttributeError:
        run_name = "UNKNOWN_RUN"

    ax.set_title(f"Classification Heatmap of run {run_name}")

    # Add colorbar
    fig.colorbar(heatmap, label="Score")

    # Function to format cell text with values
    def format_grid_value(value):
        return "{:.4f}".format(value)

    # Loop through each cell and set the text with value
    for (i, j), z in np.ndenumerate(data_array):
        ax.text(j + 0.5, i, format_grid_value(z), ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    fig.tight_layout()
    fig.canvas.draw()

    image = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    return image
