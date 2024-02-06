import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data as provided from the images uploaded by the user
data = {
    "Approach": ["Fine-Tuning Only", "With O-Tokens", "Without O-Tokens"] * 4,
    "Class": ["LOC", "PER", "ORG", "MISC"] * 3,
    "F1-Score": [
        # FT Only
        0.7011, 0.8581, 0.4484, 0.5047,
        # With O
        0.7003, 0.8936, 0.5499, 0.5897,
        # No O
        0.6827, 0.9155, 0.5199, 0.5785
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Pivot the DataFrame to have Classes as columns and Approaches as rows
pivot_df = df.pivot(index="Approach", columns="Class", values="F1-Score")

# Plot the data
plt.rcParams['savefig.dpi'] = 300
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap=sns.color_palette("Blues", as_cmap=True))
plt.title("F1-Scores by Approach and Class")
plt.savefig(os.getcwd() + "/f1_scores.png", bbox_inches='tight')