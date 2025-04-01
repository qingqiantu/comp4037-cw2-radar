import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the dataset
df = pd.read_csv("Processed_Diet_Impact_Table.csv")

# Step 2: Min-Max normalization
df_scaled = df.copy()
columns_to_scale = df.columns[1:]
scaler = MinMaxScaler()
df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

# Step 3: Convert to long format
df_long = pd.melt(
    df_scaled,
    id_vars=["Diet Type"],
    var_name="Environmental Impact",
    value_name="Value"
)

# Step 4: Radar chart setup
categories = df_long["Environmental Impact"].unique().tolist()
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
diet_types = df_long["Diet Type"].unique()
colors = plt.cm.tab10.colors

for idx, diet in enumerate(diet_types):
    values = df_long[df_long["Diet Type"] == diet]["Value"].tolist()
    values += values[:1]
    ax.plot(angles, values, label=diet, color=colors[idx % len(colors)], linewidth=2)
    ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

# Final chart styling
ax.set_thetagrids(np.degrees(angles[:-1]), categories)
ax.set_title("Normalized Environmental Impact by Diet Type", size=16, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()

# Save chart
plt.savefig("Normalized_Radar_Chart_Diet_Impact.png")
plt.show()
