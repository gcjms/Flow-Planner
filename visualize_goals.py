import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 加载 goal_vocab
goal_vocab = np.load('goal_vocab.npy')
print(f'Goal vocab shape: {goal_vocab.shape}')
print(f'Number of clusters: {len(goal_vocab)}')

# 统计信息
distances = np.sqrt(goal_vocab[:, 0]**2 + goal_vocab[:, 1]**2)
angles = np.arctan2(goal_vocab[:, 1], goal_vocab[:, 0])

print(f'\nDistance statistics:')
print(f'  Min: {distances.min():.2f} m')
print(f'  Max: {distances.max():.2f} m')
print(f'  Mean: {distances.mean():.2f} m')
print(f'  Std: {distances.std():.2f} m')

print(f'\nX range: [{goal_vocab[:, 0].min():.2f}, {goal_vocab[:, 0].max():.2f}] m')
print(f'Y range: [{goal_vocab[:, 1].min():.2f}, {goal_vocab[:, 1].max():.2f}] m')

# 创建可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 散点图 (x, y)
ax = axes[0, 0]
scatter = ax.scatter(goal_vocab[:, 0], goal_vocab[:, 1], c=range(len(goal_vocab)), cmap='tab20', s=100, alpha=0.7)
for i in range(len(goal_vocab)):
    ax.text(goal_vocab[i, 0], goal_vocab[i, 1], str(i), fontsize=6, ha='center')
ax.set_xlabel('X (meters)', fontsize=12)
ax.set_ylabel('Y (meters)', fontsize=12)
ax.set_title('Goal Clusters (X-Y Position)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# 2. 极坐标图
ax = axes[0, 1]
scatter = ax.scatter(angles, distances, c=range(len(goal_vocab)), cmap='tab20', s=100, alpha=0.7)
ax.set_xlabel('Angle (radians)', fontsize=12)
ax.set_ylabel('Distance (meters)', fontsize=12)
ax.set_title('Goal Clusters (Polar Coordinates)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 3. 距离分布
ax = axes[1, 0]
ax.hist(distances, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Distance from origin (meters)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Goal Distances', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 4. 统计信息
ax = axes[1, 1]
ax.axis('off')
stats_text = f"""Goal Vocabulary Statistics:

Total clusters: {len(goal_vocab)}
Dimensions: {goal_vocab.shape[1]}

Distance statistics:
  Min: {distances.min():.2f} m
  Max: {distances.max():.2f} m
  Mean: {distances.mean():.2f} m
  Std: {distances.std():.2f} m

X range: [{goal_vocab[:, 0].min():.2f}, {goal_vocab[:, 0].max():.2f}] m
Y range: [{goal_vocab[:, 1].min():.2f}, {goal_vocab[:, 1].max():.2f}] m

Angle range: [{angles.min():.2f}, {angles.max():.2f}] rad
"""
ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', verticalalignment='center')

plt.tight_layout()
plt.savefig('goal_clusters_visualization.png', dpi=150, bbox_inches='tight')
print('\nVisualization saved to: goal_clusters_visualization.png')
