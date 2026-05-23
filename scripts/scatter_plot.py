"""Scatter plot: per-CWE data volume vs control effectiveness."""
import matplotlib.pyplot as plt

# Data from pattern-based evaluation
cwe_data = {
    'cwe-089 (SQL注入)':   (53,  100, 11),     # (samples, sec_rate_sec, sec_rate_vul)
    'cwe-022 (路径遍历)':  (60,  100, 0),
    'cwe-079 (XSS)':       (90,  100, 43),
    'cwe-078 (命令注入)':  (80,   55, 65),
}

x = [d[0] for d in cwe_data.values()]
y_sec = [d[1] for d in cwe_data.values()]
y_vul = [d[2] for d in cwe_data.values()]
labels = list(cwe_data.keys())

# Control gap = sec_rate - vul_rate (higher = better control)
y_gap = [s - v for s, v in zip(y_sec, y_vul)]

fig, ax = plt.subplots(figsize=(8, 5))

# Control gap vs data volume
colors = ['#2196F3', '#4CAF50', '#FF9800', '#f44336']
for i, (label, gap) in enumerate(zip(labels, y_gap)):
    ax.bar(i, gap, color=colors[i], width=0.6, label=label.split(' ')[0])
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([l.split(' ')[0] for l in labels])
ax.set_ylabel('Control Gap (sec_rate - vul_rate) %')
ax.set_title('CWE Control Effectiveness')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

for i, gap in enumerate(y_gap):
    ax.text(i, gap + 2, f'{gap:.0f}%', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(r'C:\Users\sven-master\trained\scatter_plot.png', dpi=150)
print('Saved')
