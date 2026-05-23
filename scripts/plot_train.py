import re
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    steps, lm_loss, contrastive_loss, kl_loss, total_loss = [], [], [], [], []
    val_data = []  # (cum_step_start, cum_step_end, lm, ct, kl, total)

    cum_step = 0
    last_step = 0
    epoch_start_step = 0
    epoch_step_max = 0
    epoch_num = 0

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r'steps:\s*(\d+)/(\d+),\s*lm_loss:\s*([\d.]+),\s*contrastive_loss:\s*([\d.]+),\s*kl_loss:\s*([\d.]+),\s*loss:\s*([\d.]+)', line)
            if m:
                raw_step = int(m.group(1))
                raw_total = int(m.group(2))
                if raw_step < last_step:
                    cum_step = epoch_step_max
                cum_step_ref = cum_step + raw_step
                steps.append(cum_step_ref)
                lm_loss.append(float(m.group(3)))
                contrastive_loss.append(float(m.group(4)))
                kl_loss.append(float(m.group(5)))
                total_loss.append(float(m.group(6)))
                last_step = raw_step
                epoch_step_max = max(epoch_step_max, cum_step + raw_total)

            vm = re.search(r'val epoch\s+(\d+):\s*lm_loss:\s*([\d.]+),\s*contrastive_loss:\s*([\d.]+),\s*kl_loss:\s*([\d.]+),\s*loss:\s*([\d.]+)', line)
            if vm:
                epoch_num = int(vm.group(1))
                val_data.append({
                    'step_start': epoch_start_step,
                    'step_end': steps[-1] if steps else epoch_start_step,
                    'lm': float(vm.group(2)),
                    'ct': float(vm.group(3)),
                    'kl': float(vm.group(4)),
                    'total': float(vm.group(5)),
                })
                epoch_start_step = steps[-1] if steps else 0

    return steps, lm_loss, contrastive_loss, kl_loss, total_loss, val_data


def main():
    parser = argparse.ArgumentParser(description='Plot SVEN training loss curves')
    parser.add_argument('--log', type=str, required=True, help='Path to train.log')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    args = parser.parse_args()

    steps, lm_loss, contrastive_loss, kl_loss, total_loss, val_data = parse_log(args.log)

    if not steps:
        print('No training steps found in log file')
        return

    print(f'Found {len(steps)} logged steps (cumulative step {steps[0]} to {steps[-1]})')
    print(f'Found {len(val_data)} validation checkpoints')

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'Training & Validation Loss — {os.path.basename(args.log)}', fontsize=14, fontweight='bold')

    axes_map = [
        (axes[0][0], 'Total Loss', total_loss, 'b'),
        (axes[0][1], 'LM Loss', lm_loss, 'g'),
        (axes[1][0], 'Contrastive Loss', contrastive_loss, 'r'),
        (axes[1][1], 'KL Loss', kl_loss, 'm'),
    ]
    val_keys = ['total', 'lm', 'ct', 'kl']

    for (ax, title, data, color), vk in zip(axes_map, val_keys):
        ax.plot(steps, data, color=color, linewidth=1.2, alpha=0.9, label='train')
        for vd in val_data:
            ax.axhline(y=vd[vk], xmin=0, xmax=1, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=-1, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='val')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Cumulative Step')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f'Saved to {args.output}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
