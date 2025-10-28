# visualize_pt_samples.py
import os
import torch
import matplotlib.pyplot as plt
import argparse

def visualize_pt_samples(pt_dir, num_samples=512, save_fig=True, output_dir=None):
    pt_files = sorted(f for f in os.listdir(pt_dir) if f.endswith('.pt'))
    if len(pt_files) == 0:
        print(f"No .pt files found in {pt_dir}")
        return

    os.makedirs(output_dir, exist_ok=True) if save_fig and output_dir else None

    for i in range(min(num_samples, len(pt_files))):
        path = os.path.join(pt_dir, pt_files[i+128])
        signal, target = torch.load(path)  # signal: [4, H, W], target: [C, H, W]
        # print(torch.max(target))
        # print(torch.min(target))

        n_signal = signal.shape[0]
        n_target = target.shape[0]
        total_panels = n_signal + n_target

        fig, axs = plt.subplots(1, total_panels, figsize=(4 * total_panels, 4))

        for j in range(n_signal):
            axs[j].imshow(signal[j].numpy(), cmap='viridis')
            axs[j].set_title(f"Signal[{j}]")
            axs[j].axis('off')

        for j in range(n_target):
            axs[n_signal + j].imshow(target[j].numpy(), cmap='jet')
            axs[n_signal + j].set_title(f"Target[{j}]")
            axs[n_signal + j].axis('off')

        plt.suptitle(f"Sample {i} - {pt_files[i]}", fontsize=14)

        if save_fig:
            out_path = os.path.join(output_dir, f"sample_{i:03d}.png")
            plt.savefig(out_path, bbox_inches='tight')
            print(f"Saved to {out_path}")
            plt.close()
        else:
            plt.tight_layout()
            # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize preprocessed .pt samples")
    parser.add_argument('--pt_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\damageM_128', help='Directory containing .pt samples')
    parser.add_argument('--num_samples', type=int, default=512, help='Number of samples to visualize')
    parser.add_argument('--save_fig', action='store_true', help='Save figures instead of showing')
    parser.add_argument('--output_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\visualization_PToutput\damageM', help='Output directory to save figures')

    args = parser.parse_args()
    visualize_pt_samples(args.pt_dir, args.num_samples, args.save_fig, args.output_dir)
