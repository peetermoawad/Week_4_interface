"""
export_weights.py  —  Export best model weights from the Kaggle notebook
=========================================================================
Run this script on the Kaggle notebook's output to copy the correct
checkpoint file into the weights/ folder.

Usage (on your local machine after downloading the Kaggle output):
    python export_weights.py --src /path/to/kaggle/output/smp_unet_resnet34_wavg_best.pth

The best checkpoint from Week-2 is named: smp_unet_resnet34_wavg_best.pth
"""
import argparse
import os
import shutil
import torch
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model, INPUT_CHANNELS


def export(src_path: str, dst_dir: str = 'weights'):
    """
    Copy the best Kaggle checkpoint and verify it loads correctly.
    """
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, 'best_model.pth')

    print(f'Copying {src_path} -> {dst_path}')
    shutil.copy2(src_path, dst_path)

    # Verify
    print('Verifying checkpoint loads into model architecture...')
    device = torch.device('cpu')
    model  = build_model(in_channels=INPUT_CHANNELS, strategy='weight_avg')
    state  = torch.load(dst_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Quick forward pass
    dummy = torch.randn(1, INPUT_CHANNELS, 128, 128)
    with torch.no_grad():
        out = model(dummy)
    print(f'Forward pass OK: {tuple(dummy.shape)} -> {tuple(out.shape)}')
    print(f'\nWeights saved to: {os.path.abspath(dst_path)}')
    print('Ready for deployment!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True,
                        help='Path to the Kaggle checkpoint (.pth file)')
    parser.add_argument('--dst', default='weights',
                        help='Destination directory (default: weights/)')
    args = parser.parse_args()
    export(args.src, args.dst)
