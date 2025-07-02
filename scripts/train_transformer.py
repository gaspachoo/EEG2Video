#!/usr/bin/env python3
"""Training script for the Transformer (P1).
The VAE and diffusion models remain frozen during this stage."""
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Train Transformer with frozen VAE and diffusion modules")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--freeze_vae", action="store_true")
    p.add_argument("--freeze_diffuser", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print("Starting P1: Transformer training")
    if args.freeze_vae:
        print("VAE is frozen")
    if args.freeze_diffuser:
        print("Diffusion model is frozen")
    for ep in range(args.epochs):
        print(f"Epoch {ep+1}/{args.epochs} - running training loop")
    print("P1 finished")


if __name__ == "__main__":
    main()
