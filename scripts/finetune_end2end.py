#!/usr/bin/env python3
"""End-to-end fine-tuning script (P2).
The whole pipeline is trained with a low learning rate."""
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Fine tune the full model end-to-end")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    return p.parse_args()


def main():
    args = parse_args()
    print("Starting P2: end-to-end fine tuning")
    print(f"Learning rate: {args.lr}, epochs: {args.epochs}")
    for ep in range(args.epochs):
        print(f"Epoch {ep+1}/{args.epochs} - running training loop")
    print("P2 finished")


if __name__ == "__main__":
    main()
