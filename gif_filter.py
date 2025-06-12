import argparse
from PIL import Image, ImageFilter


def process_gif(input_path: str, output_path: str, mode: str, channel: str = "r", threshold: int = 128,
                blur_rgb: float = 10.0, blur_other: float = 2.0) -> None:
    """Process a GIF according to the selected mode."""
    frames = []
    with Image.open(input_path) as im:
        duration = im.info.get("duration", 100)
        loop = im.info.get("loop", 0)
        for frame_idx in range(im.n_frames):
            im.seek(frame_idx)
            frame = im.convert("RGB")
            if mode == "rgb":
                channels = frame.split()
                zero = Image.new("L", frame.size, 0)
                if channel == "r":
                    frame = Image.merge("RGB", (channels[0], zero, zero))
                elif channel == "g":
                    frame = Image.merge("RGB", (zero, channels[1], zero))
                elif channel == "b":
                    frame = Image.merge("RGB", (zero, zero, channels[2]))
                else:
                    raise ValueError("channel must be r, g or b")
                frame = frame.filter(ImageFilter.GaussianBlur(blur_rgb))
            elif mode == "bw":
                gray = frame.convert("L")
                bw = gray.point(lambda p: 255 if p > threshold else 0)
                frame = bw.filter(ImageFilter.GaussianBlur(blur_other))
            elif mode == "gray":
                gray = frame.convert("L")
                frame = gray.filter(ImageFilter.GaussianBlur(blur_other))
            else:
                raise ValueError("Invalid mode")
            frames.append(frame)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=loop)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply transformations to a GIF")
    parser.add_argument("input_gif", help="Path to input GIF")
    parser.add_argument("output_gif", help="Path to output GIF")
    parser.add_argument("--mode", choices=["rgb", "bw", "gray"], required=True,
                        help="Transformation mode")
    parser.add_argument("--channel", choices=["r", "g", "b"], default="r",
                        help="Channel for rgb mode")
    parser.add_argument("--threshold", type=int, default=128,
                        help="Threshold for bw mode")
    parser.add_argument("--blur-rgb", type=float, default=10.0,
                        help="Blur radius for rgb mode")
    parser.add_argument("--blur-other", type=float, default=2.0,
                        help="Blur radius for bw/gray modes")
    args = parser.parse_args()

    process_gif(args.input_gif, args.output_gif, args.mode, args.channel,
                args.threshold, args.blur_rgb, args.blur_other)


if __name__ == "__main__":
    main()
