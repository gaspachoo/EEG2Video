import imageio
import os
from moviepy.editor import ImageSequenceClip

def convert_gif_to_mp4(gif_path, mp4_path):
    try:
        reader = imageio.get_reader(gif_path)
        frames = [frame for frame in reader]
        fps = 3  # valeur fixée
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(mp4_path, codec="libx264", audio=False, verbose=False, logger=None)
    except Exception as e:
        print(f"Error converting {gif_path} to MP4: {e}")


def batch_convert(gif_root, output_root):
    for block in sorted(os.listdir(gif_root)):
        block_path = os.path.join(gif_root, block)
        if not os.path.isdir(block_path): continue

        output_block_path = os.path.join(output_root, block)
        os.makedirs(output_block_path, exist_ok=True)

        for gif_file in sorted(os.listdir(block_path)):
            if not gif_file.endswith(".gif"): continue
            gif_path = os.path.join(block_path, gif_file)
            mp4_file = gif_file.replace(".gif", ".mp4")
            mp4_path = os.path.join(output_block_path, mp4_file)

            print(f"Converting {gif_path} -> {mp4_path}")
            convert_gif_to_mp4(gif_path, mp4_path)

if __name__ == "__main__":
    root = os.environ.get("HOME", os.environ.get("USERPROFILE")) + "/EEG2Video"
    gif_root = "./data/Video_gifs"
    output_root = "./data/Video_mp4"
    batch_convert(gif_root, output_root)
