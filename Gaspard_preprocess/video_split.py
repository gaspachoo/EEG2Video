import os
from decord import VideoReader, cpu
import cv2
import numpy as np

def extract_and_downsample(video_path, output_dir, block_id, fps_target=3, target_size=(512, 288)):
    os.makedirs(output_dir, exist_ok=True)

    vr = VideoReader(video_path, ctx=cpu(0))
    original_fps = vr.get_avg_fps()
    total_frames = len(vr)
    
    frames_per_hint = int(3 * original_fps)+1
    frames_per_clip = int(2 * original_fps)+1
    total_concepts = 40
    clips_per_concept = 5
    clip_index = 0

    interval = int(original_fps / fps_target)  # par exemple: 25fps // 3fps -> interval=8

    for concept_id in range(total_concepts):
        base_idx = concept_id * (frames_per_hint + frames_per_clip * clips_per_concept)

        for vid in range(clips_per_concept):
            start_idx = base_idx + frames_per_hint + vid * frames_per_clip
            end_idx = start_idx + frames_per_clip

            if end_idx > total_frames:
                print(f"⚠️ Not enough frames for clip {clip_index} in block {block_id}")
                continue

            # Sélection des frames brutes
            frames = vr.get_batch(range(start_idx, end_idx)).asnumpy()

            # Downsample en choisissant 1 frame tous les `interval`
            frames = frames[::interval][:-1]

            resized_frames = [cv2.resize(frame, target_size) for frame in frames]

            out_path = os.path.join(output_dir, f"block{block_id}_clip{clip_index:03}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps_target, target_size)

            for frame in resized_frames:
                out.write(frame)

            out.release()

            print(f"✅ Saved {out_path} ({len(resized_frames)} frames)")
            clip_index += 1

def main():
    video_folder = "../data/Video"
    output_base = "../data/video_clips_downsampled"

    video_names = [
        "1st_10min.mp4",
        "2nd_10min.mp4",
        "3rd_10min.mp4",
        "4th_10min.mp4",
        "5th_10min.mp4",
        "6th_10min.mp4",
        "7th_10min.mp4",
    ]

    for block_id, video_file in enumerate(video_names):
        video_path = os.path.join(video_folder, video_file)
        output_dir = os.path.join(output_base, f"block{block_id}")
        print(f"🚀 Processing {video_file} → block{block_id}")
        extract_and_downsample(video_path, output_dir, block_id)

    print("🏁 All videos processed!")

if __name__ == "__main__":
    main()
