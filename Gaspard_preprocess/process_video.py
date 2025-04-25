import os
import cv2

def extract_2s_clips(video_path, output_dir, block_id=0):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_clip = int(fps * 2)+1
    frames_per_hint = int(fps * 3)+1
    total_concepts = 40
    clips_per_concept = 5

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    clip_index = 0

    for concept_id in range(total_concepts):
        start_frame = int(concept_id * (frames_per_hint + frames_per_clip * clips_per_concept))

        for i in range(clips_per_concept):
            clip_start = start_frame + frames_per_hint + i * frames_per_clip
            clip_end = clip_start + frames_per_clip

            cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
            out_path = os.path.join(output_dir, f"block{block_id}_clip{clip_index:03}.mp4")
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            print(f"🎬 Saving {out_path} (frames {clip_start} → {clip_end - 1})")

            for f in range(clip_start, clip_end):
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️ Failed to read frame {f}")
                    break
                out.write(frame)

            out.release()
            clip_index += 1

    cap.release()
    print(f"✅ Finished extracting {clip_index} clips.")

def downsample_video(input_path, output_path, target_res=(512, 288), target_fps=3):
    cap = cv2.VideoCapture(input_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval = int(original_fps / target_fps)  # ex: 24 // 3 = 8

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, target_res)

    print(f"⬇️ Downsampling {input_path} → {output_path} at {target_fps} FPS")

    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, target_res)
        out.write(frame_resized)

    cap.release()
    out.release()


# 📁 Découpe automatique des 7 blocs vidéo
video_names = [
    "1st_10min.mp4",
    #"2nd_10min.mp4",
    #"3rd_10min.mp4",
    #"4th_10min.mp4",
    #"5th_10min.mp4",
    #"6th_10min.mp4",
    #"7th_10min.mp4",
]

for block_id, video_file in enumerate(video_names):
    video_path = f"./dataset/Video/{video_file}"
    output_dir = f"./dataset/video_clips/block{block_id}"
    extract_2s_clips(video_path, output_dir, block_id)

    downsampled_path = output_dir.replace("video_clips", "video_clips_downsampled")
    downsample_video(output_dir, downsampled_path)
    #os.remove(output_dir)  # Optionnel : supprime l’original pour gagner de la place
