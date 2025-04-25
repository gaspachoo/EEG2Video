import os
import yaml

# mapping bloc → fichier caption
TXT_MAP = {
    0: "1st_10min.txt",
    1: "2nd_10min.txt",
    2: "3rd_10min.txt",
    3: "4th_10min.txt",
    4: "5th_10min.txt",
    5: "6th_10min.txt",
    6: "7th_10min.txt",
}

def generate_yaml_for_block(block_id, pretrained_model_path="runwayml/stable-diffusion-v1-5"):
    config = {
    "pretrained_model_path": pretrained_model_path,
    "output_dir": f"./output/block{block_id}",
    "train_data": {
        "video_path": f"./data/final_data/clips/block{block_id}",
        "prompt": f"./data/final_data/{TXT_MAP[block_id]}",
        "width": 512,
        "height": 512,
        "n_sample_frames": 8,
        "sample_start_idx": 0,
        "sample_frame_rate": 1,
    },
    "validation_data": {
        "prompts": ["a woman walking", "a person cooking"],
        "video_length": 8,
        "width": 512,
        "height": 512,
        "num_inference_steps": 50,
        "guidance_scale": 12.5,
        "use_inv_latent": False,
        "num_inv_steps": 50
    },
    "learning_rate": 3e-5,
    "train_batch_size": 1,
    "max_train_steps": 1000,
    "checkpointing_steps": 1000,
    "validation_steps": 200,
    "trainable_modules": ["attn1.to_q", "attn2.to_q", "attn_temp"],
    "seed": 33,
    "mixed_precision": "fp16",
    "use_8bit_adam": False,
    "gradient_checkpointing": True,
    "enable_xformers_memory_efficient_attention": True
    }


    os.makedirs("C:/Users/gaspa/Documents/School/Centrale Med/2A/SSE/EEG2Video/EEG2Video/configs", exist_ok=True)
    yaml_path = f"C:/Users/gaspa/Documents/School/Centrale Med/2A/SSE/EEG2Video/EEG2Video/configs/block{block_id}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)

    print(f"✅ YAML saved: {yaml_path}")

for block_id in range(7):
    generate_yaml_for_block(block_id)