import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

def generate_clip_embeddings(text_file_path, model, tokenizer, device):
    with open(text_file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 200, f"Expected 200 prompts in {text_file_path}, got {len(lines)}"

    tokens = tokenizer(
        lines,
        padding='max_length',
        max_length=77,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # (200, 77, 768)

    return embeddings.cpu().numpy()


def main():
    # Setup device and model/tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    # Paths
    root = os.environ.get("HOME", os.environ.get("USERPROFILE")) + "/EEG2Video"
    blip_dir = os.path.join(root, "data/BLIP")
    save_dir = os.path.join(root, "data/Text_embeddings")
    os.makedirs(save_dir, exist_ok=True)

    # Gather text files for each block
    list_of_files = sorted(
        [os.path.join(blip_dir, f) for f in os.listdir(blip_dir) if f.endswith("_10min.txt")]
    )

    # Process each block separately and save per-block embeddings
    for block_id, text_file in enumerate(list_of_files):
        print(f"Processing block {block_id}: {text_file}...")
        emb = generate_clip_embeddings(text_file, model, tokenizer, device)  # shape (200, 77, 768)
        # emb est un np.ndarray (200,77,768)
        tensor_emb = torch.from_numpy(emb)            # conversion
        save_path = f"{save_dir}/block{block_id}.pt"
        torch.save(tensor_emb, save_path)
        print(f"Saved block {block_id} embeddings to {save_path}")

    print("All blocks processed.")

if __name__ == '__main__':
    main()
