import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel


def generate_clip_embeddings(text_file_path, model, tokenizer, device):
    with open(text_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    assert len(lines) == 200, f"Expected 200 prompts in {text_file_path}, got {len(lines)}"

    tokens = tokenizer(lines, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # (200, 77, 768)

    return embeddings.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    root = os.environ.get("HOME", os.environ.get("USERPROFILE")) + "/EEG2Video"
    blip_dir = os.path.join(root, "data/BLIP")
    save_dir = os.path.join(root, "data/Text_embeddings")
    os.makedirs(save_dir, exist_ok=True)

    list_of_files = sorted([os.path.join(blip_dir, f) for f in os.listdir(blip_dir) if f.endswith("_10min.txt")])
    embeddings = []
    for block_id in range(7):
        text_file = list_of_files[block_id]
        print(f"Processing {text_file}...")
        emb = generate_clip_embeddings(text_file, model, tokenizer, device)  # (200, 77, 768)
        embeddings.append(emb)
    embeddings = np.concatenate(embeddings, axis=0)  # (1200, 77, 768)
    save_path = os.path.join(save_dir, f"text_embeddings.npy")
    np.save(save_path, embeddings)
    print(f"Saved embeddings to {save_path}")


if __name__ == '__main__':
    main()
