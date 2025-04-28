import numpy as np
import os

input_folder = "./data/DE_1s_sw"
output_folder = "./data/DE_1per1s"
os.makedirs(output_folder, exist_ok=True)

for i in range(1, 21):  # subjects 1 to 20
    path = os.path.join(input_folder, f"sub{i}_features.npz")
    data = np.load(path)
    de = data["de"]        # (N, 62, 5)
    labels = data["labels"]
    blocks = data["blocks"]

    per_subject = np.zeros((7, 40, 5, 62, 1, 5), dtype=np.float32)  # 6D !!
    counter = np.zeros((7, 40))

    for j in range(len(de)):
        block = blocks[j]
        label = labels[j]
        concept = label  # 0-39 déjà
        vid = int(counter[block, concept])
        if vid < 5:
            per_subject[block, concept, vid] = de[j][:, None, :]  # ajoute un time axis ici
            counter[block, concept] += 1

    np.save(f"{output_folder}/sub{i}.npy", per_subject)
    print(f"✅ Saved subject {i} with shape {per_subject.shape}")
