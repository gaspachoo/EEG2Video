.PHONY: p0 p1 p2 pairs

# Pass additional options with ARGS

# P0: pre-train GLMNet
p0:
	python EEGtoVideo/GLMNet/train_glmnet.py $(ARGS)

# Build npz/torch latent pairs
pairs:
	python utils/build_pairs.py $(ARGS)
	python utils/pairs_to_torch.py $(ARGS)

# P1: train Transformer with VAE and diffusion frozen
p1:
	python scripts/train_transformer.py --data ./data/latent_pairs \
	        --freeze_vae --freeze_diffuser $(ARGS)

# P2: fine-tune end-to-end at low learning rate
p2:
	python scripts/finetune_end2end.py --lr 1e-5 $(ARGS)
