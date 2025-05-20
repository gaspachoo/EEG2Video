🧠 EEG-to-Video: Conditioning Pipeline Overview
This diagram illustrates the data flow from raw EEG signals to the final video generation using a CLIP-like embedding strategy.

🔄 Step-by-Step Breakdown
1. Raw EEG Segment
Shape: (62 channels × 400 time points)

This corresponds to a 2-second EEG window sampled at 200 Hz.

Input shape: (batch_size, 62, 400)

2. Feature Extraction
We apply a technique like Differential Entropy (DE) or Power Spectral Density (PSD) to compress each channel into 5 frequency bands: delta, theta, alpha, beta, gamma.

Output: a 310-dimensional vector
(since 62 channels × 5 features = 310)

Shape: (batch_size, 310)

3. CLIP-style EEG Encoder (MLP)
A multi-layer perceptron (CLIP class) maps the 310-dimensional feature vector to a simulated CLIP text embedding.

Output: vector of shape (batch_size, 77 × 768)

77 = number of tokens used by CLIP

768 = embedding dimension per token

4. Reshape to Token Format
We reshape the flat vector into (batch_size, 77, 768)

This mimics a CLIP-encoded sentence, making it compatible with the UNet originally trained on text-conditioning.

5. UNet (Video Generator)
The UNet3DConditionModel receives the EEG embedding in place of a text embedding.

It uses this input to condition the denoising process and generate a video corresponding to the EEG input.

📌 Summary
Step	Description	Output Shape
EEG Signal	Raw EEG segment	(62, 400)
Feature Extraction	DE or PSD	(310,)
MLP Encoding (CLIP)	EEG → CLIP-like embedding	(77 × 768,)
Reshape	Format for UNet	(77, 768)
UNet	Generates video	video tensor
