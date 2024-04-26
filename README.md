# Denoising_Autoencoder_Text_and_MRI
This repository implements autoencoders with convolutional neural networks (CNNs) built using PyTorch for denoising text and MRI images. Each data type is processed and trained independently due to the specific nature of noise encountered.

# Key Features

PyTorch Autoencoders: Leverages PyTorch to build autoencoders with convolutional layers in both the encoder and decoder to effectively capture spatial information in MRI images.
Text Denoising: Trains on textual data to remove noise introduced during transmission, storage, or acquisition.
MRI Denoising: Trains on MRI images (with artificially added noise) to reduce noise artifacts and enhance image quality.
Data Augmentation (MRI): Employs AddNoise.py (a custom function) to simulate realistic noise patterns in MRI images for training due to limited availability of noisy data.
# Dependencies

PyTorch
NumPy
Matplotlib (or other visualization library)
