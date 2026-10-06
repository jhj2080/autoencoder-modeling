# Autoencoder Modeling for Press Predictive Maintenance

This repository collects the 1D-CNN autoencoder and feature-based autoencoder modeling artifacts prepared for the AI manufacturing data analysis competition.

## Projects

- `1d-cnn-autoencoder/`: preprocessing and training example, prepared window arrays, model checkpoint, and evaluation outputs.
- `feature-autoencoder/`: preprocessing, training, prediction and verification code; processed feature arrays; F1/F2 model checkpoints; metrics and plots.

## Data and evaluation notes

Raw source CSV files are intentionally excluded. The included prepared arrays and evaluation outputs were derived from the supplied competition data; confirm the dataset terms permit redistribution before making this repository public.

The feature-autoencoder metrics are exploratory: the supplied feature selection used anomaly labels. They should not be read as an unbiased estimate of future performance. See each project's metrics and verification files for details.

## Environments

See each project directory for its own requirements and usage instructions. The feature autoencoder was developed with Python 3.12.14, NumPy 2.3.5, and scikit-learn 1.8.0.
