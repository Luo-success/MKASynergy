# MKASynergy

This repository contains the source code and dataset for the paper: **"MKASynergy: An Adaptive Method for Drug Synergy Prediction via a Mixture-of-Experts Kernel Mechanism"**.

If you encounter any issues, bugs, or have questions regarding the code or the dataset, please feel free to contact the authors. You can open an "Issue" in this GitHub repository or reach out directly via email.
## Repository Structure

The repository is organized as follows:

*   **`data/`**: This folder contains the datasets used for training and evaluating the drug synergy prediction model.
*   **`dataprocess/`**: This folder includes the scripts and functions responsible for data preprocessing, formatting, and feature extraction.
*   **`frame/`**: This folder contains the core architectural components of the MKASynergy model, including the implementation of the Mixture-of-Experts Kernel Mechanism.
*   **`train_model.py`**: The main executable script. This file integrates the data processing and the model framework to train and evaluate the model.

## How to Run

To train the model and reproduce the results, please run the main script from your terminal or command prompt:

```bash
python train_model.py
