# Sulfur-ML

This repo contains the analysis of the Valence-to-Core X-ray Emission (VtC-XES) and X-ray Absoprtion Near Edge Structure (XANES) Spectra  of about 800 sulforganic compounds.


### The dataset
The compounds in this dataset includes sulforganic compounds in the following classes:
![dimension_reduction_overview](docs/types.svg)

### The unsupervised machine learning algorithms include

a. principal component analysis (PCA)

b. variational autoencoder (VAE)

c. t-stochastic neihgbor embedding (t-SNE)

![dimension_reduction_overview](docs/PCA_vs_VAE_vs_TSNE.svg)

PCA generates a linear mapping and is a linear dimensionality reduction algorithm as the principal componets are othogonal. However, the VAE is a nonlinear mapping that iterative learns how to map to the latent space, or information bottleneck layer. On the other hand, t-SNE is a nonlinear embedding, so it requires the entire dataset every time a lower-dimensional representation of the dataset is created.

### Structure
1. The saved trained models are saved in ```models/```.
2. All compounds used in this study are listed by name under ```Categories/```.
3. The transitions calculated by NWChem that are subsequently used to generate spectra are saved in ```Data/```.
4. The file that parses the NWChem dat files into spectra is ```tddftoutputparser.py```, which is callled in the Jupyter notebook ```Process_Dat_to_Spectra.ipynb```, which contains the broadening values used in this study.
