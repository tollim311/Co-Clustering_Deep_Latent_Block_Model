# Co-Clustering Deep Latent Block Model

This repo contains the work for the proposed Latent Block model for co-clustering. We focused on binary data with a variatinnal approach (VAE). We used graph convolutional network (GCN) for encoding and the Latent Position Model (LPM) for decoding.
It rely on the previous work of Dingge Liang on DeepLPM (Dingge Liang, Marco Corneli, Charles Bouveyron, Pierre Latouche. Deep latent position model for node clustering in graphs. ESANN 2022 - 30th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, Oct 2022, Bruges, Belgium. ⟨hal-03874698⟩). The detailled of the theorical part of the model is in the file Co-Clustering_DLBM.pdf

It was realised during my end-of-study internship in MAASAI team at Inria Université Côte d'Azur as part of my engineerign degree at CY Tech.
Supervised by Vincent Vandewalle with the support of Marco Cornelli and Charles Bouveyron.

## model_generated_data.ipynb

This notebook contain the training of the model on data generated on the same generative model.

## LBM_generated_data.ipynb

This notebook contain the training of the model on data generated on the LBM generative model.

## movielens_data.ipynb

This notebook contain the training of the model on real data from movielens dataset.

Necessary files : ratings.dat (or movielens.csv if created)

Warning : the files is training on 20000 X 8000 size matrix and read files with more than 700M datapoint. For more info on the dataset check : https://grouplens.org/datasets/movielens/

## movielens_data.zip

It contain 2 files : rating.dat (which is the original files from movielens) and the remasterised dataset for our study movielens.csv

## model.py

Necessary files : args.py

This file contain only the model architecture

## args.py

This file contain only the parameters for the model
