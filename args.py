## Arguments for the model ##
## You can modify the parameters here ##

## Parameters of the data ##

L = 2   # Numbers of clusters in rows
Q = 3   # Numbers of clusters in columns
M = 200 # Numbers of individuals
P = 300 # Numbers of variables
D = 2   # Dimension of the latent space

## Parameters of the encoder ##

input_dim1 = M
input_dim2 = P
hidden1_dim1 = 48
hidden1_dim2 = 48

hidden2_dim1 = D
hidden2_dim2 = 1

## Parameters of the decoder ##

alpha = 0.5

## Parameters of the training ##

pre_epoch = 800
learning_rate = 0.01
num_epoch = 3000