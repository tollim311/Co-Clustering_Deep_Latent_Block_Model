import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from kmeans_pytorch import kmeans
from sklearn.metrics.cluster import adjusted_rand_score
import args


def glorot_init(input_dim, output_dim):
    """
    Glorot (or Xavier) initialization for weight parameters.
    arguments:
        input_dim : Dimension of the input (Integer)
        output_dim : Dimension of the output (Integer)
    return:
        nn.Parameter : Initialized weight parameter (nn.Parameter object)
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim, dtype = torch.float32) * 2 * init_range - init_range
    #initial = initial.to(device)
    return nn.Parameter(initial)

class GraphConvSparse(nn.Module):
    """
    Graph Convolutional Layer for the encoder
    """
    def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
        """
        Initialize the layer.
        arguments:
            input_dim : Dimension of the input (Integer)
            output_dim : Dimension of the output (Integer)
            adj : Adjacency matrix of the graph (Tensor of shape (N, input_dim))
            activation : Activation function to apply after convolution (Torch function, default is ReLU)
        """
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        """
        Forward pass of the layer.
        arguments:
            inputs : Input tensor of shape (N, input_dim)
        return:
            outputs : Result tensor of shape (N, output_dim)
        """
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs

class Encoder(nn.Module):
    """
    GCN Encoder
    This encoder contains two graph convolutional layers for encoding the input data.
    It computes the mean and log standard deviation with two separate second layers and samples from a Gaussian distribution.
    """
    def __init__(self, adj):
        """
        Initialize the encoder.
        arguments:
            adj : Normalized adjacency matrix of the graph (Tensor of shape (input_dim1, input_dim2))
        """
        super(Encoder,self).__init__()
        self.base_gcn1 = GraphConvSparse(input_dim1, hidden1_dim1, adj.T)
        self.gcn_mean1 = GraphConvSparse(hidden1_dim1, hidden2_dim1, adj, activation=lambda x:x)
        self.gcn_logstddev1 = GraphConvSparse(hidden1_dim1, hidden2_dim2, adj, activation=lambda x:torch.where(x > -3, x, -3))

        self.base_gcn2 = GraphConvSparse(input_dim2, hidden1_dim2, adj)
        self.gcn_mean2 = GraphConvSparse(hidden1_dim2, hidden2_dim1, adj.T, activation=lambda x:x)
        self.gcn_logstddev2 = GraphConvSparse(hidden1_dim2, hidden2_dim2, adj.T, activation=lambda x:torch.where(x > -3, x, -3))

    def encode1(self, A):
        """
        Encode the matrix using the rows.
        arguments:
            A : Adjacency matrix of the graph (Tensor of shape (input_dim1, input_dim2))
        return:
            sampled_X : Sampled latent representation for the rows (Tensor of shape (input_dim1, hidden2_dim1))
            mean_X : Mean of the latent representation (Tensor of shape (input_dim1, hidden2_dim1))
            logstd_X : Log standard deviation of the latent representation (Tensor of shape (input_dim1, hidden2_dim1))
        """
        Id = torch.eye(input_dim1)
        hidden1 = self.base_gcn1(Id)
        mean1 = self.gcn_mean1(hidden1)
        logstd1 = self.gcn_logstddev1(hidden1)

        gaussian_noise1 = torch.randn(A.size(0), hidden2_dim1)
        sampled_X = gaussian_noise1*torch.exp(logstd1) + mean1

        return sampled_X, mean1, logstd1

    def encode2(self, A):
        """
        Encode the matrix using the columns.
        arguments:
            A : Adjacency matrix of the graph (Tensor of shape (input_dim1, input_dim2))
        return:
            sampled_Y : Sampled latent representation for the columns (Tensor of shape (input_dim2, hidden2_dim2))
            mean_Y : Mean of the latent representation (Tensor of shape (input_dim2, hidden2_dim2))
            logstd_Y : Log standard deviation of the latent representation (Tensor of shape (input_dim2, hidden2_dim2))
        """
        Id = torch.eye(input_dim2)
        hidden2 = self.base_gcn2(Id)
        mean2 = self.gcn_mean2(hidden2)
        logstd2 = self.gcn_logstddev2(hidden2)

        gaussian_noise2 = torch.randn(A.size(1), hidden2_dim1)
        sampled_Y = gaussian_noise2*torch.exp(logstd2) + mean2

        return sampled_Y, mean2, logstd2

    def forward(self,A):
        """
        Forward pass of the encoder.
        arguments:
            A : Adjacency matrix of the graph (Tensor of shape (input_dim1, input_dim2))
        return:
            X : Sampled latent representation for the rows (Tensor of shape (input_dim1, hidden2_dim1))
            Y : Sampled latent representation for the columns (Tensor of shape (input_dim2, hidden2_dim2))
            mean_X : Mean of the latent representation for the rows (Tensor of shape (input_dim1, hidden2_dim1))
            logstd_X : Log standard deviation of the latent representation for the rows (Tensor of shape (input_dim1, hidden2_dim1))
            mean_Y : Mean of the latent representation for the columns (Tensor of shape (input_dim2, hidden2_dim2))
            logstd_Y : Log standard deviation of the latent representation for the columns (Tensor of shape (input_dim2, hidden2_dim2))
        """
        X, mean_X, logstd_X = self.encode1(A)
        Y, mean_Y, logstd_Y = self.encode2(A)

        return X, Y, mean_X, logstd_X, mean_Y, logstd_Y


class Decoder(nn.Module):
    """
    Decoder for the model.
    This decoder computes the probability of edges between the latent representations of rows and columns using the distance (LPM).
    It uses a parameter alpha to control the threshold for edge formation.
    """
    def __init__(self, alpha):
        """
        Initialize the decoder.
        arguments:
            alpha : Parameter to control the threshold for edge formation (Float)
        """
        super(Decoder, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype = torch.float32), requires_grad=True)

    def forward(self, X, Y):
        """
        Forward pass of the decoder.
        arguments:
            X : Sampled latent representation for the rows (Tensor of shape (N1, N2))
            Y : Sampled latent representation for the columns (Tensor of shape (N3, N2))
        return:
            probs : Probability of edges between the rows and columns (Tensor of shape (N1, N3))
        """
        # Calcul of the euclidean distance squared between each pair
        X_exp = X.unsqueeze(1)
        Y_exp = Y.unsqueeze(0)
        diff = X_exp - Y_exp
        norm_sq = torch.sum(diff**2, dim=2)

        # Calculation of the probability (sigmoid(alpha - ||X-Y||^2) )
        term = self.alpha - 0.5*norm_sq
        probs = torch.sigmoid(term)
    
        return probs
    

## VAE Model ##

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model for graph data.
    It uses graph convolutional network for encoding and a distance-based decoder for reconstructing the adjacency matrix.
    """
    def __init__(self, adj_norm, alpha):
        """
        Initialize the VAE model.
        arguments:
            adj_norm : Normalized adjacency matrix of the graph (Tensor of size (M, P))
            alpha : Parameter to control the threshold for edge formation in the decoder (Float)
        """
        super(VAE, self).__init__()
        self.adj_norm = adj_norm
        self.encoder = Encoder(adj_norm)
        self.decoder = Decoder(alpha)

        # Initialization of the parameters of the clusters

        self.gamma = nn.Parameter(torch.FloatTensor(M,L).fill_(1.0/L), requires_grad=False)
        self.delta = nn.Parameter(torch.FloatTensor(P,Q).fill_(1.0/Q), requires_grad=False)

        self.pi = nn.Parameter(torch.FloatTensor(L).fill_(1.0/L), requires_grad=False)
        self.tau = nn.Parameter(torch.FloatTensor(Q).fill_(1.0/Q), requires_grad=False)
        
        self.mu = nn.Parameter(torch.FloatTensor(np.random.multivariate_normal(np.zeros(D),np.eye(D),L)), requires_grad=False)
        self.m = nn.Parameter(torch.FloatTensor(np.random.multivariate_normal(np.zeros(D),np.eye(D),Q)), requires_grad=False)

        self.log_sigma = nn.Parameter(torch.FloatTensor(L).fill_(1), requires_grad=False)
        self.log_s = nn.Parameter(torch.FloatTensor(Q).fill_(1), requires_grad=False)


    def init_param(self, columns, rows, nbr_cluster_col, nbr_cluster_row) :

        row_clusters, mu_km = kmeans(X=rows, num_clusters=nbr_cluster_row, distance='euclidean')
        col_clusters, m_km = kmeans(X=columns, num_clusters=nbr_cluster_col, distance='euclidean')

        gamma_km = F.one_hot(row_clusters)
        delta_km = F.one_hot(col_clusters)

        pi_km = torch.mean(gamma_km, dim=0, dtype= torch.float)
        tau_km = torch.mean(delta_km, dim=0, dtype= torch.float)

        sigma_km = torch.zeros(nbr_cluster_row)
        for c in range(nbr_cluster_row) :
            sigma_km[c] = torch.std(rows[row_clusters==c])

        s_km = torch.zeros(nbr_cluster_col)
        for c in range(nbr_cluster_col) :
            s_km[c] = torch.std(columns[col_clusters==c])

        log_sigma_km = torch.log(sigma_km)
        log_s_km = torch.log(s_km)

        return gamma_km, delta_km, pi_km, tau_km, mu_km, m_km, log_sigma_km, log_s_km
    


    def pretrain(self, A, adj_label, kl_weight = 0.0001):
        """
        Pretrain the VAE model using the provided adjacency matrix and labels.
        arguments:
            A : Adjacency matrix of the graph (Tensor of size (M, P))
            adj_label : Original adjacency matrix (used for computing loss) (Tensor of size (M, P))
            kl_weight : Weight for the KL divergence term in the loss (Float) (default is 0.0001)
        return:
            X : Sampled latent representation for the rows (Tensor of size (M, D))
            Y : Sampled latent representation for the columns (Tensor of size (P, D))
            A_probs : Probability of edges between the rows and columns (Tensor of size (M, P))
            store_pre_loss : Tensor to store the loss values for each epoch (Tensor of size (pre_epoch))
            store_recon_loss : Tensor to store the reconstruction loss values for each epoch (Tensor of size (pre_epoch))
            store_kl_loss : Tensor to store the KL divergence loss values for each epoch (Tensor of size (pre_epoch))
        """

        # Adam optimizer
        optimizer = Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=learning_rate)

        store_pre_loss = torch.zeros(pre_epoch)
        store_kl_loss = torch.zeros(pre_epoch)
        store_recon_loss = torch.zeros(pre_epoch)

        # Weights for the positive and negative samples in the binary cross-entropy loss
        pos_weight = float(adj_label.numel() - adj_label.sum()) / adj_label.sum()

        # Pretraining loop
        for epoch in range(pre_epoch):

            # Encoder forward pass
            X, Y, mean_X, logstd_X, mean_Y, logstd_Y = self.encoder(A)

            # Decoder forward pass
            A_probs = self.decoder(X, Y)

            # Reconstruction loss
            recon_loss = F.binary_cross_entropy(A_probs.view(-1),adj_label.view(-1), weight=pos_weight)

            # KL divergence loss
            kl_divergence_X = 0.5 * torch.mean( 2 * (- 1 - 2 * logstd_X + torch.exp(2 * logstd_X)) + mean_X.pow(2))
            kl_divergence_Y = 0.5 * torch.mean( 2 * (- 1 - 2 * logstd_Y + torch.exp(2 * logstd_Y)) + mean_Y.pow(2))

            # Total loss
            loss = recon_loss - kl_weight * (kl_divergence_X + kl_divergence_Y)

            # Model parameters update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            store_pre_loss[epoch] = loss.item()
            store_kl_loss[epoch] = kl_weight * (kl_divergence_X + kl_divergence_Y)
            store_recon_loss[epoch] = recon_loss
            
        # Initialize the parameters of the clusters
        self.gamma.data, self.delta.data, self.pi.data, self.tau.data, self.mu.data, self.m.data, self.log_sigma.data, self.log_s.data = self.init_param(Y, X, Q, L)

        print('Pretraining terminé !')
        return X, Y, A_probs, store_pre_loss, store_recon_loss, store_kl_loss
    


    def kullback_leibler_divergence(self, mean, mean_gcn, logstd, logstd_gcn):
        """
        Compute the Kullback-Leibler divergence between two Gaussian distributions.
        arguments :
            mean : Mean of the clusters (Tensor of shape (N2, D))
            mean_gcn : Mean of the GCN (Tensor of shape (N1, D))
            logstd : Log standard deviation of the clusters (Tensor of shape (N2, 1))
            logstd_gcn : Log standard deviation of the GCN (Tensor of shape (N1, 1))
        return :
            Kullback-Leibler divergence (Tensor of shape (N1, N2, D))
        """
        
        latent_dim = mean_gcn.shape[1]

        var_gcn = torch.exp(2*logstd_gcn)
        var = torch.exp(2*logstd) + 1e-16
        
        mean_gcn = mean_gcn.unsqueeze(1)
        mean = mean.unsqueeze(0)

        logvar_gcn = 2*logstd_gcn
        logvar = 2*logstd

        D_KL = 0.5 * (latent_dim * (logvar - logvar_gcn - 1 + var_gcn/var) + torch.sum((mean_gcn - mean)**2, dim=2) / var)

        return D_KL



    def update_var_probability(self, prior, mean, mean_gcn, log_std, logstd_gcn):
        """
        Update the variationnal probability of individuals in the clusters.
        arguments :
            prior : Prior distribution
            mean : Mean of the clusters
            mean_gcn : Mean of the GCN
            std : Log of standard deviation of the clusters
            std_gcn : Log of standard deviation of the GCN
        return :
            Updated variational probability for each individual in each cluster (Tensor of shape (N2, N1, D))
        """
        point_dim = mean_gcn.shape[0]
        cluster_dim = prior.shape[0]
        latent_dim = mean_gcn.shape[1]
        
        var_prob = torch.zeros((point_dim, cluster_dim, latent_dim), dtype=torch.float32)
        dkl = self.kullback_leibler_divergence(mean, mean_gcn, log_std, logstd_gcn)

        log_numerator = torch.log(prior) - dkl
        max_log_numerator = torch.max(log_numerator, dim=1, keepdim=True).values
        log_denominator = max_log_numerator + (torch.logsumexp(torch.log(prior) - dkl - max_log_numerator, dim=1, keepdim=True))
        var_prob = torch.exp(log_numerator - log_denominator)
    
        return var_prob
    

    
    def update_others_parameters(self, prob_var, mean, mean_gcn, logstd_gcn):
        """
        Update the parameters of the clusters.
        arguments :
            prob_var : Variational probability for each individual in each cluster (Tensor of shape (N1, N2))
            mean : Mean of the clusters (Tensor of shape (N2, D))
            mean_gcn : Mean of the GCN (Tensor of shape (N2, D))
            logstd_gcn : Log standard deviation of the GCN (Tensor of shape (N2, 1))
        return :
            prior : Updated prior distribution (Tensor of shape (N2,))
            new_mean : Updated means of the clusters (Tensor of shape (N2, D))
            new_logstd : Updated log of standard deviations of the clusters (Tensor of shape (N2, 1))
        """

        latent_dim = mean_gcn.shape[1]
    
        prob_var_sum = torch.sum(prob_var, dim=0) + 1e-16
        prior = prob_var_sum / (prob_var.shape[0] + 1e-16)

        mean_gcn = mean_gcn.unsqueeze(1)
        mean = mean.unsqueeze(0)

        # Update the mean
        new_mean = torch.sum(prob_var.unsqueeze(2) * mean_gcn, dim=0) / prob_var_sum.unsqueeze(1)
    
        # Update the standard deviation
        numerator_mean = torch.sum(prob_var * (torch.sum((mean_gcn - mean)**2, dim=2)), dim=0)
        mean_part = numerator_mean / (latent_dim * prob_var_sum + 1e-16)
        # Calculation with logsumexp for numerical stability
        max_numerator_std = torch.max(torch.log(prob_var) + 2*logstd_gcn, dim=0, keepdim=True).values
        numerator_std = max_numerator_std + torch.logsumexp(torch.log(prob_var) + 2*logstd_gcn - max_numerator_std, dim=0)
        denominator_std = torch.logsumexp(torch.log(prob_var), dim=0)
        ratio_std = numerator_std - denominator_std
        std_part = torch.exp(ratio_std)
        
        new_logstd = 0.5 * torch.log((mean_part + std_part))

        return prior, new_mean, new_logstd
    

    
    def train(self, A, A_label, row_cluster, column_cluster):
        """
        Train the VAE model
        arguments:
            A : Adjacency matrix of the graph (Tensor of size (M, P))
            adj_label : Original adjacency matrix (used for computing loss) (Tensor of size (M, P))
            row_cluster : True cluster labels for the rows (Numpy array of size (M,))
            column_cluster : True cluster labels for the columns (Numpy array of size (P,))
        return:
            X : Sampled latent representation for the rows (Tensor of size (M, D))
            Y : Sampled latent representation for the columns (Tensor of size (P, D))
            A_probs : Probability of edges between the rows and columns (Tensor of size (M, P))
            store_loss : Tensor to store the loss values for each epoch (Tensor of size (num_epoch))
            store_loss1 : Tensor to store the reconstruction loss values for each epoch (Tensor of size (num_epoch))
            store_loss2 : Tensor to store the KL divergence loss values for rows for each epoch (Tensor of size (num_epoch))
            store_loss3 : Tensor to store the KL divergence loss values for columns for each epoch (Tensor of size (num_epoch))
            store_loss4 : Tensor to store the cluster loss values for rows for each epoch (Tensor of size (num_epoch))
            store_loss5 : Tensor to store the cluster loss values for columns for each epoch (Tensor of size (num_epoch))
            store_ari_gamma : List to store the ARI score for rows for each epoch (List of length (num_epoch))
            store_ari_delta : List to store the ARI score for columns for each epoch (List of length (num_epoch)) 
        """

        # Adam optimizer
        optimizer = Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=learning_rate)

        # Store the loss and Ari score
        store_loss = torch.zeros(num_epoch)
        store_loss1 = torch.zeros(num_epoch)
        store_loss2 = torch.zeros(num_epoch)
        store_loss3 = torch.zeros(num_epoch)
        store_loss4 = torch.zeros(num_epoch)
        store_loss5 = torch.zeros(num_epoch)
        store_ari_gamma = []
        store_ari_delta = []

        # Training loop
        for i in range(num_epoch):
        
            # Encoder forward pass
            X, Y, mean_X, logstd_X, mean_Y, logstd_Y = self.encoder(A)

            # Decoder forward pass
            A_probs = self.decoder(X, Y)

            # Update gamma and delta
            self.gamma.data = self.update_var_probability(self.pi, self.mu, mean_X, self.log_sigma, logstd_X)
            self.delta.data = self.update_var_probability(self.tau, self.m, mean_Y, self.log_s, logstd_Y)

            if torch.isnan(self.gamma).any() or torch.isnan(self.delta).any():
                print("NaN detected in gamma or delta")
                break

            # Update pi, tau, mu, m, sigma and s
            self.pi.data, self.mu.data, self.log_sigma.data = self.update_others_parameters(self.gamma, self.mu, mean_X, logstd_X)
            self.tau.data, self.m.data, self.log_s.data = self.update_others_parameters(self.delta, self.m, mean_Y, logstd_Y)

            # Loss of reconstruction
            recon_loss = -torch.sum(A_label * (torch.log(A_probs + 1e-16)) + (1 - A_label) * (torch.log(1 - A_probs + 1e-16)))

            # KL divergence loss
            kl_divergence_X = torch.sum(self.gamma * self.kullback_leibler_divergence(self.mu, mean_X, self.log_sigma, logstd_X))
            kl_divergence_Y = torch.sum(self.delta * self.kullback_leibler_divergence(self.m, mean_Y, self.log_s, logstd_Y))

            # Cluster loss
            loss4 = torch.sum(self.gamma * (torch.log(self.pi + 1e-16) - torch.log(self.gamma + 1e-16)))
            loss5 = torch.sum(self.delta * (torch.log(self.tau + 1e-16) - torch.log(self.delta + 1e-16)))

            # Total loss
            loss = recon_loss -  kl_divergence_X - kl_divergence_Y + (loss4 + loss5)

            # Model parameters update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store the loss value
            store_loss[i] = torch.Tensor.item(loss)  # save train loss for visu
            store_loss1[i] = torch.Tensor.item(recon_loss)
            store_loss2[i] = torch.Tensor.item(torch.mean(kl_divergence_X))
            store_loss3[i] = torch.Tensor.item(torch.mean(kl_divergence_Y))
            store_loss4[i] = torch.Tensor.item(loss4)
            store_loss5[i] = torch.Tensor.item(loss5)
            
            # Store the ARI score
            store_ari_gamma.append(adjusted_rand_score(row_cluster, torch.argmax(self.gamma, axis=1).cpu().numpy()))
            store_ari_delta.append(adjusted_rand_score(column_cluster, torch.argmax(self.delta, axis=1).cpu().numpy()))
            
            if (i+1) % 100 == 0 :
                print(f"Epoch: {i + 1:04d} / {num_epoch:.0f}  |  Loss: {loss.item():.5f}  |  Recon loss: {recon_loss.item():.5f} |  ARI delta: {store_ari_delta[-1]:.5f} | ARI gamma: {store_ari_gamma[-1]:.5f}")
                
        return X, Y, A_probs, store_loss, store_loss1,store_loss2,store_loss3,store_loss4,store_loss5, store_ari_gamma, store_ari_delta