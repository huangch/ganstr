import torch
import torch.nn as nn
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False

EPSILON = torch.finfo(torch.float32).eps # or  torch.tensor(torch.finfo(torch.float32).eps)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

    
class VQVAE(nn.Module):
    def __init__(self, gene_num, latent_dim, num_embeddings = 1024, commitment_cost = 0.25, decay = 0.99):
        super(VQVAE, self).__init__()
        
        self.vq_loss = 0
        self.perplexity = 0
        
        self._convBlocks = nn.Sequential(
            nn.BatchNorm2d(3),
            # 48 x 48 x 3
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.LeakyReLU(0.1, inplace=True), #nn.ReLU(inplace=True),
            # nn.Dropout2d(0.25),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            # 24 x 24 x 32
            nn.Conv2d(32 ,64, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1, inplace=True), #nn.ReLU(inplace=True),
            # nn.Dropout2d(0.25),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            # 12 x 12 x 64
            nn.Conv2d(64 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1, inplace=True), #nn.ReLU(inplace=True),
            # nn.Dropout2d(0.25),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            # 6 x 6 x 128
            nn.Conv2d(128 ,256, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1, inplace=True), #nn.ReLU(inplace=True),
            # nn.Dropout2d(0.25),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2),
            # 3 x 3 x 256
            )
        
        self._pre_vq_conv = nn.Conv2d(in_channels=256, 
                                      out_channels=latent_dim,
                                      kernel_size=1, 
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, latent_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, latent_dim,
                                           commitment_cost)
            
        self._flatten = nn.Flatten()
        
        self._linearBlocks = nn.Sequential(
            nn.BatchNorm1d(3*3*latent_dim),
            nn.Linear(3*3*latent_dim, 3*3*latent_dim),
            nn.LeakyReLU(0.1, inplace=True), # nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            # nn.BatchNorm1d(1024),
            # nn.Linear(1024, 1024),
            # nn.LeakyReLU(0.1, inplace=True), #nn.ReLU(inplace=True),
            # nn.Dropout(0.25),
            nn.BatchNorm1d(3*3*latent_dim),
            nn.Linear(3*3*latent_dim, gene_num),
            nn.Softmax(dim=1)  
            )  
         
    def forward(self, img):
        encoder_output = self._convBlocks(img)
        
        z = self._pre_vq_conv(encoder_output)
        self.vq_loss, quantized, self.perplexity, _ = self._vq_vae(z)
        
        quantized = self._flatten(quantized)
            
        decoder_output = self._linearBlocks(quantized)
        
        return decoder_output
        