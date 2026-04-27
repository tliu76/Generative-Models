import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import BasicDecoder as BasicGenerator

class BasicDiscriminator(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, output_dim=1, leaky=False):
        super(BasicDiscriminator, self).__init__()
        #############################################################################
        # TODO:                                                                     #
        # 1. Implement a basic discriminator with one hidden layer:                #
        #    Linear layer followed by ReLU/LeakyReLU activation                 #
        #    Output layer with sigmoid activation                               #
        # 2. use RelU (leaky=False) or LeakyReLU (leaky=True) based on leaky param     #
        #############################################################################
        # First fully connected layer maps input image -> hidden features
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Choose activation type based on the leaky flag
        self.act1 = nn.LeakyReLU() if leaky else nn.ReLU()
        # Second fully connected layer maps hidden features -> scalar output
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Second fully connected layer maps hidden features -> scalar output  # 1 means "real", 0 means "fake"
        self.act2 = nn.Sigmoid()





        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):   
        #############################################################################
        # TODO:                                                                     #
        # 1. implement forward pass given input x. note that x here is pre-flattened   #
        #############################################################################
        # Hidden representation after first layer + activation
        h = self.act1(self.fc1(x))
        # Final probability that input x is real
        out = self.act2(self.fc2(h))

        #############################################################################
        #                              END OF YOUR CODE                             #
        ############################################################################# 
        return out

class BasicLeakyGenerator(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(BasicLeakyGenerator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.act1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = nn.Sigmoid()

    def forward(self, z):
        h = self.act1(self.fc1(z))
        return self.act2(self.fc2(h))
