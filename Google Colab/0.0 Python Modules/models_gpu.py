import torch as th
from torch import nn
from torch.autograd import Variable as V
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------
# Blackbox IDS Model
class DefaultBlackboxIDS(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.layer = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim*2),
            nn.Dropout(0.6),   
            # nn.ELU(),
            nn.LeakyReLU(True),
            # nn.BatchNorm1d(input_dim*2),
            nn.Linear(input_dim *2, input_dim *2),
            nn.Dropout(0.5),       
            # nn.ELU(),
            # nn.ReLU(True),
            nn.LeakyReLU(True),   
            # nn.BatchNorm1d(input_dim*2),
            nn.Linear(input_dim *2, input_dim//2),
            nn.Dropout(0.5),       
            # nn.ReLU(True),
            # nn.ELU(),
            nn.LeakyReLU(True),
            # nn.BatchNorm1d(input_dim//2),
            nn.Linear(input_dim//2,input_dim//2),
            nn.Dropout(0.4),       
            # nn.ELU(),
            # nn.ReLU(True),
            nn.LeakyReLU(True),
            
             nn.Linear(input_dim//2,output_dim),
        ).to(device)
        #nn.init.kaiming_normal_(self.layer.weight)
        self.output = nn.Sigmoid().to(device)
        #self.output = nn.Softmax()
    def forward(self,x):
        x = self.layer(x)
        return x

# SVM
class SVM(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.fully_connected = nn.Linear(input_dim, output_dim).to(device)
    def forward(self, x):
        fwd = self.fully_connected(x)  # Forward pass
        return fwd

# MLP - 2 layers
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        hidden_size = int((input_size + output_size)/2)
        super(MLP , self).__init__()
        # layer 1 
        self.layer1 = nn.Linear(input_size, hidden_size, bias=True).to(device)
        # layer 2
        self.layer2 = nn.Linear(hidden_size, output_size, bias=True).to(device)
    
    # forward pass 
    def forward(self, x):
        y_hidden = self.layer1(x)
        y = self.layer2(nn.functional.relu(y_hidden))
        return y

# LR
class LR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LR, self).__init__()
        # Linear part
        self.linear = nn.Linear(input_dim, output_dim).to(device)
        
    def forward(self, x):
        out = self.linear(x)
        return out

# ----------------------------------------------------------------
# # GAN Model
# GAN - Discriminator
class Discriminator(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim * 2, input_dim *2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim*2 , input_dim*2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim*2,input_dim//2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim//2,output_dim)
        ).to(device)
    def forward(self,x):
        return self.layer(x)

# GAN - Generator
# GAN-G Adversarial Attack 1
class Generator_A1(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Generator_A1, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim //2, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2,input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2,output_dim),
            # 08/05 - Add Tanh func.
            nn.Tanh()
        ).to(device)
    def forward(self, noise_dim, raw_attack, attack_category, POS_NONFUNCTIONAL_FEATURES):
        '''
        Generate Aversarial Attack Traffic that kept functional features.
        '''
        if attack_category != 'DOS' and attack_category != 'U2R_AND_R2L':
            raise ValueError("Preprocess Data Fail: Invalid Attack Category")
        batch_size = len(raw_attack)
        pos_nonfunctional_feature = POS_NONFUNCTIONAL_FEATURES[attack_category]
        noise = V(th.Tensor(np.random.uniform(0,1,(batch_size, noise_dim)))).to(device)
        generator_out = self.layer(noise)
        # Keep the functional features
        adversarial_attack = raw_attack.clone().type(torch.FloatTensor).to(device)            #.detach() to remove operation history; .clone() to make a copy
        for idx in range(batch_size):
            adversarial_attack[idx][pos_nonfunctional_feature] = generator_out[idx]
        return th.clamp(adversarial_attack,0.,1.).to(device)
        
# GAN-G Adversarial Attack 2
class Generator_A2(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Generator_A2, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim //2, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2,input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2,output_dim),
            # 08/05 - Add Tanh func.
            nn.Tanh()
        ).to(device)
    def forward(self, z):
        generator_output = self.layer(z)
        return th.clamp(generator_output,0.,1.).to(device)
        


# gen_adversarial_attack_a2 - Tao adversarial attack traffic doi voi A2
def gen_adversarial_attack_a2(generator_out, raw_attack, attack_category, POS_NONFUNCTIONAL_FEATURES):
    if attack_category != 'DOS' and attack_category != 'U2R_AND_R2L':
        raise ValueError("Preprocess Data Fail: Invalid Attack Category")
    pos_nonfunctional_feature = POS_NONFUNCTIONAL_FEATURES[attack_category]
    # Keep the functional features
    adversarial_attack = raw_attack.clone().type(torch.FloatTensor).to(device)
    for idx in range(len(adversarial_attack)):
        adversarial_attack[idx][pos_nonfunctional_feature] = generator_out[idx]
    return adversarial_attack.to(device)
