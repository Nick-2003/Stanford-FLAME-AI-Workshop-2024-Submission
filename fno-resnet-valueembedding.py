import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

device = get_device()
print('Device: ',device)

# ============================================================

# Paths
input_path = '/kaggle/input/2024-flame-ai-challenge/dataset/'

# Load data
train_df = pd.read_csv(os.path.join(input_path, 'train.csv'))
test_df = pd.read_csv(os.path.join(input_path, 'test.csv'))

# Function to load data
def load_dataX(idx, df, data_dir):
    csv_file = df.reset_index().to_dict(orient='list')
    dir_path = os.path.join(input_path, data_dir)
    
    id = csv_file['id'][idx]
    nt, Nx, Ny = csv_file['Nt'][idx], csv_file['Nx'][idx], csv_file['Ny'][idx]
    
    theta = np.fromfile(os.path.join(dir_path, csv_file['theta_filename'][idx]), dtype="<f4").reshape(nt, Nx, Ny)
    ustar = np.fromfile(os.path.join(dir_path, csv_file['ustar_filename'][idx]), dtype="<f4").reshape(nt, Nx, Ny)
    xi_f = np.fromfile(os.path.join(dir_path, csv_file['xi_filename'][idx]), dtype="<f4").reshape(nt, Nx, Ny)
    
    uin = csv_file['u'][idx]
    alpha = csv_file['alpha'][idx]

    return theta, ustar, xi_f, uin, alpha, id

# Function to extract fire positions
def extract_fire_positions(xi_f):
    return [np.argmax(np.mean(xi_f[t], axis=1)) for t in range(xi_f.shape[0])]

# Prepare training data
Datalist = []

for idx in range(len(train_df)):
    theta, ustar, xi_f, uin, alpha, id = load_dataX(idx, train_df, 'train')
    
    theta = torch.Tensor(theta).unsqueeze(1)
    ustar = torch.Tensor(ustar).unsqueeze(1)
    xi_f = torch.Tensor(xi_f).unsqueeze(1)
    
    uin_tensor = torch.zeros_like(xi_f) + uin
    alpha_tensor = torch.zeros_like(xi_f) + alpha
    
    TUXUA = torch.cat([theta,ustar,xi_f, uin_tensor, alpha_tensor], dim=1)
    TUXUA = TUXUA.unsqueeze(0)
    
    Datalist.append(TUXUA)
    
Data_train = torch.cat(Datalist)
print(Data_train.shape)

# Prepare testing data
Datalist = []

for idx in range(len(test_df)):
    theta, ustar, xi_f, uin, alpha, id = load_dataX(idx, test_df, 'test')
    
    theta = torch.Tensor(theta).unsqueeze(1)
    ustar = torch.Tensor(ustar).unsqueeze(1)
    xi_f = torch.Tensor(xi_f).unsqueeze(1)
    
    uin_tensor = torch.zeros_like(xi_f) + uin
    alpha_tensor = torch.zeros_like(xi_f) + alpha
    
    TUXUA = torch.cat([theta,ustar,xi_f, uin_tensor, alpha_tensor], dim=1)
    TUXUA = TUXUA.unsqueeze(0)
    
    Datalist.append(TUXUA)
    
Data_test = torch.cat(Datalist)
print(Data_test.shape)

D1 = Data_train.reshape([Data_train.shape[0]*Data_train.shape[1], Data_train.shape[2], Data_train.shape[3], Data_train.shape[4]])
D2 = Data_test.reshape([Data_test.shape[0]*Data_test.shape[1], Data_test.shape[2], Data_test.shape[3], Data_test.shape[4]])
print(D1.shape, D2.shape) 

D = torch.cat([D1,D2], dim=0)
MeanX = torch.mean(D, (0,2,3)).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
StdX = torch.std(D, (0,2,3)).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
print(StdX.shape)   # sample, timestep, channels [theta,ustar,xi_f, uin, alpha], X, Y
print(MeanX.shape)

MeanY = torch.mean(D, (0,2,3))[:3].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
StdY = torch.std(D, (0,2,3))[:3].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
print(StdY.shape)   # sample, timestep, channels [theta,ustar,xi_f], X, Y
print(MeanY.shape)

# ============================================================

# Create custom PyTorch dataset
class FlameDataset(Dataset):
    def __init__(self, Data, history = 1, prediction = 1):
        self.X = Data    #torch.Size([9, 150, 5, 113, 32])
        self.history = history
        self.prediction = prediction
        self.count_cases = Data.shape[0]
        self.count_timeIndices = Data.shape[1] - history - prediction + 1
        self.indices = torch.arange(self.count_cases*self.count_timeIndices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        case = idx%self.count_cases
        index = idx%self.count_timeIndices
        
        X = self.X[case,index:index + self.history,...]
        Y = self.X[case,index + self.history:index + self.history + self.prediction,:3,...]
        T = index + self.history #* torch.ones(idx.shape[0])
        return X, Y, T

# ============================================================

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes 
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

# class FNO1d(nn.Module):
#     def __init__(self, num_channels, modes=16, width=64, initial_step=10):
#         super(FNO1d, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
#         input: the solution of the initial condition and location (a(x), x)
#         input shape: (batchsize, x=s, c=2)
#         output: the solution of a later timestep
#         output shape: (batchsize, x=s, c=1)
#         """

#         self.modes1 = modes
#         self.width = width
#         self.padding = 2 # pad the domain if input is non-periodic
#         self.fc0 = nn.Linear(initial_step*num_channels+1, self.width) # input channel is 2: (a(x), x)

#         self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
#         self.w0 = nn.Conv1d(self.width, self.width, 1)
#         self.w1 = nn.Conv1d(self.width, self.width, 1)
#         self.w2 = nn.Conv1d(self.width, self.width, 1)
#         self.w3 = nn.Conv1d(self.width, self.width, 1)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, num_channels)

#     def forward(self, x):
#         # x dim = [b, x1, t*v]
#         #x = torch.cat((x, grid), dim=-1)
#         x = self.fc0(x)
#         x = x.permute(0, 2, 1)
        
#         x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic

#         x1 = self.conv0(x)
#         x2 = self.w0(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv1(x)
#         x2 = self.w1(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv2(x)
#         x2 = self.w2(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv3(x)
#         x2 = self.w3(x)
#         x = x1 + x2

#         x = x[..., :-self.padding]
#         x = x.permute(0, 2, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         return x.unsqueeze(-2)


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, num_channels, modes1=12, modes2=12, width=20, initial_step=10, predict=1):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x, y, c)
        output: the solution of the next timestep
        output shape: (batchsize, x, y, c)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(initial_step*num_channels, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, predict)

    def forward(self, x):
        # x dim = [b, x1, x2, t*v]   
        #x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding] # Unpad the tensor
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x
    

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

# class FNO3d(nn.Module):
#     def __init__(self, num_channels, modes1=8, modes2=8, modes3=8, width=20, initial_step=10):
#         super(FNO3d, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
#         input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
#         input shape: (batchsize, x=64, y=64, t=40, c=13)
#         output: the solution of the next 40 timesteps
#         output shape: (batchsize, x=64, y=64, t=40, c=1)
#         """

#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.modes3 = modes3
#         self.width = width
#         self.padding = 6 # pad the domain if input is non-periodic
#         self.fc0 = nn.Linear(initial_step*num_channels+3, self.width)
#         # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

#         self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
#         self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
#         self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
#         self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
#         self.w0 = nn.Conv3d(self.width, self.width, 1)
#         self.w1 = nn.Conv3d(self.width, self.width, 1)
#         self.w2 = nn.Conv3d(self.width, self.width, 1)
#         self.w3 = nn.Conv3d(self.width, self.width, 1)
#         self.bn0 = torch.nn.BatchNorm3d(self.width)
#         self.bn1 = torch.nn.BatchNorm3d(self.width)
#         self.bn2 = torch.nn.BatchNorm3d(self.width)
#         self.bn3 = torch.nn.BatchNorm3d(self.width)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, num_channels)

#     def forward(self, x):
#         # x dim = [b, x1, x2, x3, t*v]
#         #x = torch.cat((x, grid), dim=-1)
#         x = self.fc0(x)
#         x = x.permute(0, 4, 1, 2, 3)
        
#         x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic

#         x1 = self.conv0(x)
#         x2 = self.w0(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv1(x)
#         x2 = self.w1(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv2(x)
#         x2 = self.w2(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv3(x)
#         x2 = self.w3(x)
#         x = x1 + x2

#         x = x[..., :-self.padding]
#         x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         return x.unsqueeze(-2)

# ============================================================

import torch.nn.functional as F

def sinusoidal_embedding(n, d, device):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d).to(device)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)]).to(device)
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1)).to(device)
    embedding[:, ::2] = torch.sin(t * wk[:, ::2]).to(device)
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2]).to(device)

    return embedding

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            #nn.LeakyReLU(1., inplace=True)    # to allow negative output
        )

    def forward(self, x):
        x = self.double_conv(x)
        #x = self.fno_layer(x)
        return x


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        #x = self.fno_layerUp(x)
        return x


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)


# class UNet(nn.Module):
#     def __init__(self, n_filters = 4, n_channels=3, n_classes=1, n_inner=8, bilinear=False, uin_steps = 10, alpha_steps = 300, time_emb_dim=100, device = 'cpu'):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels        
#         self.n_classes = n_classes
#         self.n_inner = n_inner
#         self.bilinear = bilinear
#         self.device = device
#         self.uin_steps = uin_steps
#         self.alpha_steps = alpha_steps

#         # Sinusoidal embedding
#         self.alpha_embed = nn.Embedding(alpha_steps, time_emb_dim, device=device)
#         self.alpha_embed.weight.data = sinusoidal_embedding(alpha_steps, time_emb_dim, device)
#         self.alpha_embed.requires_grad_(False)
        
#         self.uin_embed = nn.Embedding(uin_steps, time_emb_dim, device=device)
#         self.uin_embed.weight.data = sinusoidal_embedding(uin_steps, time_emb_dim, device)
#         self.uin_embed.requires_grad_(False)
         
#         self.alpha_encoding_open = self._make_te(time_emb_dim, n_channels)
#         self.uin_encoding_open = self._make_te(time_emb_dim, n_channels)

#         self.alpha_encoding_inner = self._make_te(time_emb_dim, n_inner)
#         self.uin_encoding_inner = self._make_te(time_emb_dim, n_inner)

#         self.alpha_encoding_close = self._make_te(time_emb_dim, n_inner)
#         self.uin_encoding_close = self._make_te(time_emb_dim, n_inner)
        
#         self.fno_layer_1st = FNO2d(num_channels=1, modes1=16, modes2=16, width=96, initial_step=n_channels, predict=n_inner)
#         self.fno_layer_inner = FNO2d(num_channels=1, modes1=16, modes2=16, width=96, initial_step=n_inner, predict=n_inner)
#         self.fno_layer_last = FNO2d(num_channels=1, modes1=16, modes2=16, width=96, initial_step=n_inner, predict=n_classes)
        
        
#     def forward(self, x, t):
        
#         # x has 5 channels [theta,ustar,xi_f, uin_tensor, alpha_tensor]
#         # variables for embedding  # passed to nn positional encoding hence has to be integer or long integer
#         uin =x[:,0,3,0,0].long()   
#         alpha = (x[:,0,4,0,0]*10.).long()
        
#         #print(alpha)
#         x = (x - MeanX)/StdX        
#         # print(x.shape)     # [32, 4, 5, 113, 32]
        
#         x = x[:,:,:3,:,:]
#         Xshape = x.shape
#         #print(x.shape)
#         x = x.reshape([Xshape[0], Xshape[1] * Xshape[2], Xshape[3], Xshape[4]])
        
#         n = len(x)
#         #x_ = x   
        
#         uin = self.uin_embed(uin)  # takes in long or int tensor not float
#         alpha = self.alpha_embed(alpha)
        
#         uin_open = self.uin_encoding_open(uin).reshape(n, -1, 1, 1)
#         alpha_open = self.uin_encoding_open(alpha).reshape(n, -1, 1, 1)
        
#         uin_inner = self.uin_encoding_inner(uin).reshape(n, -1, 1, 1)
#         alpha_inner = self.uin_encoding_inner(alpha).reshape(n, -1, 1, 1)
        
#         uin_close = self.uin_encoding_close(uin).reshape(n, -1, 1, 1)
#         alpha_close = self.uin_encoding_close(alpha).reshape(n, -1, 1, 1)
        
#         x = self.fno_layer_1st(x + uin_open + alpha_open)
#         for i in range(10):
#             x = self.fno_layer_inner(x + uin_inner + alpha_inner) + x
#         x = self.fno_layer_last(x + uin_close + alpha_close)
        
#         x = x.reshape([Xshape[0], 1, 3, Xshape[3], Xshape[4]])
        
#         x = (x * StdY) + MeanY    
#         x[:,:,2:,:,:][x[:,:,2:,:,:]<0.02] = 0.        
        
#         #print(x.shape)  32,T,3,113,32
        
#         return x
#         # logits = self.outc(x)
#         # return logits
        
#     def _make_te(self, dim_in, dim_out):
#         return nn.Sequential(
#             nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
#         )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================

# Model parameters
batch_size = 32
learning_rate = 1e-3
num_epochs = 100
num_filters = 64
history = 4
prediction = 1

# Create dataset and dataloaders for training and test sets
train_dataset = FlameDataset(Data_train, history = history, prediction = prediction)
test_dataset = FlameDataset(Data_test, history = history, prediction = prediction)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = torch.load('/kaggle/input/fno_resnet_vembedding/pytorch/default/1/best_model (9).pth')
best_model = torch.load('/kaggle/input/fno_resnet_vembedding/pytorch/default/1/best_model (9).pth')
print('Number of model parameters = %3d'%(count_parameters(model)))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ============================================================

# Train the model
train_loss = []
validation_loss = []
best_loss = 10000
best_epoch = 0
for epoch in range(num_epochs):
    running_loss = 0.0
    running_val_loss = 0.0
    for xx,yy,tt  in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        tt = tt.to(device)
        optimizer.zero_grad()
        outputs = model(xx, tt*0.)
        
        theta_misfit = criterion(outputs[:,:,0:1,:,:], yy[:,:,0:1,:,:])/criterion(outputs[:,:,0:1,:,:]*0., yy[:,:,0:1,:,:])  
        ustar_misfit = criterion(outputs[:,:,1:2,:,:], yy[:,:,1:2,:,:])/criterion(outputs[:,:,1:2,:,:]*0., yy[:,:,1:2,:,:]) 
        xi_f_misfit = criterion(outputs[:,:,2:3,:,:], yy[:,:,2:3,:,:])/criterion(outputs[:,:,2:3,:,:]*0., yy[:,:,2:3,:,:])
        
        loss = theta_misfit + ustar_misfit + xi_f_misfit
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    for xx,yy,tt  in test_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        tt = tt.to(device)
        with torch.no_grad():
            outputs = model(xx, tt*0.)
        
            val_loss = criterion(outputs, yy)
        running_val_loss += val_loss.item()
    
    trn_loss = running_loss/len(train_loader)
    val_loss = running_val_loss/len(test_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {trn_loss:.3e}, Validation Loss: {val_loss:.3e}")
    
    train_loss.append(trn_loss)
    validation_loss.append(val_loss)
    
    if val_loss <= best_loss:
        best_model.load_state_dict(model.state_dict())
        torch.save(best_model, 'best_model.pth')
        best_epoch = epoch
        best_loss = val_loss

# ============================================================

torch.save(model, 'model.pth')
print('best epoch: ', best_epoch + 1)

plt.plot(train_loss, label="Train Loss")
plt.plot(validation_loss, label="Validation Loss")
plt.legend()
plt.yscale("log")
plt.title("Loss Curve")
plt.show()

# ============================================================

# Train the model AGAIN
learning_rate = 1e-4
num_epochs = 500
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0.0
    running_val_loss = 0.0
    for xx,yy,tt  in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        tt = tt.to(device)
        optimizer.zero_grad()
        outputs = model(xx, tt*0.)
        
        theta_misfit = criterion(outputs[:,:,0:1,:,:], yy[:,:,0:1,:,:])/criterion(outputs[:,:,0:1,:,:]*0., yy[:,:,0:1,:,:])  
        ustar_misfit = criterion(outputs[:,:,1:2,:,:], yy[:,:,1:2,:,:])/criterion(outputs[:,:,1:2,:,:]*0., yy[:,:,1:2,:,:]) 
        xi_f_misfit = criterion(outputs[:,:,2:3,:,:], yy[:,:,2:3,:,:])/criterion(outputs[:,:,2:3,:,:]*0., yy[:,:,2:3,:,:])
        
        loss = theta_misfit + ustar_misfit + xi_f_misfit
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    for xx,yy,tt  in test_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        tt = tt.to(device)
        optimizer.zero_grad()
        outputs = model(xx, tt*0.)
        
        theta_misfit = criterion(outputs[:,:,0:1,:,:], yy[:,:,0:1,:,:])/criterion(outputs[:,:,0:1,:,:]*0., yy[:,:,0:1,:,:])  
        ustar_misfit = criterion(outputs[:,:,1:2,:,:], yy[:,:,1:2,:,:])/criterion(outputs[:,:,1:2,:,:]*0., yy[:,:,1:2,:,:]) 
        xi_f_misfit = criterion(outputs[:,:,2:3,:,:], yy[:,:,2:3,:,:])/criterion(outputs[:,:,2:3,:,:]*0., yy[:,:,2:3,:,:])
        
        loss = theta_misfit + ustar_misfit + xi_f_misfit
        
        loss.backward()
        optimizer.step()
    
    for xx,yy,tt  in test_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        tt = tt.to(device)
        with torch.no_grad():
            outputs = model(xx, tt*0.)
        
            val_loss = criterion(outputs, yy)
        running_val_loss += val_loss.item()
    
    trn_loss = running_loss/len(train_loader)
    val_loss = running_val_loss/len(test_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {trn_loss:.3e}, Validation Loss: {val_loss:.3e}")
    
    train_loss.append(trn_loss)
    validation_loss.append(val_loss)
    
    if val_loss <= best_loss:
        best_model.load_state_dict(model.state_dict())
        torch.save(best_model, 'best_model.pth')
        best_epoch = epoch
        best_loss = val_loss

# ============================================================

# Train the model AGAIN
#train_loss = []
#validation_loss = []
#best_loss = 10000
#best_epoch = 0
learning_rate = 1e-6
num_epochs = 100
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0.0
    running_val_loss = 0.0
    for xx,yy,tt  in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        tt = tt.to(device)
        optimizer.zero_grad()
        outputs = model(xx, tt*0.)
        
        theta_misfit = criterion(outputs[:,:,0:1,:,:], yy[:,:,0:1,:,:])/criterion(outputs[:,:,0:1,:,:]*0., yy[:,:,0:1,:,:])  
        ustar_misfit = criterion(outputs[:,:,1:2,:,:], yy[:,:,1:2,:,:])/criterion(outputs[:,:,1:2,:,:]*0., yy[:,:,1:2,:,:]) 
        xi_f_misfit = criterion(outputs[:,:,2:3,:,:], yy[:,:,2:3,:,:])/criterion(outputs[:,:,2:3,:,:]*0., yy[:,:,2:3,:,:])
        
        loss = theta_misfit + ustar_misfit + xi_f_misfit
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    for xx,yy,tt  in test_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        tt = tt.to(device)
        optimizer.zero_grad()
        outputs = model(xx, tt*0.)
        
        theta_misfit = criterion(outputs[:,:,0:1,:,:], yy[:,:,0:1,:,:])/criterion(outputs[:,:,0:1,:,:]*0., yy[:,:,0:1,:,:])  
        ustar_misfit = criterion(outputs[:,:,1:2,:,:], yy[:,:,1:2,:,:])/criterion(outputs[:,:,1:2,:,:]*0., yy[:,:,1:2,:,:]) 
        xi_f_misfit = criterion(outputs[:,:,2:3,:,:], yy[:,:,2:3,:,:])/criterion(outputs[:,:,2:3,:,:]*0., yy[:,:,2:3,:,:])
        
        loss = theta_misfit + ustar_misfit + xi_f_misfit
        
        loss.backward()
        optimizer.step()
    
    for xx,yy,tt  in test_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        tt = tt.to(device)
        with torch.no_grad():
            outputs = model(xx, tt*0.)
        
            val_loss = criterion(outputs, yy)
        running_val_loss += val_loss.item()
    
    trn_loss = running_loss/len(train_loader)
    val_loss = running_val_loss/len(test_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {trn_loss:.3e}, Validation Loss: {val_loss:.3e}")
    
    train_loss.append(trn_loss)
    validation_loss.append(val_loss)
    
    if val_loss <= best_loss:
        best_model.load_state_dict(model.state_dict())
        torch.save(best_model, 'best_model.pth')
        best_epoch = epoch
        best_loss = val_loss
    torch.save(model, 'model.pth')

# ============================================================

torch.save(model, 'model.pth')
print('best epoch: ', best_epoch + 1)

plt.plot(train_loss, label="Train Loss")
plt.plot(validation_loss, label="Validation Loss")
plt.legend()
plt.yscale("log")
plt.title("Loss Curve")
plt.show()

# ============================================================

# Evaluate the model on the training and test sets
running_test_loss = 0.0
for xx,yy,tt  in test_loader:
    xx = xx.to(device)
    yy = yy.to(device)
    tt = tt.to(device)
    with torch.no_grad():
        outputs = model(xx, tt*0.)    
        test_loss = criterion(outputs, yy)
    running_test_loss += test_loss.item()
    
# Calculate the MSE test sets
test_mse = running_test_loss/len(test_loader)
print(f"Test MSE: {test_mse:.4f}")

# Plot predictions for a few samples from the test set
sample = 0
for xx,yy,tt  in test_loader:
    sample = sample + 1
    xx = xx.to(device)
    yy = yy.to(device)
    tt = tt.to(device)
    with torch.no_grad():
        outputs = model(xx, tt*0.)    
    
    fire_location_pred = outputs[0,0,2,:,:]
    fire_location_pred = fire_location_pred.detach().cpu().numpy()
    fire_location_true = yy[0,0,2,:,:]
    fire_location_true = fire_location_true.detach().cpu().numpy()
    print("sample: ", sample)
    plt.imshow(fire_location_pred.T)
    plt.colorbar(orientation='horizontal')
    plt.title("Fire Location Prediction")
    plt.show()
    plt.imshow(fire_location_true.T)
    plt.colorbar(orientation='horizontal')
    plt.title("Fire Location True")
    plt.show()
    err = (fire_location_true.T - fire_location_pred.T)
    plt.imshow(err)
    plt.colorbar(orientation='horizontal')
    plt.title("Fire Location Error")
    plt.show()
        
    if sample >= 5:
        break

# ============================================================

# Submission generation 
# 20 autoregression steps for 27 test samples
y_preds = {}
ids = []
for idx in range(len(test_df)):
    theta, ustar, xi_f, uin, alpha, id = load_dataX(idx, test_df, 'test')
    
    theta = torch.Tensor(theta).unsqueeze(1)
    ustar = torch.Tensor(ustar).unsqueeze(1)
    xi_f = torch.Tensor(xi_f).unsqueeze(1)
    
    uin_tensor = torch.zeros_like(xi_f) + uin
    alpha_tensor = torch.zeros_like(xi_f) + alpha
    
    TUXUA = torch.cat([theta,ustar,xi_f, uin_tensor, alpha_tensor], dim=1)
    TUXUA = TUXUA.unsqueeze(0)
    
    xx = TUXUA[:,-history:,...]
    xx = xx.to(device)
    
    fire_loc = []
    for i in range(20):
        with torch.no_grad():        
            output = model(xx, torch.zeros(1).to(device)) 
            Temp = torch.cat([output,xx[:,-1:,-2:,:,:]], dim=2) 
            xx = torch.cat([xx[:,1:,:,:,:], Temp], dim=1)
            fire_loc.append(output[0,:,2,:,:])
    
    fire_location20 = torch.cat(fire_loc, dim=0)
    fire_location20 = fire_location20.detach().cpu().numpy().flatten(order='C').astype(np.float32)
    
    y_preds[id]= fire_location20
    ids.append(id)

df = pd.DataFrame.from_dict(y_preds,orient='index')
df['id'] = ids
#df.info()

#move id to first column
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
#reset index
df = df.reset_index(drop=True)

#df.head()
df.info()
df.to_csv('submission.csv',index=False)
print('Generated Submission file' )
