
# - data preprocess
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='.')

x = mnist.data / 255
y = mnist.target


# - create dataloader
# 正規化したデータをニューラルネット用の変数に変換



import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

if __name__ == "__main__":
    model = Net(n_in=)