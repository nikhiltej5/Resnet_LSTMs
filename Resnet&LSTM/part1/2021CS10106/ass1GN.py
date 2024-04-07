import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import gc
import argparse

# Define data transforms
transform = transforms.Compose([
    transforms.RandomCrop(256, padding=32, padding_mode='reflect'), 
    transforms.ToTensor(),
    transforms.Normalize((0.4722, 0.4815, 0.4019),(0.2427, 0.2408, 0.2654))
])


transform_val = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.4722, 0.4815, 0.4019),(0.2427, 0.2408, 0.2654))
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes =  ['Asian-Green-Bee-Eater', 'Brown-Headed-Barbet', 'Cattle-Egret', 'Common-Kingfisher', 'Common-Myna', 'Common-Rosefinch', 'Common-Tailorbird', 'Coppersmith-Barbet', 'Forest-Wagtail', 'Gray-Wagtail', 'Hoopoe', 'House-Crow', 'Indian-Grey-Hornbill', 'Indian-Peacock', 'Indian-Pitta', 'Indian-Roller', 'Jungle-Babbler', 'Northern-Lapwing', 'Red-Wattled-Lapwing', 'Ruddy-Shelduck', 'Rufous-Treepie', 'Sarus-Crane', 'White-Breasted-Kingfisher', 'White-Breasted-Waterhen', 'White-Wagtail']

class GroupNorm(nn.Module):
    def __init__(self,groups, num_features, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.eps = eps
        self.groups = groups
        self.shape = num_features
        self.weight = nn.Parameter(torch.ones(self.shape)).to(device) ## TODO
        self.bias = nn.Parameter(torch.zeros(self.shape)).to(device) ## TODO

    def forward(self, input):
        
        input_size = input.shape
        dimension = input.shape[2]
        
        input = input.view(self.groups,self.shape,dimension,-1)

        # calculate running estimates
        dims = [2,3]
        mean = input.mean(dims)
        # use biased var in train
        var = input.var(dims, unbiased=False)

        input = (input - mean[:, :, None, None]) / (torch.sqrt(var[:, :, None, None] + self.eps))
        input = input * self.weight[None,:,None,None]  + self.bias[None,:,None,None]
        input = input.view(input_size)
        return input
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)
        identity = x
        out = self.conv1(x)
        shape1 = out.shape[1]
        out = GroupNorm(8,shape1)(out)
        out = self.relu(out)
        out = self.conv2(out)
        shape1 = out.shape[1]
        out = GroupNorm(8,shape1)(out)
        
        out += identity
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=25):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32,layers[1])
        self.layer3 = self._make_layer(block, 64,layers[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=3, stride=2,padding=1),
                GroupNorm(8,planes)
            )

        layers = []
        self.inplanes = planes
        layers.append(block(self.inplanes, planes,1, downsample))
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        shape1 = x.shape[1]
        x = GroupNorm(8,shape1)(x)
        x = self.relu(x)
       
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1) 
        x = self.fc(x)

        return x
    
class Model(nn.Module):
    def __init__(self,num_layers,learning_rate=0.0001):
        super(Model, self).__init__()
        self.model = ResNet(ResBlock, [num_layers, num_layers, num_layers]).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=learning_rate,weight_decay=0.0001)

def main(args):
    val_data_dir = args.test_data_file

    # Create datasets
    val_dataset = ImageFolder(root=val_data_dir, transform=transform_val)

    batch_size = 32
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Model(args.n).to(device)
    model.load_state_dict(torch.load(args.model_file))

    #eval 
    with torch.no_grad():
        with open(args.output_file, "w") as f:
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images).to(device)
                _, predicted = torch.max(outputs, 1)
                m = labels.shape[0]
                for i in range(m):
                    f.write(f'{classes[predicted[i]]}\n')
                del images, labels, outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--n', type=int, default=3, help='Number of layers in ResNet')
    parser.add_argument('--model_file', type=str, default='model.pth', help='Path to model file')
    parser.add_argument('--test_data_file', type=str, default='test', help='Path to test data')
    parser.add_argument('--output_file', type=str, default='output.txt', help='Path to output file')
    args = parser.parse_args()
    main(args)