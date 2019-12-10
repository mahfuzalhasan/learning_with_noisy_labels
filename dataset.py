# load dataset
import parameters as params
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST


import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

class Dataset(object):
    def __init__(self, run_id, dataset_name, test=False):
        super(Dataset,self).__init__()
        self.dataset_name = dataset_name
        self.load_data()

    def load_data(self):
        if self.dataset_name.lower() =='mnist':
            self.input_channel = 1
            self.num_classes = 10
            params.top_bn = False
            params.epoch_decay_start = 80
            params.n_epoch = 200
            self.train_dataset = MNIST(root='./data/',
                                        download = True,  
                                        train = True, 
                                        transform = transforms.ToTensor(),
                                        noise_type = params.noise_type,
                                        noise_rate = params.noise_rate
                                )
            
            self.test_dataset = MNIST(root='./data/',
                                    download=True,  
                                    train=False, 
                                    transform=transforms.ToTensor(),
                                    noise_type = params.noise_type,
                                    noise_rate = params.noise_rate
                                )

        elif self.dataset_name.lower() == 'cifar10':
            self.input_channel=3
            self.num_classes=10
            params.top_bn = False
            params.epoch_decay_start = 80
            params.n_epoch = 500
            self.train_dataset = CIFAR10(root='./data/',
                                        download=True,  
                                        train=True, 
                                        transform=transforms.ToTensor(),
                                        noise_type=params.noise_type,
                                        noise_rate=params.noise_rate
                                )
            
            self.test_dataset = CIFAR10(root='./data/',
                                        download=True,  
                                        train=False, 
                                        transform=transforms.ToTensor(),
                                        noise_type=params.noise_type,
                                        noise_rate=params.noise_rate
                                )


        elif  self.dataset_name.lower() =='cifar100':
            self.input_channel=3
            self.num_classes=100
            params.top_bn = False
            params.epoch_decay_start = 100
            params.n_epoch = 200
            self.train_dataset = CIFAR100(root='./data/',
                                        download=True,  
                                        train=True, 
                                        transform=transforms.ToTensor(),
                                        noise_type=params.noise_type,
                                        noise_rate=params.noise_rate
                                    )
            
            self.test_dataset = CIFAR100(root='./data/',
                                        download=True,  
                                        train=False, 
                                        transform=transforms.ToTensor(),
                                        noise_type=params.noise_type,
                                        noise_rate=params.noise_rate
                                    )