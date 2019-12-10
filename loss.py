import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import deque


class_dict = {
    0: deque(maxlen=100),
    1: deque(maxlen=100),
    2: deque(maxlen=100),
    3: deque(maxlen=100),
    4: deque(maxlen=100),
    5: deque(maxlen=100),
    6: deque(maxlen=100),
    7: deque(maxlen=100),
    8: deque(maxlen=100),
    9: deque(maxlen=100),
}

# Loss functions
def loss_coteaching(y_1, t, forget_rate, ind, noise_or_not, embedd, epoch):
    # print('t: ',t)
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    #print('ind: ',ind)
    #print('loss_1 : ',loss_1.size())
    #print('loss_1: ',loss_1)
    ind_1_sorted = np.argsort(loss_1.cpu().data)
    #print('ind_1_sorted: ',ind_1_sorted)
    loss_1_sorted = loss_1[ind_1_sorted]
    #print('loss_1_sorted: ',loss_1_sorted)
    class_menas = []
    class_std = []
    #import pdb; pdb.set_trace()
    embedd = embedd.detach()
    mean_holder = torch.zeros(size=tuple(embedd.shape)).cuda()
    std_holder = torch.zeros(size=tuple(embedd.shape)).cuda()

    for iterator in range(10):
        if len(t.eq(iterator).nonzero().squeeze(1)) > 0:
            class_dict[iterator].append(torch.mean(embedd[t.eq(iterator).nonzero().squeeze(1)], 0))
    
    #if forget_rate == 0.2:
    valid_class =[]
    if epoch > 20:
        #import pdb; pdb.set_trace()
        for iterator in range(10):
            if len(class_dict[iterator]) > 0:
                valid_class.append(iterator)
                class_menas.append(torch.mean(torch.stack(list(class_dict[iterator])), 0))
                class_std.append(torch.std(torch.stack(list(class_dict[iterator])), 0))
        #import pdb; pdb.set_trace()
        for iterator in range(10):
            if len(t.eq(iterator).nonzero().squeeze(1)) > 0:
                mean_holder[t.eq(iterator).nonzero().squeeze(1)] = class_menas[valid_class.index(iterator)]
                std_holder[t.eq(iterator).nonzero().squeeze(1)] = class_std[valid_class.index(iterator)]
        
        std_diff = torch.sum((std_holder - torch.abs(embedd - mean_holder)), 1).cpu().numpy()

        selected_ind = np.where(std_diff>=0)[0]

    '''
    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.cpu().data)
    
    loss_2_sorted = loss_2[ind_2_sorted]
    '''

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    #print('ind[]: ',ind[ind_1_sorted[:num_remember]])       #data point for which loss was small

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    #pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)


    ind_1_update = ind_1_sorted[:num_remember]
    #ind_2_update=ind_2_sorted[:num_remember]

    # if forget_rate == 0.2:
    if epoch > 20:
        ind_1_update = np.union1d(ind_1_update.numpy(), selected_ind)
    

    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_1_update], t[ind_1_update])
    #loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])    

    return torch.sum(loss_1_update)/num_remember, pure_ratio_1


