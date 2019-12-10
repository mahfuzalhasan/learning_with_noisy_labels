from datetime import datetime
from tensorboardX import SummaryWriter


import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F


import os
import numpy as np
import time

import parameters as params
from dataset import Dataset
from model import CNN
from loss import loss_class_mean





def adjust_learning_rate(optimizer, epoch, alpha_plan, beta1_plan):
    
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999)


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(run_id, use_cuda, epoch, rate_schedule, noise_or_not, writer, train_loader, model1, optimizer1):


    pure_ratio_1_list=[]
    loss_net_1 = []
    accuracy_net_1 = []
    train_total=0
    train_correct=0 
    
    start_time = time.time()
    for i, (images, labels, indexes) in enumerate(train_loader):
      
        ind=indexes.cpu().numpy().transpose()

        if use_cuda:
            images = images.float().cuda()
            labels = labels.cuda()
      
        
        logits1, embeddings1 = model1(images)
        
        prec1,_ = accuracy(logits1, labels, topk=(1, 5))
        accuracy_net_1.append(prec1.item())

        train_total+=1
        train_correct+=prec1

        loss_1, pure_ratio_1 = loss_class_mean(logits1, labels, rate_schedule[epoch], ind, noise_or_not, embeddings1, epoch) 

        pure_ratio_1_list.append(100*pure_ratio_1)
        
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

        loss_net_1.append(loss_1.item())

        if i % params.print_freq == 0:
            print ('Epoch [%d/%d], Batch [%d] Training Accuracy1: %.4F, Loss1: %.4f, Pure Ratio: %.4f' 
                  %(epoch, params.n_epoch, i, np.mean(accuracy_net_1), np.mean(loss_net_1), \
                    np.mean(pure_ratio_1_list)))

    time_taken = time.time() - start_time

    print('epoch ',epoch,' time taken: ', time_taken, '  Acc1: ',np.mean(accuracy_net_1),'  Loss1: ',np.mean(loss_net_1), \
        '  pure ratio1: ',np.mean(pure_ratio_1_list))
    

    save_dir = os.path.join(params.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    if epoch % params.save_frequency == 0: 
        save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
        states = {
            'epoch': epoch,
            'state_dict1': model1.state_dict(),
            'optimizer1': optimizer1.state_dict(),
        }
        torch.save(states, save_file_path)

    writer.add_scalar('Training Loss 1', np.mean(loss_net_1), epoch)
    writer.add_scalar('Training Accuracy 1', np.mean(accuracy_net_1), epoch)
    

    return np.mean(accuracy_net_1), pure_ratio_1_list, model1

def evaluate(run_id, use_cuda, epoch, writer, test_loader, model):
    
    model.eval()
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for i, (images, labels, indexes) in enumerate(test_loader):

        if use_cuda:
            images = images.float().cuda()
            # labels = labels.cuda()

      
        
        
        # Forward + Backward + Optimize
        logits, _ = model(images)


        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)


        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()

    acc = float(correct)/float(total)


    time_taken = time.time() - start_time
    print('epoch ',epoch,' time taken: ',time_taken, '  Acc: ',acc)
    writer.add_scalar('Validation Accuracy ', acc, epoch)
    



def train_classifier(run_id, use_cuda):
    writer = SummaryWriter(os.path.join(params.logs_dir, str(run_id)))


    dataset_info = Dataset(run_id, params.dataset)


    train_dataset = dataset_info.train_dataset
    test_dataset = dataset_info.test_dataset

    noise_or_not = train_dataset.noise_or_not
    
    if params.forget_rate is None:
        forget_rate=params.noise_rate
    else:
        forget_rate = params.forget_rate


    # Adjust learning rate and betas for Adam Optimizer
    mom1 = 0.9
    mom2 = 0.1
    alpha_plan = [params.learning_rate] * params.n_epoch
    beta1_plan = [mom1] * params.n_epoch


    for i in range(params.epoch_decay_start, params.n_epoch):
        alpha_plan[i] = float(params.n_epoch - i) / (params.n_epoch - params.epoch_decay_start) * params.learning_rate
        beta1_plan[i] = mom2


    # define drop rate schedule
    rate_schedule = np.ones(params.n_epoch) * forget_rate
    rate_schedule[:params.num_gradual] = np.linspace(0, forget_rate**params.exponent, params.num_gradual)
    #print('rate_schedule: ',rate_schedule)


    saved_model = None
    cnn1 = CNN(input_channel=dataset_info.input_channel, n_outputs=dataset_info.num_classes)
    if saved_model is not None:
        cnn1.load_state_dict(torch.load(saved_model)['state_dict1'])
        print('model loaded from: ',saved_model)
        
    if use_cuda:
        cnn1.cuda()

    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=params.learning_rate)


    for epoch in range(params.n_epoch):

        train_dataloader = DataLoader(train_dataset, batch_size = params.batch_size, shuffle=True, num_workers=4)
        print('train dataloader: ',len(train_dataloader),flush=True)
        
        accuracy_1, pure_ratio_1_list, model1 = train(run_id, use_cuda, \
            epoch, rate_schedule, noise_or_not, writer, train_dataloader, cnn1, optimizer1)
        
        adjust_learning_rate(optimizer1, epoch, alpha_plan, beta1_plan)

        test_dataloader = DataLoader(test_dataset, batch_size = params.batch_size, shuffle=True, num_workers=4)
        print('valid dataloader: ',len(test_dataloader),flush=True)
        evaluate(run_id, use_cuda, epoch, writer, test_dataloader, model1)


if __name__ == "__main__":
    
    run_started = datetime.today().strftime('%d-%m-%y_%H%M')
    use_cuda = torch.cuda.is_available()
    print('USE_CUDA: ',use_cuda,flush=True)
    print('run id: ',run_started)
    train_classifier(run_started, use_cuda)