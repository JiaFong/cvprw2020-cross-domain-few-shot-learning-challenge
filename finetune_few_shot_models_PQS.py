import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot

from methods.protonet import ProtoNet
from methods.relationnet import RelationNet

from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 
from utils import *

from pseudo_query_generator import PseudoQeuryGenerator
from methods.loss import PrototypeTripletLoss, Arcface, LargeMarginCosineLoss




def finetune(novel_loader, n_query = 15, pretrained_dataset='miniImageNet', n_pseudo=100, data_parallel=False, n_way = 5, n_support = 5): 
    correct = 0
    count = 0

    iter_num = len(novel_loader) 

    acc_all = []
    for ti, (x, y) in enumerate(novel_loader):

        ###############################################################################################
        # load pretrained model on miniImageNet
        if params.method in ['protonet_ptl', 'protonet']:
            ptl   = False if params.method == 'protonet' else True
            pretrained_model = ProtoNet( model_dict[params.model], ptl=ptl,  n_way = n_way, n_support = n_support  )
        elif params.method == 'relationnet':
            feature_model = lambda: model_dict[params.model]( flatten = False )
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
            pretrained_model = RelationNet( feature_model, loss_type = loss_type , n_way = n_way, n_support = n_support )

        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, pretrained_dataset, params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'
        if not params.method in ['baseline'] :
            checkpoint_dir += '_%dway_5shot' %(params.train_n_way)

        params.save_iter = -1
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
        elif params.method in ['baseline', 'baseline++'] :
            modelfile   = get_resume_file(checkpoint_dir)
        else:
            modelfile   = get_best_file(checkpoint_dir)

        tmp = torch.load(modelfile)
        state = tmp['state']
        pretrained_model.load_state_dict(state)
        ###############################################################################################

        ###############################################################################################
        # data processing, slice data into support set and query set
        n_query = x.size(1) - n_support
        


        x = x.cuda()
        x_var = Variable(x)

        support_size = n_way * n_support 
       
        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda()    # (25,)

        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # query set
        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:])  # support set
 
        ###############################################################################################

        ###############################################################################################
        # Finetune components initialization 

        pseudo_q_genrator  = PseudoQeuryGenerator(n_way, n_support,  n_pseudo)

        delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, pretrained_model.parameters()))

        pretrained_model.cuda() 
        if data_parallel:
            pretrained_model = nn.DataParallel(pretrained_model, device_ids=[0,1])
        ###############################################################################################

        ###############################################################################################
        # finetune process 
        finetune_epoch = 100
        
        fine_tune_n_query = n_pseudo // n_way
        pretrained_model.n_query = fine_tune_n_query

        pretrained_model.train()
        z_support = x_a_i.view(n_way, n_support, *x_a_i.size()[1:])        
        for epoch in range(finetune_epoch):
            delta_opt.zero_grad()

            # generate pseudo query images
            psedo_query_set, _ = pseudo_q_genrator.generate(x_a_i)
            psedo_query_set = psedo_query_set.cuda().view(n_way, fine_tune_n_query,  *x_a_i.size()[1:])

            x = torch.cat((z_support, psedo_query_set), dim=1)

            loss = pretrained_model.set_forward_loss(x)
            loss.backward()
            delta_opt.step()

        del loss, psedo_query_set
        torch.cuda.empty_cache()
        ###############################################################################################

        ###############################################################################################
        # inference process 
        #pretrained_model.eval()  for transductive inference.

        # check if the support set is modified unexpected.
        x_a_i_c = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:])
        if False in (x_a_i_c == x_a_i):
            print('support set changed!')
        pretrained_model.n_query = n_query

        with torch.no_grad():
            scores = pretrained_model.set_forward(x_var.cuda())

            if type(scores) is tuple: #when apply ptloss in meta-training
                scores = scores[0]
        
        y_query = np.repeat(range( n_way ), n_query )
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()

        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)

        acc_all.append((correct_this/ count_this *100))        
        print("Task %d : %4.2f%%  Now avg: %4.2f%%" %(ti, correct_this/ count_this *100, np.mean(acc_all) ))
        ###############################################################################################
    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    ##################################################################
    image_size = 224
    iter_num = 600

    n_query = max(1, int(16* params.test_n_way/params.train_n_way))
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
   

    # number of pseudo images
    n_pseudo = 100 if params.n_shot == 5 else 200
    
    # apply dataparallel or not
    data_parallel = False
    if params.model == 'ResNet18' and params.n_shot == 50:
        data_parallel = True

    ##################################################################
    pretrained_dataset = "miniImageNet"

    dataset_names = ["CropDisease", "EuroSAT", "ISIC", "ChestX"]
    novel_loaders = []

    print ("Loading CropDisease")
    datamgr             =  CropDisease_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    novel_loaders.append(novel_loader)

    print ("Loading EuroSAT")
    datamgr             =  EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    novel_loaders.append(novel_loader)
    
    print ("Loading ISIC")
    datamgr             =  ISIC_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    novel_loaders.append(novel_loader)

    print ("Loading ChestX")
    datamgr             =  Chest_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    novel_loaders.append(novel_loader)
    
    print("n_pseudo: ", n_pseudo)
    print("data_parallel: ", data_parallel)
    #########################################################################
    for idx, novel_loader in enumerate(novel_loaders):
        print (dataset_names[idx])
        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch
      
        # replace finetine() with your own method
        finetune(novel_loader, n_query = 15, pretrained_dataset=pretrained_dataset, n_pseudo=n_pseudo, data_parallel=data_parallel, **few_shot_params)
