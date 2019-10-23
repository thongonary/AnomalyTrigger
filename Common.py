#!/usr/bin/env python
# coding: utf-8

import os
import torch
import uproot
import glob
import torchvision
import numpy as np
import  matplotlib 
matplotlib.use("Agg")
from collections import OrderedDict
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle

from torch.utils.data import Dataset, DataLoader
print(uproot.__version__) # Need latest uproot v3.7.1 for LazzyArrays


# ## A class for load in data from ROOT file, using uproot
# 
# It should be generic for all kind of flattree
# LazzyArrays is very new for uproot. Need more testing for performances

#folder = "~/Data/Phaes2L1Ntuple/"
folder = "/eos/cms/store/group/cmst3/group/l1tr/cepeda/triggerntuples10X/NeutrinoGun_E_10GeV/"
sig_folder = "/eos/cms/store/user/dsperka/"
bg_files  = "%s/NeutrinoGun_E_10GeV_V7_5_2_MERGED.root" % folder
sg_files  = "%s/VBF_HToInvisible_M125_14TeV_pythia8_PU200_V7_4_2.root" % sig_folder
sg_files2 = "%s/VBFHToBB_M-125_14TeV_powheg_pythia8_weightfix_V_7_5_2.root" % sig_folder
sg_files3 = "%s/GluGluToHHTo4B_node_SM_14TeV-madgraph_V7_5_2.root" % sig_folder

sampleMap   = {
    "BG"   :  
    {
        "file" : bg_files,
        "histtype" : 'bar', 
        "label" : 'BG', 
        "color" : 'yellow', 
    },
    'HtoInvisible' :
    {
        "file" :  sg_files,
        "histtype" : 'step', 
        "label" : 'HtoInvisible', 
        "color" : 'r', 
    },
    'VBFHToBB' : 
    {
        "file" :  sg_files2,
        "histtype" : 'step', 
        "label" : 'VBFHToBB', 
        "color" : 'g', 
    },
    'GluGlutoHHto4B' : 
    {
        "file" :  sg_files3,
        "histtype" : 'step', 
        "label" : 'GluGlutoHHto4B', 
        "color" : 'b', 
    },
}

batch_size = 2000 #144
num_epochs = 100
learning_rate = 1e-3
trainingfrac = 0.8
globalcutfunc = None

class P2L1NTP(Dataset):
    def __init__(self, dir_name, features = None,
                 tree_name="l1PhaseIITree/L1PhaseIITree",
                 sequence_length=50, verbose=False,
                 cutfunc =None):
        print("Initiating Dataset from {}".format(dir_name))
        self.tree_name = tree_name
        self.features = features
        self.sequence_length = sequence_length
        self.file_names = glob.glob(dir_name)
        ## Cache will be needed in case we train with >1 eposh
        ## Having issue and reported in https://github.com/scikit-hep/uproot/issues/296
        self.cache = uproot.cache.ArrayCache(1024**3)
        self.upTree = uproot.lazyarrays(self.file_names, self.tree_name, self.features.keys(), cache=self.cache)
        self.cutfunc = cutfunc
        if self.cutfunc is not None:
            self.upTree = self.upTree[self.cutfunc(self.upTree)]

    def __len__(self):
        if self.cutfunc is None:
            return uproot.numentries(self.file_names, self.tree_name, total=True)
        else:
            return len(self.upTree)

    def __getitem__(self, idx):
        print("getting {}".format(idx))
        reflatnp = []
        event = self.upTree[idx]
        for b, v in self.features.items():
            g  = event[b]
            ln = v[0]
            scale = v[1]
            if isinstance(g,float) == True:
                tg = np.array([g])
            else:
                if len(g)>= ln:
                    tg = g[:ln]
                else:
                    tg = np.pad(g, (0, ln-len(g)), 'constant', constant_values=0)

            if scale > 10 :
                tg = tg / scale
            elif scale > 1 :
                tg = tg + scale
            reflatnp.append(tg)
        org = np.concatenate(reflatnp, axis=0).astype(np.float32)
        return org

    def GetCutArray(self, cutfunc):
        select = cutfunc(self.upTree)
        sel = np.array(select)[np.newaxis]
        nfeatures = sum([v[0] for b, v in self.features.items()])
        ones = np.ones((self.__len__(), nfeatures))
        out = np.multiply(ones, sel.T)
        return out

def EvalLoss(samplefile, PhysicsObt, model, criterion, cut=None):
    sample = P2L1NTP(samplefile, PhysicsObt)
    dataloader = DataLoader(sample, batch_size=batch_size, pin_memory=True, shuffle=False)
    for batch_idx, vbg_data in enumerate(dataloader):
        _vbg_img = Variable(vbg_data.type(torch.FloatTensor))
        if torch.cuda.is_available():
            _vbg_img = _vbg_img.cuda()

        vout = model(_vbg_img)
        vloss = criterion(vout, _vbg_img)
        _vbg_out = vout.cpu().detach().numpy()
        _vbg_loss = vloss.cpu().detach().numpy()
        if batch_idx == 0:
            vbg_out = _vbg_out
            vbg_loss = _vbg_loss
        else:
            vbg_loss = np.append([vbg_loss],[_vbg_loss])
            vbg_out = np.concatenate((vbg_out,_vbg_out))

    if cut is not None:
        cutmask = sample.GetCutArray(cut)
        vbg_loss = np.multiply(vbg_loss, cutmask.flatten())
        vbg_out = np.multiply(vbg_out, cutmask)

    return vbg_loss

def DrawLoss(modelname, lossMap, features):
    plt.figure(figsize=(8,6))
    bins = np.linspace(0, 30, 60)
    for k, v in lossMap.items():
        reshape_vbg_loss = np.reshape(v, (-1,features))
        vloss = np.sum(reshape_vbg_loss, axis=1).flatten()
        plt.hist(vloss,bins,label=sampleMap[k]['label'],  
                 histtype=sampleMap[k]['histtype'],  
                 color=sampleMap[k]['color'],  normed=True)
    plt.legend(loc='best',fontsize=16)
    plt.xlim(-1,30)
    plt.xlabel('Reconstruction Loss', fontsize=16)
    plt.savefig("%s_Loss.png" % modelname)
    plt.yscale("log")
    plt.savefig("%s_LogLoss.png" % modelname)


def DrawROC(modelname, lossMap, features):
    plt.figure(figsize=(8,6))
    reshape_bg_loss = np.reshape(lossMap["BG"], (-1,features))
    bloss = np.sum(reshape_bg_loss, axis=1).flatten()
    for k, v in lossMap.items():
        if k == "BG":
            continue
        if len(v.shape) > 1:
            reshape_vbg_loss = np.reshape(v, (-1,features))
            vloss = np.sum(reshape_vbg_loss, axis=1).flatten()
        else: 
            vloss = v.flatten()
        Tr = np.concatenate(( np.zeros_like(bloss), np.ones_like(vloss)), axis=0)
        Loss = np.concatenate((bloss, vloss), axis=0)
        fpr, tpr, thresholds = roc_curve(Tr, Loss)
        roc_auc = auc(fpr, tpr)
        rate = fpr * 40*1000
        plt.plot(rate, tpr, color=sampleMap[k]['color'],
                 lw=2, label='%s (AUC = %0.2f)' % (sampleMap[k]['label'], roc_auc))
    plt.legend(loc='best',fontsize=16)
    plt.xlabel('Rate (kHz)', fontsize=16)
    plt.ylabel('Signal Efficiency', fontsize=16)
    plt.savefig("%s_ROC.png" % modelname)
    plt.xlim(0,300)
    plt.ylim(0,1.)
    plt.grid(True)
    plt.savefig("%s_ROCZoom.png" % modelname)
