#!/home/batuhangundogdu/ai8x-training/venv/bin/python3
###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
#
###################################################################################################

import os
import numpy as np
from DML_functions import DML, MaskedXentMarginLoss, get_batch
from torch.optim import Adam
from config import *
import torch
from tester import tester



def main():
    
    print('Loading training data...')
    loaded = np.load('data/embeddings_for_DML.npz')
    data = loaded['embeddings']
    targets = loaded['words']
    speakers = loaded['speaker_id']
    
    print('Loading test data...')
    data_address = 'data/test_features.npz'
    data_test = np.load(data_address, allow_pickle=True)
    test_features = data_test.f.test_features.item()
    background_embeddings = test_features['background']['wav2vec2']
    
    
    sigma = DML()
    DRL_PATH = 'models/DRL.pt'
    sigma = sigma.float().cuda().train()
    lr = 0.003
    margin = 0.5
    optimizer = Adam(sigma.parameters(), lr=lr)
    optimizer.zero_grad()
    loss_fn = MaskedXentMarginLoss(margin=margin)
    
    print('Training DML...')
    ctr = 0
    total_loss = 0
    for ep in range(5001):
        anchor, alien, labels = get_batch(data, targets, speakers, batch_size=128)
        optimizer.zero_grad()
        output = sigma.forward(anchor.float(), alien.float())
        loss = loss_fn(output, labels)
        total_loss += loss.item()
        ctr += 1
        if not ep%50:
            print(f'epoch = {ep}, loss = {total_loss/ctr}')
            total_loss = 0
            ctr = 0
            DML_embedding_bck = sigma.forward_one(torch.from_numpy(background_embeddings).float().cuda()).detach().cpu().numpy()
            test_features['background']['DML'] = DML_embedding_bck
            for keyword in keywords:
                keyword_embeddings = test_features[keyword]['wav2vec2']
                DML_embedding_keyword = sigma.forward_one(torch.from_numpy(keyword_embeddings).float().cuda()).detach().cpu().numpy()
                test_features[keyword]['DML'] = DML_embedding_keyword           
                #np.savez(data_address, test_features=test_features)
                # TODO : Do this at the very end
                tester(test_features, embedding = 'DML', keyword = keyword, ep=ep, detailed=False)
            print('tested')
            if not ep:
                lr *= 0.75
                optimizer = Adam(sigma.parameters(), lr=lr)
            #margin += 0.01
            #loss_fn = MaskedXentMarginLoss(margin=margin)
        loss.backward()
        optimizer.step()
        torch.save(sigma, DRL_PATH)
    
if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
    print('Done!')   
