#!/home/batuhangundogdu/ai8x-training/venv/bin/python

import numpy as np
import ai8x
from ai8x_student import KWSInfNet
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
from tester import tester

#import os
#from config import *
#import matplotlib.pyplot as plt

def save_students(model, test_features, keyword='hey_snapdragon'):
    
    """
    This function creates the student embeddings for the keyword using the ai8x student model
    """
    background_waveform = test_features['background']['raw']
    emb_bck = model(torch.from_numpy(background_waveform).float().cuda().permute(0, 2, 1)).detach().cpu().numpy()
    test_features['background']['ai8x'] = emb_bck
    keyword_waveform = test_features[keyword]['raw']
    emb_kwrd = model(torch.from_numpy(keyword_waveform).float().cuda().permute(0, 2, 1)).detach().cpu().numpy()
    test_features[keyword]['ai8x'] = emb_kwrd
    



def main():
    
    logname = 'training_log'
    logging.basicConfig(filename=logname, level=logging.INFO)
    keyword='hey_snapdragon'
    print('Loading the teacher training set, this can take a few mins...')
    #TODO: exception rule to check if the npz exists
    data = np.load('data/embeddings_for_student.npz', allow_pickle=True)
    student_input = data.f.student_input
    embeddings = data.f.embeddings
    
    print('Loading test data...')
    data_address = 'data/test_features.npz'
    data_test = np.load(data_address, allow_pickle=True)
    test_features = data_test.f.test_features.item()
    
    act_mode_8bit = False
    avg_pool_rounding = True
    ai8x.set_device(85, act_mode_8bit, avg_pool_rounding)
    model = KWSInfNet().cuda()
    #save_students(model, test_features)
    
    lr = 3e-5
    batch_size = 256
    num_epoochs = 5001
    
    loss_fn = torch.nn.MSELoss()
    cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=5, verbose=True)

    
    
    training_student_input = torch.from_numpy(student_input[:600_000])
    training_embeddings = torch.from_numpy(embeddings[:600_000])
    valid_student_input = torch.from_numpy(student_input[600_000:])
    valid_embeddings = torch.from_numpy(embeddings[600_000:])
    
    training_dataset = TensorDataset(training_student_input, training_embeddings)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size)
    valid_dataset = TensorDataset(valid_student_input, valid_embeddings)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    
    for ep in tqdm(range(num_epoochs)):
        _loss = 0
        v_loss = 0
        for f, emb in training_dataloader:
            x = f.float().cuda().permute(0, 2, 1)
            y = model(x)
            optimizer.zero_grad()
            #sim = cosine_similarity(y, emb.float().cuda())
            loss = loss_fn(y, emb.float().cuda()) #0.1*(1 - sim.mean()) + 
            loss.backward()
            optimizer.step()
            _loss += loss.item()
        for f, emb in valid_dataloader:
            x = f.float().cuda().permute(0, 2, 1)
            y = model(x)
            loss = loss_fn(y, emb.float().cuda())
            #sim = cosine_similarity(y, emb.float().cuda())
            #loss = 0.1*(1 - sim.mean()) + loss_fn(y, emb.float().cuda())
            v_loss += loss.item()
            
        scheduler.step(v_loss)     
        logging.info(f'{ep}, {_loss/len(training_dataloader):.5f}, {v_loss/len(valid_dataloader):.5f}')
        background_waveform = test_features['background']['raw']
        if not ep%100:
            emb_bck = model(torch.from_numpy(background_waveform).float().cuda().permute(0, 2, 1)).detach().cpu().numpy()
            test_features['background']['ai8x'] = emb_bck
            keyword_waveform = test_features[keyword]['raw']
            emb_kwrd = model(torch.from_numpy(keyword_waveform).float().cuda().permute(0, 2, 1)).detach().cpu().numpy()
            test_features[keyword]['ai8x'] = emb_kwrd
            tester(test_features, embedding = 'ai8x', keyword = 'hey_snapdragon', ep=ep)
            PATH = 'models/KWSNetv4.pt'
            torch.save(model, PATH)
            variable_address = 'data/test_features.npz'
            np.savez(variable_address, test_features=test_features)
            

'''

--------------------------------------------------------------
        
            fig, ax = plt.subplots(3,1)
            for i in range(3):
                x = train_student_input[1000*i].unsqueeze(dim=0).float().cuda().permute(0, 2, 1)
                y = model(x)
                ax[i].plot(np.squeeze(y.cpu().detach().numpy()), label = 'student') 
                ax[i].plot(np.squeeze(train_embeddings[1000*i].cpu().detach().numpy()), label='teacher') 
                ax[i].legend()
            fig.savefig(f'graphs/sample_comparison_{ep}.png')
            
            fig2, ax2 = plt.subplots(3,1)
            for i in range(3):
                x = valid_student_input[1000*i].unsqueeze(dim=0).float().cuda().permute(0, 2, 1)
                y = model(x)
                ax2[i].plot(np.squeeze(y.cpu().detach().numpy()), label = 'student') 
                ax2[i].plot(np.squeeze(valid_embeddings[1000*i].cpu().detach().numpy()), label='teacher') 
                ax2[i].legend()
            fig2.savefig(f'graphs/valid_sample_comparison_{ep}.png')
            save_students(model)
            tester(ep=ep)

            

            
    
    


        medoids = np.empty((train_DML_embeddings.shape[1], num_classes), dtype=np.float32)
        for _class in range(num_classes):
            class_inx = (np.squeeze(train_class) == _class)
            class_samples = train_DML_embeddings[class_inx,:]
            medoids[:,_class] = np.mean(class_samples, axis=0)
            
               
        test = torch.from_numpy(test_student_data).float().cuda().permute(0, 2, 1)
        
        for training_size, batch_size, epochs in curriculum:
            lr *=0.9
            best_acc = 0
            print(f'Curriculum training size = {training_size}, batch size = {batch_size}')             
            raw_audio = torch.from_numpy(train_student_data[:training_size])
            embedding = torch.from_numpy(train_DML_embeddings[:training_size])
            
            
            
            ep = 0
            patience = 0
            while True:
            
                test_embeddings = model(test).detach().cpu().numpy()
                acc = calculate_scores(test_embeddings, medoids, test_class)
                
                
                    
            
                if (ep > 150) and training_size != -1:
                    if acc > best_acc:
                        best_acc = acc
                        patience = 0
                    else :
                        patience += 1
                        
                    if patience > 20:
                        break
                ep += 1
        torch.save(model, model_name)
    else:
        print('Please run the prepare_embeddings.py first!')
        return 1
    torch.save(model, model_name)
    
    
'''
    
if __name__ == "__main__":
    main()
    print('Done!')
         
            




        

