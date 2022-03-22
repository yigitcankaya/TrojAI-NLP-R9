
import torch
import os
import numpy as np

def read_model(df, model_idx, main_path, models_path):
    model_name = df['model_name'][model_idx]
    model_dirpath = os.path.join(models_path, model_name)
    arch = df['model_architecture'][model_idx]
    if arch == 'roberta-base':
        tokenizer_filepath = os.path.join(main_path,'tokenizers/tokenizer-roberta-base.pt')
    if arch == 'google/electra-small-discriminator':
        tokenizer_filepath = os.path.join(main_path,'tokenizers/tokenizer-google-electra-small-discriminator.pt')
    if arch == 'distilbert-base-cased':
        tokenizer_filepath = os.path.join(main_path,'tokenizers/distilbert-base-cased.pt')
    poisoned = df['poisoned'][model_idx]
    model_filepath = os.path.join(model_dirpath, 'model.pt')
    examples_dirpath = os.path.join(model_dirpath, 'example_data')
    return arch, poisoned, model_dirpath, model_filepath, tokenizer_filepath, examples_dirpath


def white_noise(model,dim = 768, sequence = True, config_json={}, device ='cpu'):
    #white noise 
    num_cls = 2
    # generate random pattern

    iter_ = config_json.get('white_noise_iters', 10)
    all_size = config_json.get('white_noise_size', 10000)

    # iter_ = 10
    # all_size = 10000
    
    #iter_ = 1
    #all_size = 100

    noise = {}
    stats = {}

    for cls in range(num_cls):
        noise[cls] = []
        stats[cls] = 0

    for i in range(iter_):#uniform -1,1 yaptık
        #print(os.system("nvidia-smi"))
        #sys.stdout.flush()
        torch.manual_seed(i)#to reproduce
        if sequence:
            z = torch.rand(all_size, 1, dim).to(device) * 2 - 1 ##why uniform [0,1)?? it is ok for image but maybe not ok for nlp
        else:
            z = torch.rand(all_size, dim).to(device) * 2 - 1
        #z = torch.tensor(np.random.uniform(-1.49,1.47,(all_size, 1, 768))).to(device).float()
        with torch.no_grad():
            out = model.classifier(z)
            z = z.reshape(all_size, 1, dim)
            pred = out.max(1)[1]

        for cls in range(num_cls):
            noise[cls].append(z[pred == cls].cpu())
            stats[cls] += (pred == cls).sum().cpu()
            
        out = None

    for cls in range(num_cls):
        noise[cls] = torch.cat(noise[cls])
        
    return noise,stats


def extract_features_by_embedding(df,main_path,models_path,config_json):
    features_g = []
    features_r = []
    features_d = []
    labels_g = []
    labels_r = []
    labels_d = []
    noises = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for idx in range(len(df)):
        if df['task_type_level'][idx] != 0:
            continue
        params = read_model(df, idx, main_path, models_path)
        model = torch.load(params[3], map_location=device).eval()
        
        if df['model_architecture'][idx] == 'google/electra-small-discriminator':
            noise,stats = white_noise(model,256,config_json=config_json,device = device)
            noises.append(noise)
            if params[2]:
                labels_g.append(1)
            else:
                labels_g.append(0)

            one_model = []
            for cls in range(2):
                a = noise[cls].mean(0)
                a = (a-a.min())/(a.max()-a.min())
                one_model.append(a)
            features_g.append(torch.cat(one_model,dim=1))  
            
        if df['model_architecture'][idx] == 'roberta-base':
            noise,stats = white_noise(model,768,config_json=config_json,device = device)
            noises.append(noise)
            if params[2]:
                labels_r.append(1)
            else:
                labels_r.append(0)

            one_model = []
            for cls in range(2):
                a = noise[cls].mean(0)
                a = (a-a.min())/(a.max()-a.min())
                one_model.append(a)
            features_r.append(torch.cat(one_model,dim=1))
            
        if df['model_architecture'][idx] == 'distilbert-base-cased':
            noise,stats = white_noise(model,768,sequence = False,config_json=config_json,device = device)
            noises.append(noise)
            if params[2]:
                labels_d.append(1)
            else:
                labels_d.append(0)

            one_model = []
            for cls in range(2):
                a = noise[cls].mean(0)
                a = (a-a.min())/(a.max()-a.min())
                one_model.append(a)
            features_d.append(torch.cat(one_model,dim=1))
        
        features_g_cat = np.array([])
        features_r_cat = np.array([])
        features_d_cat = np.array([])
        
        if len(features_g) > 0:
            features_g_cat = torch.cat(features_g,dim=0)
            
        if len(features_r) > 0:
            features_r_cat = torch.cat(features_r,dim=0)
            
        if len(features_d) > 0:
            features_d_cat = torch.cat(features_d,dim=0)
            

    return features_g_cat,np.array(labels_g),features_r_cat,np.array(labels_r),features_d_cat,np.array(labels_d)


def get_feature_vector(model, architecture, config_json):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if architecture == 'google/electra-small-discriminator':
        noise,stats = white_noise(model,256,config_json=config_json,device = device)
        one_model = []
        for cls in range(2):
            a = noise[cls].mean(0)
            a = (a-a.min())/(a.max()-a.min())
            one_model.append(a)
        
        result = torch.cat(one_model,dim=1)
        return result
    
    if architecture == 'roberta-base':
        noise,stats = white_noise(model,768,config_json=config_json,device = device)
        one_model = []
        for cls in range(2):
            a = noise[cls].mean(0)
            a = (a-a.min())/(a.max()-a.min())
            one_model.append(a)
        
        result = torch.cat(one_model,dim=1)
        return result
    
    if architecture == 'distilbert-base-cased':
        noise,stats = white_noise(model,768,sequence = False,config_json=config_json,device = device)
        one_model = []
        for cls in range(2):
            a = noise[cls].mean(0)
            a = (a-a.min())/(a.max()-a.min())
            one_model.append(a)
        
        result = torch.cat(one_model,dim=1)
        return result
    


