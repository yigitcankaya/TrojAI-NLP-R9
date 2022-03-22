import torch.optim as optim
import os
import sys
import torch
import utils_qa
import utils as ut
import json
import copy
import pickle
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mix_models(left_model, right_model, l):
    new_copy = copy.deepcopy(left_model)
    for (n1, w1), (n2, w2), (n3, w3) in zip(left_model.named_parameters(), right_model.named_parameters(), new_copy.named_parameters()):
        if 'weight' in n1:
            w3.data = l*w1 + (1 - l)*w2

    return new_copy

def find_same_clean_models(df, idx,train_path,models_path):
    arch, poisoned, model_dirpath, model_filepath, tokenizer_filepath, examples_dirpath = ut.read_model(df, idx, train_path, models_path)    
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
        
    dataset = config['source_dataset']
    
    models = []
    for i in range(len(df)):
        if i == idx:
            continue
        arch_new, poisoned, model_dirpath, model_filepath, tokenizer_filepath, examples_dirpath = ut.read_model(df, i, train_path, models_path)    
        model_dirpath, _ = os.path.split(model_filepath)
        with open(os.path.join(model_dirpath, 'config.json')) as json_file:
            config = json.load(json_file)
            
        if poisoned == False and arch_new == arch and dataset == config['source_dataset']:
            models.append(i)
    return models


class SharpnessApproximation():
    def __init__(self, model):#use sgd to find gradients
        self.model = model
        self.parameters = model.parameters()
        self.optimizer = optim.SGD(self.parameters, lr=0.01)
        self.param_groups = self.optimizer.param_groups
        
    def forward_pass(self,tensor_dict):
        self.optimizer.zero_grad()
        #criterion = nn.CrossEntropyLoss()
        input_ids = tensor_dict['input_ids'].to(device)
        attention_mask = tensor_dict['attention_mask'].to(device)
        token_type_ids = tensor_dict['token_type_ids'].to(device)
        start_positions = tensor_dict['start_positions'].to(device)
        end_positions = tensor_dict['end_positions'].to(device)
        if 'distilbert' in self.model.name_or_path or 'bart' in self.model.name_or_path:
            loss = self.model(input_ids,
                                attention_mask=attention_mask,
                                start_positions=start_positions,
                                end_positions=end_positions)['loss']
        else:
            loss = self.model(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                start_positions=start_positions,
                                end_positions=end_positions)['loss']
        
        loss.backward()
        return loss
        
    @torch.no_grad()
    def add_noise(self,rho):
        #find proper noise bounded by rho
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = rho / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                #if len(p.grad.shape) == 1: continue
                
                #print(p.grad.shape)
                #print(len(p.grad.shape))
                e_w = (torch.pow(p, 2) if False else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.optimizer.state[p]["e_w"] = e_w
            
    @torch.no_grad()
    def remove_noise(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                #if len(p.grad.shape) == 1: continue
                p.sub_(self.optimizer.state[p]["e_w"])
            
        
        
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if False else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
def get_all_approx_mixup(df, train_path, models_path, config_json={}, save_path = None):
    all_models = []
    labels = []
    for idx in range(len(df)):
        print(idx)
        if df['task_type_level'][idx] != 2:
            continue
        arch, poisoned, model_dirpath, model_filepath, tokenizer_filepath, examples_dirpath = ut.read_model(df, idx, train_path, models_path)
        dataset_clean  = ut.get_all_examples(df,idx,model_filepath,examples_dirpath,train_path,models_path,rate=1.0)
        tokenizer = torch.load(tokenizer_filepath)
        dataloader,tokenized_dataset = ut.tokenize_dataset(dataset_clean,tokenizer,batch_size=20)
        
        model = torch.load(model_filepath, map_location=torch.device(device))
        
        tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions'])
        tensor_dict = next(iter(dataloader))
        
        key = str(df['model_architecture_level'][idx]) + str(df['source_dataset'][idx]).replace(':', '_')
        
        dict_path = key + "_tensor_dict.pickle"
        
        tensor_dict_path = os.path.join(save_path,dict_path)
        with open(tensor_dict_path, 'wb') as handle:
            pickle.dump(tensor_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        model_sharp = SharpnessApproximation(model)

        num_steps = config_json.get('loss_surface_num_steps', -1)

        if num_steps == -1:
            lams = [0.0,0.05,0.1,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1.0] 
        else:
            lams = np.linspace(0, 1, num=num_steps)


        loss_list = []
        loss_list_clean = []
        for lam in lams:
            #print(lam)
            model_sharp.forward_pass(tensor_dict)
            model_sharp.add_noise(lam)
            loss = model_sharp.forward_pass(tensor_dict)
            loss_list.append(loss.detach().item())
            model_sharp.remove_noise()
        
        labels.append(poisoned)
        all_models.append(((lams,loss_list), (lams,loss_list_clean)))
    return all_models,labels


def feature_from_loss_surface(model, tensor_dict, config_json={}, device = 'cuda'):
    model_sharp = SharpnessApproximation(model)

    num_steps = config_json.get('loss_surface_num_steps', -1)

    if num_steps == -1:
        lams = [0.0,0.05,0.1,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1.0] 
    else:
        lams = np.linspace(0, 1, num=num_steps)

    loss_list = []
    for lam in lams:
        model_sharp.forward_pass(tensor_dict)
        model_sharp.add_noise(lam)
        loss = model_sharp.forward_pass(tensor_dict)
        loss_list.append(loss.detach().item())
        model_sharp.remove_noise()
        
    feature = np.array(loss_list) #- np.array(loss_list_clean)
    return feature