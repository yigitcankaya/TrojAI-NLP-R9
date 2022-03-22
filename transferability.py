import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import os
import json
import pickle
import numpy as np
import copy
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_cpu(x):
    device = 'cpu'
    x = torch.as_tensor(x)
    x = x.to(device)
    return x.reshape(1,-1)

'''def get_poisoned_path(df,idx,train_path):
    if df['poisoned'][idx] == True:
        path =  os.path.join(train_path,'models')
        ind = os.path.join(path,df['model_name'][idx])
        return os.path.join(ind,'poisoned_example_data')
    return -1'''

def get_json_path(df,idx,train_path):
    path =  os.path.join(train_path,'models')
    ind = os.path.join(path,df['model_name'][idx])
    return os.path.join(ind,'config.json')

def find_trigger_class(dic,class_name):
    #print(dic)
    out =  dic['B-'+class_name]
    return out
   

def max_length(array):
    maximum = 0
    for arr in array:
        length = arr.shape[1]
        if length  > maximum:
            maximum = length
    return maximum

def pad(all_input_ids, all_attention_masks,all_labels):
    maximum = max_length(all_input_ids)
    for i in range(len(all_input_ids)):
        extend = maximum - all_input_ids[i].shape[1]
        all_input_ids[i] = torch.cat((all_input_ids[i].to('cpu'), torch.zeros(1,extend).long().to('cpu')), dim = 1)
        all_attention_masks[i] = torch.cat((all_attention_masks[i].to('cpu'), torch.zeros(1,extend).long().to('cpu')), dim = 1)
        all_labels[i] = torch.cat((all_labels[i].to('cpu'), torch.zeros(1,extend).long().to('cpu')), dim = 1)
        
def make_batch(arr):
    arr = np.array(arr)
    arr= torch.tensor(np.concatenate(arr))
    return arr


class TextDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels,target_masks = None):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.target_masks = target_masks
       
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        if self.target_masks != None:
            sample = self.input_ids[idx], self.attention_masks[idx], self.labels[idx],self.target_masks[idx]
        else:
            sample = self.input_ids[idx], self.attention_masks[idx], self.labels[idx]
        return sample
    
    
def tokenize_sentences(tokenizer_filepath, sentences):
    embedding_type = os.path.basename(tokenizer_filepath).split('-')[0]
    # Load the provided tokenizer
    tokenizer = torch.load(tokenizer_filepath)

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # identify the max sequence length for the given embedding
    if embedding_type == 'MobileBERT':
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    else:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]
        
    all_input_ids, all_attention_masks, all_labels, all_labels_mask = [], [], [], []

    for sentence in sentences:
        original_words, original_labels = sentence[0], sentence[1]
        input_ids, attention_mask, labels, labels_mask = manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)
        all_labels_mask.append(labels_mask)

    return all_input_ids, all_attention_masks, all_labels, tokenizer


def find_sources(input_ids, trigger_id,glob = False,target_class = -1,labels = None):
    trigger_id = torch.tensor(trigger_id).to('cpu')
    length = len(trigger_id)
    trigger_mask = torch.zeros(len(input_ids))
    for i in range(len(input_ids)- length) :#Triggers are always within the sentence not end
        if torch.equal(input_ids[i:i+length],trigger_id):
            if glob == False:
                trigger_mask[i:i+length+1] = torch.ones(length+1)#take the nearest element
            else:
                trigger_mask[i:i+length] = torch.ones(length)
    if glob:#take all possible source classes
        mask1= labels == target_class
        mask2 = labels == target_class + 1
        mask = torch.logical_or(mask1,mask2)
        trigger_mask = torch.logical_or(trigger_mask.to(device), mask.to(device))
        
    return trigger_mask.to(device)


def find_trigger_place(input_ids, old_id):
    old_id = torch.tensor(old_id).to(device)
    length = len(old_id)
    trigger_mask = torch.zeros(len(input_ids))
    for i in range(len(input_ids)- length) :
        if torch.equal(input_ids[i:i+length],old_id):#Triggers are always within the sentence not end
            trigger_mask[i:i+length] = torch.ones(length)#take the nearest element
    return trigger_mask


def acc_trigger(pred,labels,trigger_mask,device='cpu'):
    mask1 = labels != -100
    mask2 = labels != 0
    mask = torch.logical_and(mask1, mask2)
    mask = mask.to(device)
    mask = torch.logical_and(trigger_mask, mask)
    binary = torch.masked_select(pred,mask) == torch.masked_select(labels,mask) 
    correct = torch.sum(binary)
    wrong = len(binary) - correct
    return correct.item(), wrong.item()


def prep_labels(labels, trigger_mask,device='cpu'):
    lab = torch.clone(labels)
    mask1 = lab != -100
    mask2 = lab != 0
    mask = torch.logical_and(mask1, mask2)
    mask = mask.to(device)
    mask = torch.logical_and(trigger_mask.to(device), mask)
    #to be continued
    lab[torch.logical_not(mask)] = -100
    return lab
    
    
def perturb_trigger(model, inp_ids,labels,attention_mask,optimizer):
    model.train()
    optimizer.zero_grad()
    
    loss,out = model(inp_ids,attention_mask,labels)
    loss.backward()
    optimizer.step() 
    model.eval()
    
def acc_clean(pred,labels,attention_masks):
    mask1 = labels != -100
    mask2 = attention_masks != 0
    mask = torch.logical_and(mask1, mask2)
    binary = torch.masked_select(pred,mask) == torch.masked_select(labels,mask) 
    correct = torch.sum(binary)
    wrong = len(binary) - correct
    return correct.item(), wrong.item()
    
def inject_trigger(input_ids,attention_masks,labels,source_class,target_class,trigger_ids,is_glob,char_level = False):## We need extraif for non-global char (adjacent) 
    #if is global
    if is_glob:#put a random place?
        for idx in range(len(input_ids)):
            if  not(source_class in labels[idx][0,:]):
                 #change labels to get rid of confusion
                first_length = input_ids[idx].shape[1]
                for i in range(first_length):
                    if labels[idx][0,i] == target_class:
                        labels[idx][0,i] = -100
                    if labels[idx][0,i] == target_class + 1:
                        labels[idx][0,i] = -100
                continue
                
               
            #first make concat
            new_inp = torch.zeros(1,len(trigger_ids)).long()
            new_mask = torch.ones(1,len(trigger_ids)).long()
            new_label = torch.zeros(1,len(trigger_ids)).long()
            
            first_length = input_ids[idx].shape[1]
            add_length = new_inp.shape[1]
            
            for i in range(first_length):
                if labels[idx][0,i] == target_class:
                    labels[idx][0,i] = -100
                if labels[idx][0,i] == target_class + 1:
                    labels[idx][0,i] = -100
            
            #find a random place
            place_start = random.randint(0,first_length)
            
            #make concat
            input_ids[idx] = torch.cat((input_ids[idx],new_inp),dim = 1)
            attention_masks[idx] = torch.cat((attention_masks[idx],new_mask),dim = 1)
            labels[idx] = torch.cat((labels[idx],new_label),dim = 1)
            
            #make placement
            input_ids[idx][0,place_start+add_length:first_length+add_length]  = input_ids[idx][0,place_start:first_length].clone()
            input_ids[idx][0,place_start:place_start + add_length] = torch.tensor(trigger_ids) 
            
            labels[idx][0,place_start+add_length:first_length+add_length]  = labels[idx][0,place_start:first_length].clone()
            labels[idx][0,place_start:place_start + add_length] = torch.zeros(add_length) 
            
            #change labels
            for i in range(first_length + add_length):
                if labels[idx][0,i] == source_class:
                    labels[idx][0,i] = target_class
                if labels[idx][0,i] == source_class + 1:
                    labels[idx][0,i] = target_class + 1
                    
    else:
        for idx in range(len(input_ids)):
            if not(source_class in labels[idx][0,:]):
                continue
            
            start_places = []
            
            for i in range(input_ids[idx].shape[1]):
                if labels[idx][0,i] == source_class:
                    start_places.append(i)            
            for place_start in start_places:
                new_inp = torch.zeros(1,len(trigger_ids)).long()
                new_mask = torch.ones(1,len(trigger_ids)).long()
                new_label = torch.zeros(1,len(trigger_ids)).long()

                first_length = input_ids[idx].shape[1]
                add_length = new_inp.shape[1]

                #make concat
                input_ids[idx] = torch.cat((input_ids[idx],new_inp),dim = 1)
                attention_masks[idx] = torch.cat((attention_masks[idx],new_mask),dim = 1)
                labels[idx] = torch.cat((labels[idx],new_label),dim = 1)

                #make placement
                input_ids[idx][0,place_start+add_length:first_length+add_length]  = input_ids[idx][0,place_start:first_length].clone()
                input_ids[idx][0,place_start:place_start + add_length] = torch.tensor(trigger_ids) 

                labels[idx][0,place_start+add_length:first_length+add_length]  = labels[idx][0,place_start:first_length].clone()
                labels[idx][0,place_start:place_start + add_length] = torch.zeros(add_length)
                
            if char_level:
                for i in range(first_length + add_length):
                    if i in start_places:
                        labels[idx][0,i] = target_class#give adjacent char to a label
                        
                    if labels[idx][0,i] == source_class:
                        labels[idx][0,i] = -100
                    if labels[idx][0,i] == source_class + 1:
                        labels[idx][0,i] = -100
            else:
                for i in range(first_length + add_length):
                    if labels[idx][0,i] == source_class:
                        labels[idx][0,i] = target_class
                    if labels[idx][0,i] == source_class + 1:
                        labels[idx][0,i] = target_class + 1

    return  input_ids,attention_masks,labels     


def collect_all_samples_from_json(examples_json):
    #print(examples_json)
    f = open(examples_json)
    js = json.load(f)
    sentences = []
    for i in range(len(js['data'])):
        original_words = js['data'][i]['tokens']
        original_labels = js['data'][i]['ner_tags']
        sentences.append([original_words,original_labels])

    return sentences

def read_model(df, model_idx, main_path, models_path):
    model_name = df['model_name'][model_idx]
    model_dirpath = os.path.join(models_path, model_name)
    arch = df['model_architecture'][model_idx]
    if arch == 'roberta-base':
        tokenizer_filepath = os.path.join(main_path,'tokenizers/roberta-base.pt')
    if arch == 'google/electra-small-discriminator':
        tokenizer_filepath = os.path.join(main_path,'tokenizers/google-electra-small-discriminator.pt')
    if arch == 'distilbert-base-cased':
        tokenizer_filepath = os.path.join(main_path,'tokenizers/distilbert-base-cased.pt')
    poisoned = df['poisoned'][model_idx]
    model_filepath = os.path.join(model_dirpath, 'model.pt')
    examples_dirpath = os.path.join(model_dirpath, 'clean-example-data.json')
    poison_dirpath = os.path.join(model_dirpath, 'poisoned-example-data.json')
    return arch, poisoned, model_dirpath, model_filepath, tokenizer_filepath, examples_dirpath, examples_dirpath


def manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length):
    labels = []
    label_mask = []
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token
    tokens = []
    attention_mask = []
    
    # Add cls token
    tokens.append(cls_token)
    attention_mask.append(1)
    labels.append(-100)
    label_mask.append(0)
    
    for i, word in enumerate(original_words):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label = original_labels[i]
        
        # Variable to select which token to use for label.
        # All transformers for this round use bi-directional, so we use first token
        token_label_index = 0
        for m in range(len(token)):
            attention_mask.append(1)
            
            if m == token_label_index:
                labels.append(label)
                label_mask.append(1)
            else:
                labels.append(-100)
                label_mask.append(0)
        
    if len(tokens) > max_input_length - 1:
        tokens = tokens[0:(max_input_length-1)]
        attention_mask = attention_mask[0:(max_input_length-1)]
        labels = labels[0:(max_input_length-1)]
        label_mask = label_mask[0:(max_input_length-1)]
            
    # Add trailing sep token
    tokens.append(sep_token)
    attention_mask.append(1)
    labels.append(-100)
    label_mask.append(0)
    #print(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    #print(input_ids)
    
    return input_ids, attention_mask, labels, label_mask

def get_sentences(models_path,df,tokenizer_filepath, dataset_name, sample_ratio):
    
    sentences = []
    #get all sentences having the same dataset
    for i in range(int(len(df) * sample_ratio)):
        if (df['task_type_level'][i] == 1) and (df['source_dataset'][i] == dataset_name):
            model_name = df['model_name'][i]
            model_dirpath = os.path.join(models_path, model_name)
            clean_dirpath = os.path.join(model_dirpath, 'clean-example-data.json')
            sentences_partial = collect_all_samples_from_json(clean_dirpath)
            sentences += sentences_partial
            
    all_input_ids, all_attention_masks, all_labels, _ = tokenize_sentences(tokenizer_filepath, sentences)

    print(f'Querying the model with {len(all_input_ids)} sentences...')

    return all_input_ids, all_attention_masks, all_labels

class Loss_Wrapper(torch.nn.Module):
    def __init__(self,model):
        super(Loss_Wrapper, self).__init__()
        self.model = model
        
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        emissions = self.model(input_ids, attention_mask=attention_mask).logits

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = emissions.view(-1, emissions.shape[2])
                active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(emissions.view(-1,  emissions.shape[2]), labels.view(-1))

        return loss, emissions
    
    
def model_differences(df,train_path, models_path,save_location="model_data/"):
    all_models = []
    count = 0
    dics = {}
    for idx in range(len(df)):
        #if idx == 4 or idx == 13 or idx == 57 or idx == 79 or idx ==104:
        #    continue
        #print(df['model_architecture'][idx])
        #if idx != 105:
        #    continue
        sys.stdout.flush()
        if df['poisoned'][idx] == False  or df['task_type_level'][idx] != 1:
            continue
        
        key = str(df['model_architecture'][idx]) + str(df['source_dataset'][idx]).replace(':', '_')
        print(key)
        if key in dics.keys():
            continue
        else:
            dics[key] = 1
        print(idx)
        _, label, model_dirpath, model_filepath, tokenizer_filepath, examples_dirpath, poisoned_dirpath = read_model(df, idx, train_path, models_path)
        model = torch.load(model_filepath, map_location=torch.device(device))
        model = Loss_Wrapper(model)
        json_path = get_json_path(df,idx,train_path)
        sentences = collect_all_samples_from_json(poisoned_dirpath)
        js = open(json_path)
        js_dict = json.load(js)
        print(idx)
        if 'trigger_text' in js_dict['trigger']['trigger_executor']:
            trigger = js_dict['trigger']['trigger_executor']['trigger_text']
        else:
            trigger_list = js_dict['trigger']['trigger_executor']['trigger_text_list']
            trigger = ''
            for string in trigger_list:
                if string != trigger_list[-1]:
                    trigger += string + ' '
                else:
                    trigger += string
        
        char_level = len(trigger) == 1
        
        class_mapping = js_dict['trigger']['trigger_executor']["label_to_id_map"]
        class_name = js_dict['trigger']['trigger_executor']["target_class_label"]
        source_class_name = js_dict['trigger']['trigger_executor']["source_class_label"]
        target_class = find_trigger_class(class_mapping,class_name)
        source_class = find_trigger_class(class_mapping,source_class_name)
        print(source_class)
        print(target_class)

        is_global = js_dict['trigger']['trigger_executor']["global_trigger"]
        
        _, _, _, tokenizer = tokenize_sentences(tokenizer_filepath, sentences)
        
        tokens = tokenizer.tokenize(trigger)
        trigger_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        glob = is_global
        print(glob)
        amount = 1000

        #dataset_name = identify_dataset(examples_dirpath) 
        dataset_name = df['source_dataset'][idx]
        all_input_ids, all_attention_masks, all_labels = get_sentences(models_path,df,tokenizer_filepath, dataset_name, 1.0)
        all_input_ids = all_input_ids[0:amount]
        all_attention_masks = all_attention_masks[0:amount]
        all_labels = all_labels[0:amount]
        
        all_input_ids = [to_cpu(x) for x in all_input_ids]
        all_attention_masks = [to_cpu(x) for x in all_attention_masks]
        all_labels = [to_cpu(x) for x in all_labels]
        
        input_ids,attention_masks,labels   = inject_trigger(copy.deepcopy(all_input_ids),copy.deepcopy(all_attention_masks),copy.deepcopy(all_labels),source_class,target_class,trigger_ids,glob,char_level)
        pad(input_ids, attention_masks,labels)
       
        
        input_ids = make_batch(input_ids)
        print(input_ids.shape)
        attention_masks = make_batch(attention_masks)
        labels = make_batch(labels)
       
        
        
        pad(all_input_ids, all_attention_masks,all_labels)
        clean_input_ids = make_batch(all_input_ids)
        clean_attention_masks = make_batch(all_attention_masks)
        clean_labels = make_batch(all_labels)
      
    
        print(trigger_ids)
        #return input_ids,all_input_ids
        target_masks= []
        for inp_id in range(input_ids.shape[0]):
            target_mask = find_sources(input_ids[inp_id], trigger_ids, glob =  glob,target_class = target_class, labels = labels[inp_id])
            target_masks.append(target_mask.reshape(1,-1))
        target_mask = torch.cat(target_masks,dim = 0).to(device)
        #print(target_mask)

        lab = prep_labels(labels, target_mask)
        
        batch_size = 50
        
        #Create data generators
        pert_trigger = TextDataset(input_ids, attention_masks,lab)
        pert_trigger_generator = torch.utils.data.DataLoader(pert_trigger,batch_size = batch_size)
        
        test_trigger = TextDataset(input_ids, attention_masks,labels,target_mask)
        test_trigger_generator = torch.utils.data.DataLoader(test_trigger,batch_size = batch_size)
                
        dic = {}

        model_pert = copy.deepcopy(model)
        
        dic['prev'] = model.to('cpu')
        
        optimizer = torch.optim.Adam(model_pert.parameters(), lr=0.00001)
        for group in optimizer.param_groups:#make gradient ascent
            group['lr'] *= -1 
        
        for epoch in range(50):
            for (inp, at,label) in  pert_trigger_generator:  
                inp,at,label = inp.to(device),at.to(device),label.to(device)
                perturb_trigger(model_pert, inp,label,at,optimizer)
                
                correct_trigger = 0
                wrong_trigger = 0
                for (inp, at,label,tar) in  test_trigger_generator: 
                    inp,at,label,tar = inp.to(device),at.to(device),label.to(device),tar.to(device)
                    out = model_pert(inp,at)[1]
                    pred = out.max(2)[1]
                    #print(tar)
                    #return inp,at,label,pred
                    
                    correct,wrong = acc_trigger(pred,label,tar,device=device)
                    print(correct,wrong)
                    

                    correct_trigger += correct
                    wrong_trigger += wrong
                print(correct_trigger/(correct_trigger + wrong_trigger))
                    
                if correct_trigger/(correct_trigger + wrong_trigger) < 0.05:
                    print(correct_trigger/(correct_trigger + wrong_trigger))
                    break
            
            if correct_trigger/(correct_trigger + wrong_trigger) < 0.1:
                break
                
        dic['after'] = model_pert.to('cpu')
        
        key_save = str(df['model_architecture_level'][idx]) + str(df['source_dataset'][idx]).replace(':','_')
       
        loc = os.path.join(save_location,  key_save + "_troj.pickle")
        
        with open(loc, 'wb') as handle:
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def mix_models(left_model, right_model, l):
    new_copy = copy.deepcopy(left_model)
    for (n1, w1), (n2, w2), (n3, w3) in zip(left_model.named_parameters(), right_model.named_parameters(), new_copy.named_parameters()):
        if 'weight' in n1:
            w3.data = l*w1 + (1 - l)*w2
    return new_copy

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_model_diffs(m1, m2):
    n_params = get_n_params(m1)
    diffs = np.zeros( n_params)
    start = 0
    for (param1,param2) in zip(m1.parameters(), m2.parameters()):
        param1 = param1.flatten().detach().cpu().numpy()
        param2 = param2.flatten().detach().cpu().numpy()
        dif = param1 - param2
        diffs[start:start+len(dif)] = dif
        start += len(dif)
    return diffs


def seperate_keys(df,train_path, models_path):
    all_keys = {}
    for idx in range(len(df)):
        if df['task_type_level'][idx] != 1:
            continue
        key = str(df['model_architecture'][idx]) + str(df['source_dataset'][idx])
        _, label, model_dirpath, model_filepath, tokenizer_filepath, examples_dirpath, poisoned_dirpath = read_model(df, idx, train_path, models_path)
        
        
        if not (key in all_keys):
            all_keys[key] = {'trojaneds':[], 'cleans':[], 'mixup_id': 0, 'troj_id': 0}
           
        if label == True:
            all_keys[key]['trojaneds'].append(idx)
        else:
            all_keys[key]['cleans'].append(idx)
            
    return all_keys
   
    
def get_dataset(df,idx,tokenizer_filepath,examples_dirpath,train_path,models_path,amount = 100, batch_size = 50):
    _, label, model_dirpath, model_filepath, tokenizer_filepath, examples_dirpath, poisoned_dirpath = read_model(df, idx, train_path, models_path)
    json_path = get_json_path(df,idx,train_path)
    sentences = collect_all_samples_from_json(poisoned_dirpath)
    js = open(json_path)
    js_dict = json.load(js)
    print(idx)
    print(js_dict)
    if 'trigger_text' in js_dict['trigger']['trigger_executor']:
        trigger = js_dict['trigger']['trigger_executor']['trigger_text']
    else:
        trigger_list = js_dict['trigger']['trigger_executor']['trigger_text_list']
        trigger = ''
        for string in trigger_list:
            if string != trigger_list[-1]:
                trigger += string + ' '
            else:
                trigger += string

    char_level = len(trigger) == 1

    class_mapping = js_dict['trigger']['trigger_executor']["label_to_id_map"]
    class_name = js_dict['trigger']['trigger_executor']["target_class_label"]
    source_class_name = js_dict['trigger']['trigger_executor']["source_class_label"]
    target_class = find_trigger_class(class_mapping,class_name)
    source_class = find_trigger_class(class_mapping,source_class_name)
    print(source_class)
    print(target_class)

    is_global = js_dict['trigger']['trigger_executor']["global_trigger"]

    _, _, _, tokenizer = tokenize_sentences(tokenizer_filepath, sentences)

    tokens = tokenizer.tokenize(trigger)
    trigger_ids = tokenizer.convert_tokens_to_ids(tokens)

    glob = is_global
    print(glob)

    #dataset_name = identify_dataset(examples_dirpath) 
    dataset_name = df['source_dataset'][idx]
    all_input_ids, all_attention_masks, all_labels = get_sentences(models_path,df,tokenizer_filepath, dataset_name, 1.0)
    all_input_ids = all_input_ids[0:amount]
    all_attention_masks = all_attention_masks[0:amount]
    all_labels = all_labels[0:amount]

    all_input_ids = [to_cpu(x) for x in all_input_ids]
    all_attention_masks = [to_cpu(x) for x in all_attention_masks]
    all_labels = [to_cpu(x) for x in all_labels]

    input_ids,attention_masks,labels   = inject_trigger(copy.deepcopy(all_input_ids),copy.deepcopy(all_attention_masks),copy.deepcopy(all_labels),source_class,target_class,trigger_ids,glob,char_level)
    pad(input_ids, attention_masks,labels)


    input_ids = make_batch(input_ids)
    print(input_ids.shape)
    attention_masks = make_batch(attention_masks)
    labels = make_batch(labels)



    pad(all_input_ids, all_attention_masks,all_labels)
    clean_input_ids = make_batch(all_input_ids)
    clean_attention_masks = make_batch(all_attention_masks)
    clean_labels = make_batch(all_labels)


    print(trigger_ids)
    #return input_ids,all_input_ids
    target_masks= []
    for inp_id in range(input_ids.shape[0]):
        target_mask = find_sources(input_ids[inp_id], trigger_ids, glob =  glob,target_class = target_class, labels = labels[inp_id])
        target_masks.append(target_mask.reshape(1,-1))
    target_mask = torch.cat(target_masks,dim = 0).to(device)
    #print(target_mask)

    lab = prep_labels(labels, target_mask)

       

    #Create data generators

    test_trigger = TextDataset(input_ids, attention_masks,labels,target_mask)
    test_trigger_generator = torch.utils.data.DataLoader(test_trigger,batch_size = batch_size)
    test_clean = TextDataset(clean_input_ids, clean_attention_masks,clean_labels)
    test_clean_generator = torch.utils.data.DataLoader(test_clean,batch_size = batch_size)
    
    return test_trigger_generator, test_clean_generator


def poison_by_weights(m_prev, m_after, model,scale = 1, th  = 0):
    for (param_prev,param_after, param) in zip(m_prev.parameters(),  m_after.parameters(), model.parameters()):
        param_dif = param_prev - param_after
        mask = torch.abs(param_dif) > th
        param.data =  param.data + param_dif * mask * scale
        
def get_model_diffs(m1, m2):
    n_params = get_n_params(m1)
    diffs = np.zeros( n_params)
    start = 0
    for (param1,param2) in zip(m1.parameters(), m2.parameters()):
        param1 = param1.flatten().detach().cpu().numpy()
        param2 = param2.flatten().detach().cpu().numpy()
        dif = param1 - param2
        diffs[start:start+len(dif)] = dif
        start += len(dif)
    return diffs

def get_feature_set(df,train_path, models_path, all_keys, config_json={}, saved_models_path = None):

    th = config_json.get('transferability_sparsity', 1.0) 
    
    all_models = {}
    for key in  all_keys.keys():
        all_models[key] = {'labels': [], 'trojan_curves': [], 'clean_curves': []}
        
        mixup_id = all_keys[key]['mixup_id']
        troj_id = all_keys[key]['troj_id']
        
        m_idx = all_keys[key]['cleans'][mixup_id]
        t_idx = all_keys[key]['trojaneds'][troj_id]
        
        #load trojaned model
        #pos_ind = find_trojaned_order(t_idx, df)
        key_save = str(df['model_architecture_level'][m_idx]) + str(df['source_dataset'][m_idx]).replace(':','_')
        
        loc = os.path.join(saved_models_path , key_save + "_troj.pickle")#troj
        with open(loc, 'rb') as handle:
            troj_model = pickle.load(handle)
         
        
        #load mixup model
        _, label, model_dirpath, model_filepath, tokenizer_filepath, examples_dirpath, _ = read_model(df, m_idx, train_path, models_path)
        
        mixup_model = torch.load(model_filepath, map_location=torch.device(device))
        
        
        
        mod_pth = key_save + "_mixup.pt"
        model_filepath = os.path.join(saved_models_path,mod_pth)
        
        torch.save(mixup_model,model_filepath)
        
        mixup_model = Loss_Wrapper(mixup_model)
        n_param = get_n_params(mixup_model)
        
        test_trigger_generator, test_clean_generator = get_dataset(df,t_idx,tokenizer_filepath,examples_dirpath,train_path,models_path,amount = 250, batch_size = 50)
        print('Finished Data')
        sys.stdout.flush()
        
        trigger_dataset_name = key_save + "_trigger_dataset.pt"
        test_trigger_generator_path = os.path.join(saved_models_path,trigger_dataset_name)
        
        #with open(test_trigger_generator_path, 'wb') as handle:
        #    pickle.dump(test_trigger_generator, handle, protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(test_trigger_generator,test_trigger_generator_path)
        
        
        for c_idx in all_keys[key]['cleans']:
            print(c_idx)
            sys.stdout.flush()
            #if c_idx == m_idx:
            #    continue
            _, label, model_dirpath, model_filepath, tokenizer_filepath, examples_dirpath, _ = read_model(df, c_idx, train_path, models_path)
            c_model = torch.load(model_filepath, map_location=torch.device(device))
            c_model = Loss_Wrapper(c_model)
            
            all_models[key]['labels'].append(0)
            
            mixup_rates = torch.arange(0,6) * 0.2
            for rate in mixup_rates:
                model_test = mix_models(mixup_model,c_model, rate)
                clean_accs = []
                trojans_accs = []
                for scale in range(10):#10
                    model_pert = copy.deepcopy(model_test)
                    diffs = get_model_diffs(troj_model['prev'], troj_model['after'])
                    s,_ = torch.sort(torch.abs(torch.tensor(diffs)), descending = True)
                    poison_by_weights(troj_model['prev'].to(device), troj_model['after'].to(device), model_pert.to(device),scale = scale,th =s[int(n_param * th)-1])
                    correct_trigger = 0
                    wrong_trigger = 0
                    correct_clean = 0
                    wrong_clean = 0
                    for (inp, at,label,tar) in  test_trigger_generator: 
                        inp,at,label,tar = inp.to(device),at.to(device),label.to(device),tar.to(device)
                        out = model_pert(inp,at)[1]
                        pred = out.max(2)[1]
                        correct,wrong = acc_trigger(pred,label,tar,device=device)
                        correct_trigger += correct
                        wrong_trigger += wrong
                        
                    trojans_accs.append(correct_trigger/(correct_trigger + wrong_trigger))
                    '''for (inp, at,label) in  test_clean_generator: 
                        inp,at,label = inp.to(device),at.to(device),label.to(device)
                        out = model_pert(inp.to(device), at.to(device))[1]
                        pred = out.max(2)[1]
                        correct,wrong = acc_clean(pred,label,at)
                        correct_clean += correct
                        wrong_clean += wrong

                
                    clean_accs.append(correct_clean/(correct_clean + wrong_clean))'''
            
                #all_models[key]['clean_curves'].append(clean_accs)
                all_models[key]['trojan_curves'].append(trojans_accs)
                
        for tr_idx in all_keys[key]['trojaneds']:
            print(tr_idx)
            sys.stdout.flush()
            #if tr_idx == t_idx:
            #    continue
            _, label, model_dirpath, model_filepath, tokenizer_filepath, examples_dirpath, _ = read_model(df, tr_idx, train_path, models_path)
            t_model = torch.load(model_filepath, map_location=torch.device(device))
            
            all_models[key]['labels'].append(1)
            
            mixup_rates = torch.arange(0,6) * 0.2
            for rate in mixup_rates:
                model_test = mix_models(mixup_model,t_model, rate)
                clean_accs = []
                trojans_accs = []
                for scale in range(10):#10
                    model_pert = copy.deepcopy(model_test)
                    diffs = get_model_diffs(troj_model['prev'], troj_model['after'])
                    s,_ = torch.sort(torch.abs(torch.tensor(diffs)), descending = True)
                    poison_by_weights(troj_model['prev'].to(device), troj_model['after'].to(device), model_pert.to(device),scale = scale,th =s[int(n_param * th)-1])
                    correct_trigger = 0
                    wrong_trigger = 0
                    correct_clean = 0
                    wrong_clean = 0
                    for (inp, at,label,tar) in  test_trigger_generator: 
                        inp,at,label,tar = inp.to(device),at.to(device),label.to(device),tar.to(device)
                        out = model_pert(inp,at)[1]
                        pred = out.max(2)[1]
                        correct,wrong = acc_trigger(pred,label,tar,device=device)
                        correct_trigger += correct
                        wrong_trigger += wrong
                        
                   
                    trojans_accs.append(correct_trigger/(correct_trigger + wrong_trigger))
                    '''for (inp, at,label) in  test_clean_generator: 
                        inp,at,label = inp.to(device),at.to(device),label.to(device)
                        out = model_pert(inp.to(device), at.to(device))[1]
                        pred = out.max(2)[1]
                        correct,wrong = acc_clean(pred,label,at)
                        correct_clean += correct
                        wrong_clean += wrong

               
                    clean_accs.append(correct_clean/(correct_clean + wrong_clean))'''
            
                #all_models[key]['clean_curves'].append(clean_accs)
                all_models[key]['trojan_curves'].append(trojans_accs)
                
        
    return all_models





def get_th_trojan(curve):
    for i,value in enumerate(curve):
        if value > 0.9:
            return i
    return torch.argmax(torch.tensor(curve))


def get_th_clean(curve):
    for i,value in enumerate(curve):
        if value < 0.9:
            return i
    return torch.argmax(torch.tensor(curve))

def extract_feature_mixup_trojan(trojan_curves):
    features = []
    for (i,curve) in enumerate(trojan_curves):
        if i%6 == 0:
            features.append([])
        th = get_th_trojan(curve)  
        features[-1].append(th)
    return features


def extract_feature_mixup_clean(trojan_curves):
    features = []
    for (i,curve) in enumerate(trojan_curves):
        if i%6 == 0:
            features.append([])
        th = get_th_clean(curve)  
        features[-1].append(th)
    return features

def normalize(features):
    for i in range(len(features)):
        arr = np.array(features[i])
        arr = arr / (np.max(arr)) 
        features[i] = arr
        
def extract_all_features(all_models):
    features_trojan1 = extract_feature_mixup_trojan(all_models['distilbert-base-casedner:conll2003']['trojan_curves'])
    features_trojan2 = extract_feature_mixup_trojan(all_models['roberta-basener:conll2003']['trojan_curves'])
    features_trojan3 = extract_feature_mixup_trojan(all_models['google/electra-small-discriminatorner:conll2003']['trojan_curves'])
    normalize(features_trojan1)
    normalize(features_trojan2)
    normalize(features_trojan3)
    labels1 = all_models['distilbert-base-casedner:conll2003']['labels']
    labels2 = all_models['roberta-basener:conll2003']['labels']
    labels3 = all_models['google/electra-small-discriminatorner:conll2003']['labels']
    labels = labels1 + labels2 + labels3
    features = features_trojan1 + features_trojan2 + features_trojan3
    return features, labels


def get_feature_vector(model,troj_model,mixup_model,test_trigger_generator,config_json):
    mixup_model = Loss_Wrapper(mixup_model)
    n_param = get_n_params(mixup_model)
    model = Loss_Wrapper(model)
    mixup_rates = torch.arange(0,6) * 0.2
    result = []
    th = config_json.get('transferability_sparsity', 1.0) 
    for rate in mixup_rates:
        model_test = mix_models(mixup_model,model, rate)
        trojans_accs = []
        for scale in range(10):
            model_pert = copy.deepcopy(model_test)
            diffs = get_model_diffs(troj_model['prev'], troj_model['after'])
            s,_ = torch.sort(torch.abs(torch.tensor(diffs)), descending = True)
            poison_by_weights(troj_model['prev'].to(device), troj_model['after'].to(device), model_pert.to(device),scale = scale,th =s[int(n_param * th)-1])
            correct_trigger = 0
            wrong_trigger = 0
            correct_clean = 0
            wrong_clean = 0
            for (inp, at,label,tar) in  test_trigger_generator: 
                inp,at,label,tar = inp.to(device),at.to(device),label.to(device),tar.to(device)
                out = model_pert(inp,at)[1]
                pred = out.max(2)[1]
                correct,wrong = acc_trigger(pred,label,tar,device=device)
                correct_trigger += correct
                wrong_trigger += wrong

            trojans_accs.append(correct_trigger/(correct_trigger + wrong_trigger))
            print(trojans_accs)
        result.append(trojans_accs)
        
    output = []
    for i in range(len(result)):
        th = get_th_trojan(result[i])
        output.append(th)
    output = np.array(output)
    output = output / (np.max(output))
    return output
        
    
