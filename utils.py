import os
import json
import pickle
import torch
import datasets

# The inferencing approach was adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def tokenize_for_qa(tokenizer, dataset):
    column_names = dataset.column_names
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(tokenizer.model_max_length, 384)
    
    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    
    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        
        pad_to_max_length = True
        doc_stride = 128
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created
        
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # offset_mapping = tokenized_examples.pop("offset_mapping")
        # offset_mapping = copy.deepcopy(tokenized_examples["offset_mapping"])
        
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        
        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            
            context_index = 1 if pad_on_right else 0
            
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
    
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                
                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
            
            # This is for the evaluation side of the processing
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        
        return tokenized_examples
    
    # Create train feature from dataset
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        keep_in_memory=True)
    
    if len(tokenized_dataset) == 0:
        print(
            'Dataset is empty, creating blank tokenized_dataset to ensure correct operation with pytorch data_loader formatting')
        # create blank dataset to allow the 'set_format' command below to generate the right columns
        data_dict = {'input_ids': [],
                     'attention_mask': [],
                     'token_type_ids': [],
                     'start_positions': [],
                     'end_positions': []}
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    return tokenized_dataset



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
    #examples_dirpath = os.path.join(model_dirpath, 'example_data')
    examples_dirpath = model_dirpath
    return arch, poisoned, model_dirpath, model_filepath, tokenizer_filepath, examples_dirpath


def get_all_examples(df,idx,model_filepath,examples_dirpath,train_path,models_path,rate=1.0):
    scratch_dirpath = './scratch'
    data_files_clean = []
    data_files_trojaned = []
    model_dirpath, _ = os.path.split(model_filepath)
    
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
        
    print('Source dataset name = "{}"'.format(config['source_dataset']))
    source =  config['source_dataset']
    '''pick = source + '.pickle'
    
    path = os.path.join('data',pick)
    print(path)
    if os.path.isfile(path):
        with open(path, 'rb') as handle:
            dataset_clean = pickle.load(handle)
            
        return dataset_clean'''
    
    if 'data_filepath' in config.keys():
        print('Source dataset filepath = "{}"'.format(config['data_filepath']))
        
    for i in range(len(df)):
        arch, poisoned, model_dirpath, model_filepath, tokenizer_filepath, examples_dirpath = read_model(df, i, train_path, models_path)
        model_dirpath, _ = os.path.split(model_filepath)
        
        with open(os.path.join(model_dirpath, 'config.json')) as json_file:
            config = json.load(json_file)
            
        if source == config['source_dataset']:
            fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
            fns.sort()
            if len(fns) == 2:
                data_files_trojaned.append(fns[1])
                
            data_files_clean.append(fns[0])
            
    size = int(len(data_files_clean) *rate)
    dataset_clean = datasets.load_dataset('json', data_files= data_files_clean[0:size], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))#extend data files to get whole data
    #with open(path, 'wb') as handle:
    #    pickle.dump(dataset_clean, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #dataset_trojaned = datasets.load_dataset('json', data_files= data_files_trojaned, field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))#extend data files to get whole data
    return   dataset_clean      

def tokenize_dataset(dataset,tokenizer,batch_size=20):
    tokenized_dataset = tokenize_for_qa(tokenizer, dataset)
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size)
    return dataloader , tokenized_dataset 


