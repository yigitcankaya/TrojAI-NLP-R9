import os

import numpy as np
import torch
import json
import jsonschema
import sys

import warnings

import white_noise as wn
import transferability as tr
import loss_surface as ls
import pandas
from sklearn.neural_network import MLPClassifier
import pickle

warnings.filterwarnings("ignore")


def calibrate_prob(prob):
    if prob < 0.15: # prevent extremely low probs
        return 0.15
    
    elif prob > 0.85: # prevent extremely high probs
        return 0.85

    else:
        return prob

def trojan_detector(model_filepath, tokenizer_filepath, result_filepath, 
                    parameters_dirpath, config_json):

    print('model_filepath = {}'.format(model_filepath))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('result_filepath = {}'.format(result_filepath))

    print('Using parameters_dirpath = {}'.format(parameters_dirpath))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Device: {device}')
    
    # load the config file to retrieve parameters
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)

    source_dataset = config['source_dataset']

    task_type = source_dataset.split(':')[0]


    arc_idx = 0
    architecture = config['model_architecture']
    if 'roberta' in architecture:
        arc_idx = 0
    elif 'electra' in architecture:
        arc_idx = 1
    else:
        arc_idx = 2

    print('Source dataset name = "{}" - task type: {} - arc : {} - arc idx : {} '.format(config['source_dataset'], task_type, architecture, arc_idx))

    if task_type == 'sc':
        print('Sentiment Classification')
        # load the classification model and move it to the GPU
        model = torch.load(model_filepath, map_location=torch.device(device))
        
        if architecture == 'google/electra-small-discriminator':
            clf_g_path = os.path.join(parameters_dirpath,"clf_g.pickle")
            with open(clf_g_path, 'rb') as handle:
                clf= pickle.load(handle)
                
        if architecture == 'roberta-base':
            clf_r_path = os.path.join(parameters_dirpath,"clf_r.pickle")
            with open(clf_r_path, 'rb') as handle:
                clf= pickle.load(handle)
                
        if architecture == 'distilbert-base-cased':
            clf_d_path = os.path.join(parameters_dirpath,"clf_d.pickle")
            with open(clf_d_path, 'rb') as handle:
                clf = pickle.load(handle)
                
        feature = wn.get_feature_vector(model,architecture,config_json)
        prob = clf.predict_proba(feature.reshape(1,-1))
        print(prob)
        trojan_probability  = calibrate_prob(prob[0][1])

        print('Trojan Probability: {}'.format(trojan_probability))
        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(trojan_probability))
               
    elif task_type == 'ner':
        print('Named Entity Recognition')
        model = torch.load(model_filepath, map_location=torch.device(device))
        key_save = str(arc_idx) + str(config['source_dataset']).replace(':', '_')
        
        loc = os.path.join(parameters_dirpath,  key_save + "_troj.pickle")
        with open(loc, 'rb') as handle:
            troj_model = pickle.load(handle)
                
        mod_pth = key_save + "_mixup.pt"
        model_filepath = os.path.join(parameters_dirpath, mod_pth)
        mixup_model = torch.load(model_filepath, map_location=torch.device(device))
        
        trigger_dataset_name = key_save + "_trigger_dataset.pt"
        test_trigger_generator_path = os.path.join(parameters_dirpath,trigger_dataset_name)
        
        test_trigger_generator = torch.load(test_trigger_generator_path, pickle_module=pickle, map_location=torch.device(device))
        
        feature = tr.get_feature_vector(model,troj_model,mixup_model,test_trigger_generator,config_json)
        print(feature)
        
        clf_path = os.path.join(parameters_dirpath,"clf_ner.pickle")
        with open(clf_path, 'rb') as handle:
            clf = pickle.load(handle)
            
        prob = clf.predict_proba(feature.reshape(1,-1))
        print(prob)
        trojan_probability  = calibrate_prob(prob[0][1]) 

        print('Trojan Probability: {}'.format(trojan_probability))
        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(trojan_probability))

    elif task_type == 'qa':
        print('Question Answering')
        model = torch.load(model_filepath, map_location=torch.device(device))
        
        key = str(arc_idx) + str(config['source_dataset']).replace(':', '_')
        dict_path = key + "_tensor_dict.pickle"
        tensor_dict_path = os.path.join(parameters_dirpath,dict_path)
        with open(tensor_dict_path , 'rb') as handle:
                tensor_dict = pickle.load(handle)
        
        #load clf
        detector_path = os.path.join(parameters_dirpath,"detector.pickle")
        with open(detector_path, 'rb') as handle:
            clf = pickle.load(handle)
        
        feature = ls.feature_from_loss_surface(model,tensor_dict,config_json,device = device)
        prob = clf.predict_proba(feature.reshape(1,-1))
        print(prob)
        trojan_probability  = calibrate_prob(prob[0][1])

        print('Trojan Probability: {}'.format(trojan_probability))
        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(trojan_probability))
        
def extract_feature_loss(all_models,df):
    labels = []
    features = []
    count = 0
    for idx in range(len(df)):
        if df['task_type_level'][idx] != 2:
            continue
        if df['poisoned'][idx] == True:
            labels.append(1)
        else:
            labels.append(0)
        
        fet = np.array(all_models[count][0][1]) #- np.array(all_models[idx][1][1])
        features.append(fet.reshape(1,-1))
        count = count +  1
    return np.concatenate(features),labels


def configure(output_parameters_dirpath, configure_models_dirpath, config_json):

    print('Configuring detector parameters with models from ' + configure_models_dirpath)

    os.makedirs(output_parameters_dirpath, exist_ok=True)

    print('Writing configured parameter data to ' + output_parameters_dirpath)
    
    models_path = os.path.join(configure_models_dirpath, 'models')
    df = pandas.read_csv(os.path.join(configure_models_dirpath, 'METADATA.csv'))
    
    #Sentiment Classification Task
    
    # iter_ = config_json.get('white_noise_iters', 10)
    # all_size = config_json.get('white_noise_size', 10000)

    features_g,labels_g,features_r,labels_r,features_d,labels_d  = wn.extract_features_by_embedding(df,configure_models_dirpath,models_path,config_json)
    
    #then save these models
    clf_g = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(2048,512,64), random_state=1).fit(features_g, labels_g)
    clf_r = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(2048,512,64), random_state=1).fit(features_r, labels_r)
    clf_d = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(2048,512,64), random_state=1).fit(features_d, labels_d)
    
    clf_g_path = os.path.join(output_parameters_dirpath,"clf_g.pickle")
    clf_r_path = os.path.join(output_parameters_dirpath,"clf_r.pickle")
    clf_d_path = os.path.join(output_parameters_dirpath,"clf_d.pickle")
    
    with open(clf_g_path, 'wb') as handle:
        pickle.dump(clf_g, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    with open(clf_r_path, 'wb') as handle:
        pickle.dump(clf_r, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    with open(clf_d_path, 'wb') as handle:
        pickle.dump(clf_d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Named Entity Recog. Task
    tr.model_differences(df, configure_models_dirpath, models_path,save_location = output_parameters_dirpath)
    all_keys = tr.seperate_keys(df,configure_models_dirpath,models_path)

    # config_json['transferability_sparsity'] 

    # th = config_json['transferability_sparsity'] # th = 1.0
    all_models = tr.get_feature_set(df,configure_models_dirpath, models_path, all_keys, 
                                    config_json = config_json, saved_models_path =output_parameters_dirpath)
    
    features,labels = tr.extract_all_features(all_models)
    
    clf = MLPClassifier(solver='adam', alpha=1e-5,
                        hidden_layer_sizes=(256,128), random_state=1).fit(features, labels)
    
    clf_ner_path = os.path.join(output_parameters_dirpath,"clf_ner.pickle")
    
    with open(clf_ner_path, 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
                                    
                                    
    # all_models_path = os.path.join(output_parameters_dirpath,"all_models.pickle")
    
    # with open(all_models_path, 'wb') as handle:
    #     pickle.dump(all_models, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    
        
    #Question Answer Task
    # num_steps = config_json['loss_surface_num_steps']

    all_models,labels = ls.get_all_approx_mixup(df,configure_models_dirpath,models_path, config_json, save_path = output_parameters_dirpath)
    print(len(all_models))
    
    features,labels = extract_feature_loss(all_models,df)
    
    clf = MLPClassifier(solver='adam', alpha=1e-5,
                        hidden_layer_sizes=(256, 64, 32), random_state=1, max_iter = 230).fit(features,labels)
    
    detector_path = os.path.join(output_parameters_dirpath,"detector.pickle")
    
    with open(detector_path, 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    
    # from jsonargparse import ArgumentParser, ActionConfigFile
    # parser = ArgumentParser(description='UMD Trojan Detector - Round 9.')

    from argparse import ArgumentParser

    parser = ArgumentParser(description='UMD Trojan Detector - Round 9.')

    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')
    
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.')
    
    # parser.add_argument('--features_filepath', type=str, help='File path to the file where intermediate detector features may be written. After execution this csv file should contain a two rows, the first row contains the feature names (you should be consistent across your detectors), the second row contains the value for each of the column names.')
    
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    
    # parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    
    # parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    # parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)

    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.')

    # parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default=None)

    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")

    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in configure mode.')

    args, unknown = parser.parse_known_args()
    # args = parser.parse_args()  

    print('Known args:')
    print(args)  

    print('Unknown args:')
    print(unknown)

    # TODO - we do not validate the scheme, it sometimes throws exception, we will fix it and validate the config.

    try:
        with open(args.metaparameters_filepath, 'rb') as config_file:
            config_json = json.load(config_file)
    except Exception as e:
        print(e)
        config_json = {}
        
    # Validate config file against schema
    '''
    if args.metaparameters_filepath is not None:
        if args.schema_filepath is not None:
            with open(args.metaparameters_filepath[0]()) as config_file:
                config_json = json.load(config_file)

            with open(args.schema_filepath) as schema_file:
                schema_json = json.load(schema_file)

            # this throws a fairly descriptive error if validation fails
            jsonschema.validate(instance=config_json, schema=schema_json)
    '''

    if not args.configure_mode:
        if (args.model_filepath is not None and
                args.tokenizer_filepath is not None and
                args.result_filepath is not None and
                args.learned_parameters_dirpath is not None):

            trojan_detector(args.model_filepath,
                                    args.tokenizer_filepath,
                                    args.result_filepath,
                                    args.learned_parameters_dirpath,
                                    config_json)

        else:
            print("Required Evaluation-Mode parameters missing!")
    else:
        if (args.learned_parameters_dirpath is not None and
                args.configure_models_dirpath is not None):

            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.learned_parameters_dirpath,
                      args.configure_models_dirpath,
                      config_json)
        else:
            print("Required Configure-Mode parameters missing!")