import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import FFClassifier

from utils import word_processor

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved transformers.
    transformer_path = os.path.join(model_dir, 'transformers.pkl')
    with open(transformer_path, 'rb') as f:
        model.transformer = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.transformers is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
    
    # Process input_data so that it is ready to be sent to our model.

    stowords_set = model.transformers["stopwords"]
    lemmatizer = model.transformers["lemmatizer"]
    le = model.transformers["encoder"]
    vectorizer = model.transformers["vectorizer"]
    
    data_X = word_processor(input_data)
    data_X = vectorizer.transform(data_x)
    data_X = data_X.reshape(1, -1)
    
    data = torch.from_numpy(data_X)
    data = data.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    # TODO: Compute the result of applying the model to the input data. The variable `result` should
    #       be a numpy array which contains a single integer which is either 1 or 0

    result = model(data).numpy()
    result = str(le.inverse_transform(result)[0])                            
    
    return result
