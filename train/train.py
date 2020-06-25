import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

from model import FFClassifier

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
    model = FFlassifier(model_info['input_dim'], model_info['hidden_dim1'], model_info['hidden_dim2'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved transformers.
    transformer_path = os.path.join(model_dir, 'transformers.pkl')
    with open(transformer_path, 'rb') as f:
        model.transformers = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_train_data_loader(batch_size, training_dir, file_name):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, file_name), header=None, names=None)

    train_sample_y = torch.from_numpy(train_sample[[-1]].values).float().squeeze()
    train_sample_X = torch.from_numpy(train_sample.drop([-1], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    # TODO: Paste the train() method developed in the notebook here.

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # TODO: Complete this train method to train the model provided.
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)
            total_loss += loss.data.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--train_set', type=int, default="train1.csv", metavar='S',
                        help='filename for the training_set')

    # Model Parameters
    parser.add_argument('--input_dim', type=int, default=4080, metavar='N',
                        help='size of the input features (default: 4080)')
    parser.add_argument('--hidden_dim1', type=int, default=128, metavar='N',
                        help='size of the first hidden dimension (default: 128)')
    parser.add_argument('--hidden_dim2', type=int, default=64, metavar='N',
                        help='size of the second hidden dimension (default: 64)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, args.train_set)

    # Build the model.
    model = FFClassifier(args.input_dim, args.hidden_dim1, args.hidden_dim2).to(device)

    with open(os.path.join(args.data_dir, "transformers.pkl"), "rb") as f:
        model.transformers = pickle.load(f)

    print("Model loaded with input_dim {}, hidden_dim1 {}, hidden_dim2 {}.".format(
        args.input_dim, args.hidden_dim1, args.hidden_dim2
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.NLLLoss()

    train(model, train_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_dim': args.input_dim,
            'hidden_dim1': args.hidden_dim1,
            'hidden_dim2': args.hidden_dim2,
        }
        torch.save(model_info, f)

	# Save the transformers
    transformer_path = os.path.join(args.model_dir, 'transformers.pkl')
    with open(transformer_path, 'wb') as f:
        pickle.dump(model.transformers, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
