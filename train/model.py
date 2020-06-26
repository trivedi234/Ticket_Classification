import torch.nn as nn

class FFClassifier(nn.Module):
    """
    This is the simple feed-forward model we will be using to perform ticket-classification.
    """

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        """
        Initialize the model by settingg up the various layers.
        """
        super(FFClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim,hidden_dim1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        

    def forward(self, x):
            hidden1 = self.fc1(x)
            relu1 = self.relu(hidden1)
            hidden2 = self.fc2(relu1)
            relu2 = self.relu(hidden2)
            output = self.fc3(relu2)
            output = self.softmax(output)
            return output