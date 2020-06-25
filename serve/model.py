import torch.nn as nn

class FFClassifier(nn.Module):
    """
    This is the simple feed-forward model we will be using to perform ticket-classification.
    """

    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        """
        Initialize the model by settingg up the various layers.
        """
        super(FFClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim,hidden_dim1)
        self.relu = nn.ReLu()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.softmax = nn.LogSoftmax(dim=1)
        

    def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.softmax(output)
            return output