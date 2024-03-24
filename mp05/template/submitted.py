import torch
import torch.nn as nn


def create_sequential_layers():
    """
    Task: Create neural net layers using nn.Sequential.

    Requirements: Return an nn.Sequential object, which contains:
        1. a linear layer (fully connected) with 2 input features and 3 output features,
        2. a sigmoid activation layer,
        3. a linear layer with 3 input features and 5 output features.
    """
    linear_layer = nn.Linear(in_features= 2, out_features=3)
    sigmoid_activation = nn.Sigmoid()
    last_linear = nn.Linear(in_features=3, out_features=5)
    network = nn.Sequential(linear_layer,sigmoid_activation, last_linear)
    return network

def create_loss_function():
    """
    Task: Create a loss function using nn module.

    Requirements: Return a loss function from the nn module that is suitable for
    multi-class classification.
    """
    return nn.CrossEntropyLoss()


class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        ################# Your Code Starts Here #################
        self.flatten = nn.Flatten()
        self.activation = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32, kernel_size= 3), nn.MaxPool2d(32) )
        
        self.hidden = nn.Linear(2883,300) # Only define the out_features, it will initialize the input size during the first forward pass.
        self.activation2 = nn.ReLU()
        self.output = nn.Linear(in_features=300, out_features=5)    
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        
        # Max pooling over a (2, 2) window
        
        
        x  = self.activation(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)

        x  = self.hidden(x)
        print(x.shape)
        x = self.activation2(x)
        print(x.shape)
        
        x = self.output(x)
        
        
        ################## Your Code Ends here ##################
        return x

def train(train_dataloader, epochs):
    """
    The autograder will call this function and compute the accuracy of the returned model.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
    """

    ################# Your Code Starts Here #################
    """
    Implement backward propagation and gradient descent here.
    """
    # Create an instance of NeuralNet, a loss function, and an optimizer
    model = NeuralNet()
    loss_fn = create_loss_function()
    optimizer = torch.optim.Adam(params=  model.parameters())
    print(train_dataloader)
    model.train()

    for epoch in range(epochs):
        for sample_batch, test_batch in train_dataloader:
            predicted = model(sample_batch)
            loss = loss_fn(predicted,test_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    ################## Your Code Ends here ##################

    return model
