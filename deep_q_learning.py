import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

"""
- Feed Forward Neural Network (from input nodes through hidden and to the output) -> no loops
- Single hidden layer 
"""


# Build the Network
class Linear_Network(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Creates single layer feed forward network
        # Format : Ax = B (All matrices/vectors)
        #   A : weight
        #   x : input 
        #   B : output
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Sets an Activation Function defined as relu(x) = { 0 if x < 0, x if x > 0}
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, filename = "model.pth"):
        # Saves the Deep Q Model in a file
        make_dir = "./model"
        if os.path.exists(make_dir) == False:
            os.makedirs(make_dir)

        filename = os.path.join(make_dir, filename)
        torch.save(self.state_dict(), filename)


# Train and optimize the Network
class QTrainer:
    
    def __init__(self, model, alpha, gamma):

        # Initialization of the hyperparameters
        self.alpha = alpha
        self.gamma = gamma

        # Initialization of the Q-Network
        self.model = model

        # Optimizer helps adjust the parameters of the neural network (weights, learning rates...)
        self.optimizer = optim.Adam(model.parameters(), lr=self.alpha)

        # Measures the mean squared error
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):

        # Initialization of Pytorch tensors
        state = torch.tensor(state, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.float)
        reward = torch.tensor(reward, dtype = torch.float)

        # Check if the dimension of the tensor State is 1 (vector)
        if len(state.shape) == 1:
            # Convert tensor from 1D to 2D for batch processing
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred_q_val = self.model(state) # Tensor of predicted q-values for a given state

        target = pred_q_val.clone() # Creates a copy of the tensor

        # Calculate the new q values
        for i in range(len(done)):
            q_new = reward[i]
            if not done[i]:
                q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            target[i][torch.argmax(action[i]).item()] = q_new

        # Apply the loss function (mean squared error)
        self.optimizer.zero_grad() # Set the gradients to 0
        loss = self.criterion(target, pred_q_val) # Calculates the loss: (target - pred)^2
        loss.backward() # Applies back propagation and updates the gradients
        self.optimizer.step()






        






    
        
    


