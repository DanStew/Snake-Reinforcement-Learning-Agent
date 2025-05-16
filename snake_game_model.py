import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Creating two linear layers for our neural network
        # Param1 : input size Param2 : output size
        # Below defines input --> hidden --> output
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # Function to move forward between the layers in the neural network
    def forward(self, x):
        # Going through the first layer, optimising the value as well
        x = F.relu(self.linear1(x))
        # Going through the second layer
        x = self.linear2(x)
        # Returning the output of the second layer
        return x

    # Function to save the model
    def save(self, file_name="model.pth"):
        # Making another folder to store our models
        model_folder_path = "./model"
        # If the model folder doesn't exist, make the model folder
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        # Appending the model folder path to the given filename
        file_name = os.path.join(model_folder_path, file_name)
        # Saving the model at that file_name path
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # Optimising our model, using the parameters and the learning rate
        self.optimiser = optim.Adam(model.parameters(), lr=self.lr)
        # Creating the Loss Function / Criterion Function
        # Our Loss Function is the Mean Squared Error function
        self.criterion = nn.MSELoss()

    # The trainstep for our model
    # This model can be called with individual items or lists of items, so we must convert this first
    def train_step(self, state, action, reward, next_state, done):
        # Converting some of the inputs to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        # Checking if we are working with 1 value or lists of values
        if len(state.shape) == 1:
            # Needs these attributes in the form (1,x) so this is what the code below is doing
            # If we have a list of attributes, they are already in this form so we don't need to worry then
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Predict the Q values with the current state
        pred = self.model(state)
        # Creating a clone of the prediction
        target = pred.clone()
        # Going over all the values in the tensor
        for idx in range(len(done)):
            # Setting Q_new to be the reward at the current index
            Q_new = reward[idx]
            # Only apply the next parameters formula if the game isn't over at this step
            if not done[idx]:
                # Applying the formula to get the next set of parameters
                # Only multiple by 1 value, the max of the predicted Q value for the next state
                Q_new = Q_new + self.gamma * torch.max(self.model(next_state[idx]))

            # Setting the new target value to be Q_new at the maximum value of action
            target[idx][torch.argmax(action).item()] = Q_new

        # Applying the Loss Function
        self.optimiser.zero_grad()  # Emptying the gradient (step needed to learn within PyTorch)
        loss = self.criterion(target, pred)
        loss.backward()  # Applying backpropagation

        self.optimiser.step()
