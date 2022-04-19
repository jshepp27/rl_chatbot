import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Load Training Data from PyTorch Library
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download Test Data from PyTorch Library
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Pass defined dataset objects to *DataLoader* instance to return data.

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for x, y in test_dataloader:
#     print(f"Shape of x [N, C, H, W]: {x.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")

######################
# Construct NN Model
######################

# # Get CPU or GPU Devise for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define Model
# Inheret nn.Module Class, base-class for all neural network modules
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Reshape into a 1D tensor
        self.flatten = nn.Flatten()
        # Sequential Container - Modules will be added in the order they are passed to the constructor
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

###########################
# Optimise Model Parameters
###########################
# Note 'optimisation' implies model training. Where parameters are optimised (approximated) to best
# satisfy the constraint of the objective function.

# # Define Loss Function
# loss_fn = nn.CrossEntropyLoss()
# # Define optimisation method == Stochastic Gradient Descent
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# # Training Loop
# # In a single training loop, the model makes predictions on the training dataset (fed to it in batches)
# # And backpropogates the prediction error to adjust th emodel's parameters.
# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         # Convert Tensor to appropriate device configuration
#         X, y = X.to(device), y.to(device)
#
#         # Compute prediction error
#         pred = model(X)
#         # Construct Error Tensor
#         loss = loss_fn(pred, y)
#         # Explicitly set gradient to zero before starting back-propogation
#         # whilst in running the mini-batch.
#         # Gradients will be accumulated upon backpropogation
#         optimizer.zero_grad()
#
#         # Backpropogation (triggered upon calling loss.backward())
#         # Autograd then calculates and stores the gradient
#         # for each model paramete's .grad attribute
#         loss.backward()
#
#         # All parameters of the model are registered to the optimizer
#         # Initiate Gradient Descent Step
#         # thereby, triggering an update using the optimisers update rule
#         optimizer.step()
#
#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
#
# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)
#
#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)
#
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#
# # Test
# # We also check the model's performance against the test dataset to ensure it is learning
# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             # Convert Tensor to appropriate device configuration
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#
#     test_loss /= num_batches
#     correct /= size
#
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#
# # We run the training process over many iterations (Epochs).
# # During each Epoch, the model learns paramers (updates) to make better approximations.
# # We print the model's accuract and loss at each Epoch
# # We expect to see accuracy increase and loss decrease with each Epoch
#
# epochs = 5
#
# # Train
# for t in range(epochs):
#     print(f"Epoch {t+1} \n--------------------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print('Done!')
#
# # Save Model
# torch.save(model.state_dict(), "basic_nn_model.pth")
# print("Saved Basic nn to 'basic_nn_model.pth'")

# Loading Models
model = NeuralNetwork()
model.load_state_dict(torch.load("basic_nn_model.pth"))

# Use the loaded model to generate predictions
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: {predicted}", f"Actual: {actual}")









