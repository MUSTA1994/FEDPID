# FEDPID
CODE

import torch
from torch import nn, optim
from torchvision import datasets, transforms
import copy
import matplotlib.pyplot as plt
import numpy as np

# define data transformation
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# load MNIST dataset
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
# number of clients
N = 10
# partition trainset into non-overlapping subsets for each client
trainset_partitioned = []
for n in range(N):
    indices = (trainset.targets >= n * (10 // N)) & (trainset.targets < (n + 1) * (10 // N))
    trainset_partitioned.append(torch.utils.data.Subset(trainset, indices.nonzero().squeeze()))


# define CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(nn.functional.max_pool2d(x, 2))
        x = self.conv2(x)
        x = nn.functional.relu(nn.functional.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x)


# initialize model for each client
models = [CNN() for _ in range(N)]

# initialize learning rates for each client
lambda_n = [0.01 for _ in range(N)]

# define loss function
criterion = nn.NLLLoss()

# number of local epochs
local_epochs = 20

# number of communication rounds
T = 100

# initialize lists to store training metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for t in range(T):
    print(f"Iteration {t}/{T}")

    # update local model for each client
    for n in range(N):
        # train local model for local_epochs using partitioned trainset for client n
        for epoch in range(local_epochs):
            running_loss = 0
            correct = 0
            total = 0
            # Inside the loop for training each client's local model
            for images, labels in torch.utils.data.DataLoader(trainset_partitioned[n], batch_size=64):
                # zero the parameter gradients
                models[n].zero_grad()
                # forward pass
                output = models[n](images)
                loss = criterion(output, labels)

                # backward pass
                loss.backward()
                # update parameters using modified SGD optimizer
                with torch.no_grad():
                    for name, param in models[n].named_parameters():
                        param -= lambda_n[n] * param.grad + (1 / N) * torch.sum(
                            torch.stack([models[i].state_dict()[name] for i in range(N)]) - param)
                running_loss += loss.item()
                # calculate training accuracy
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'Client {n} Epoch {epoch} loss: {running_loss / len(trainset_partitioned[n])}')
            train_losses.append(running_loss / len(trainset_partitioned[n]))
            train_accuracies.append(correct / total)

    # server aggregates updated model parameters from all clients
    global_model = copy.deepcopy(models[0])
    for name in global_model.state_dict().keys():
        global_model.state_dict()[name].data.copy_(sum([models[n].state_dict()[name] for n in range(N)]) / N)

    # server updates local models of all clients with aggregated model parameters
    for n in range(N):
        models[n].load_state_dict(global_model.state_dict())

    # server calculates minimization formula to find optimal learning rates for each client
    for n in range(N):
        lambda_n[n] = np.argmin(
            lambda_n * torch.norm(models[n].state_dict()['fc2.weight'].grad) - global_model.state_dict()['fc2.weight'])

    # evaluate global model on validation set
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testset:
            output = global_model(images)
            val_loss += criterion(output, labels).item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_losses.append(val_loss / len(testset))
    val_accuracies.append(correct / total)

    print(f"Validation Loss: {val_loss / len(testset)}")
    print(f"Validation Accuracy: {correct / total}")

# plot training metrics
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()
