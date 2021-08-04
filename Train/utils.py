from torchvision import transforms
import torch
from torchvision import transforms, datasets
import numpy as np
import gb688Dataset
from pytorch_metric_learning import testers

# Transformations for different uses
normal_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914), (0.2023))
])
# The dictionary stores the corresponding parameters of each dataset
param_dic = {"gb688Dataset": [60, 38, 3]}


# Function from pytorch-metric-learning to the embeddings
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


# Train the model witin one epoch and returns the corresponding learning information
def train(model, loss_func, mining_func, train_loader, optimizer, epoch, device=None):
    # The information recorded from one epoch
    epoch_data = list()
    # for batch_idx, (data, labels) in enumerate(train_loader):
    for batch_idx, (data, labels) in enumerate(train_loader):
        # print(data.shape)
        if device:
            data, labels = data.to(device), torch.tensor(labels).to(device)
        else:
            labels = torch.tensor(labels)
        optimizer.zero_grad()
        embeddings = model(data)  # (batch size, embedding_size)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        # The information recorded from one iteration
        iteration_data = [batch_idx, loss.cpu().data.numpy().tolist(), mining_func.num_triplets]
        epoch_data.append(iteration_data)
        if batch_idx % 1 == 0:
            iteration_data = [batch_idx, loss, mining_func.num_triplets]
            print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(epoch, batch_idx, loss,
                                                                                           mining_func.num_triplets))
    return epoch_data


# The test function
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    print("Computing accuracy")
    # Compute accuracy using AccuracyCalculator from pytorch-metric-learning
    accuracies = accuracy_calculator.get_accuracy(test_embeddings,
                                                  train_embeddings,
                                                  np.squeeze(test_labels),
                                                  np.squeeze(train_labels),
                                                  True)
    print(accuracies.keys())
    print(accuracies.values())
    return accuracies.values()


# Return the corresponding dataset based on the parameters
# The Normal Sample Rate is applied here to prodcue a combined dataset
def get_dataset(dataset_name, trans_index=None):
    # augmented_train_transform = transfrom_dic[trans_index]
    dataset = gb688Dataset.gb688Dataset("../Data", transform=normal_train_transform)
    return dataset


# Return the corresponding parameter of each dataset
def get_param(data_name):
    return param_dic[data_name]
