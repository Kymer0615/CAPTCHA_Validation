
# !pip install pytorch-metric-learning --pre
# !pip install faiss-gpu
from torchvision import transforms
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torch
import torch.optim as optim
from torch.utils.data import random_split
import pickle
import gb688Dataset
import resnet18
import utils
from os import getcwd
# Deep metric learning related functions
mining_func = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")
loss_func = losses.TripletMarginLoss(margin=0.2)
accuracy_calculator = AccuracyCalculator(include=("mean_average_precision_at_r", "AMI", "NMI"), k=10)

# Neural network realted varaibles
BATCH_SIZE = 225
EPOCHS = {"gb688Dataset": 60}
# The datasets used to do the experiment
datasets = ["gb688Dataset"]

for dataset_name in datasets:
    device = torch.device("cuda")
    # model = resnet18.Net(utils.get_param(dataset_name)).to(device)
    model = resnet18.Net(utils.get_param(dataset_name))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Clear the parameters and caches
    model.zero_grad()
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    # Get the training and testing datasets
    dataset= utils.get_dataset(dataset_name)
    train_dataset, test_dataset = random_split(dataset, [500, 100], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE)

    # The information recorded from one complete run
    sample_data = list()
    epochs = EPOCHS[dataset_name]
    for epoch in range(1, epochs + 1):
        # The training and testing functions
        epoch_data = utils.train(model, loss_func, mining_func, train_loader, optimizer, epoch, device=None)
        accuracies = utils.test(train_dataset, test_dataset, model, accuracy_calculator)
        for accuracy in accuracies:
            epoch_data.append(accuracy)
        sample_data.append(epoch_data)

    # Save the data to a local file
    # data_file_name = "./output/"+ dataset_name +"_"+ str(round(normal_sample_rate, 1)) + ".pkl"
    # open_file = open(data_file_name, "wb")
    # pickle.dump(sample_data, open_file)
    # open_file.close()

    # Save the checkpoint to a local file(uncomment to use)

    parameter_file_name = getcwd() + 'checkpoint' + dataset_name + '.pth'
    torch.save(model.state_dict(), parameter_file_name)
