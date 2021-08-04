import torch
from torchvision.transforms import transforms

from resnet18 import Net
from gb688Dataset import gb688Dataset
PATH = "/Users/chenziyang/Documents/Ziyang/Crawler/中石化.nosync/验证码/CAPTCHA_Validation/Train/Trainingcheckpointgb688Dataset.pth"
model = Net([60, 38, 3])
model.load_state_dict(torch.load(PATH))
model.eval()

normal_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914), (0.2023))
])

dataset = gb688Dataset("../Data",transform=normal_train_transform)
test_loader = torch.utils.data.DataLoader(dataset)
for (data, label) in test_loader:
    print(label)
    print(model(data))
    break