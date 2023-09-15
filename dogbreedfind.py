import torch
from torchvision import models, transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 사전 훈련된 ResNet-50 모델 로드
model_conv = models.resnet152(pretrained=True).to(device)

# 마지막 분류 계층 제거
new_classifier = nn.Sequential(*list(model_conv.children())[:-2]).to(device)
model_conv = new_classifier

# 모델 정의
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

classifier = SimpleClassifier(100352).to(device)
classifier.load_state_dict(torch.load("./models/resnet/model_resnet152_epoch_36.pth"))

def predict_breed(image):
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # image가 PIL 객체라면 바로 전처리를 적용
    image = transform(image).unsqueeze(0).to(device)

    # 모델을 평가 모드로 설정
    model_conv.eval()
    classifier.eval()

    # 예측 수행
    with torch.no_grad():
        features = model_conv(image)
        features = features.view(features.size(0), -1)
        outputs = classifier(features)
        _, predicted_idx = torch.max(outputs, 1)

    dog_classes = ['Beagle', 'BostonTerrier', 'Dachshund', 'FrenchBulldog', 'Goldenretriever',
                   'Labradorretriever', 'Pomeranian', 'Siberianhusky', 'Welshcorgi', 'Yorkshireterrier']

    return dog_classes[predicted_idx.item()]
