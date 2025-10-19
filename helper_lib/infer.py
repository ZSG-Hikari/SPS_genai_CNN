from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

_TFM64 = transforms.Compose([
    transforms.Resize((64, 64)),   # match SpecCNN
    transforms.ToTensor(),
    transforms.Normalize([0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616])
])

def preprocess(file_bytes: bytes) -> torch.Tensor:
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    return _TFM64(img).unsqueeze(0)

def predict(model, device, x: torch.Tensor):
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(dim=0)
    return CIFAR10_CLASSES[idx.item()], float(conf), probs.cpu().tolist()
