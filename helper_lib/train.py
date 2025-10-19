import torch, torch.nn as nn, torch.optim as optim
from .model import get_model

def train_cnn(epochs=3, lr=1e-3, device=None, train_loader=None, num_classes=10):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("CNN", num_classes=num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"[epoch {ep+1}/{epochs}] loss={running/len(train_loader):.4f}")
    return model, device
