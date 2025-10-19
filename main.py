from helper_lib import get_data_loader, train_cnn
import torch, os

def main():
    os.makedirs("models", exist_ok=True)
    train_loader = get_data_loader("./data", batch_size=128, train=True)
    model, device = train_cnn(epochs=5, train_loader=train_loader)
    torch.save(model.state_dict(), "models/cnn_cifar10.pth")
    print("Saved -> models/cnn_cifar10.pth")

if __name__ == "__main__":
    main()
