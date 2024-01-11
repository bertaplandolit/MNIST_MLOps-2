import torch
from models.model import SimpleCNN
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import hydra
import logging



log = logging.getLogger(__name__)


@hydra.main(config_name = "config_train.yaml")

def train(config):
    """Train a model on MNIST."""
    lr=float(config.learning_rate)
    training_name = config.training_name
    epochs = config.epochs

    log.info(f"Training model with learning rate {lr}")
    print(lr)

    # initialize model
    model = SimpleCNN()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    data = torch.load("data/processed/processed_tensor.pt")
    train_set = data["train_loader"]

    steps = 0

    train_losses = []

    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_set:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_set)
        train_losses.append(average_loss)
        log.info(f"Epoch {e + 1}/{epochs}, Loss: {average_loss:.4f}")
    plt.plot(train_losses)

    if os.path.isdir("reports/figures/{}".format(training_name)) == False:
        os.system("mkdir reports/figures/{}".format(training_name))
    plt.savefig("reports/figures/{}/training.png".format(training_name))

    if os.path.isdir("models/{}".format(training_name)) == False:
        os.system("mkdir models/{}".format(training_name))
    torch.save(model.state_dict(), "models/{}/checkpoint.pth".format(training_name))




if __name__ == "__main__":
    train()
