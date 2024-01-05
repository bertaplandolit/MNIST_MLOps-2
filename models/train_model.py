import click
import torch
from model import MyAwesomeModel
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=0.03, help="learning rate to use for training")
@click.option("--training_name", help="name of the training for generating subfolders")
def train(lr, training_name):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # initialize model
    model = MyAwesomeModel()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    data = torch.load("data/processed/processed_tensor.pt")
    train_set = data["train_loader"]

    epochs = 5
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
        print(f"Epoch {e + 1}/{epochs}, Loss: {average_loss:.4f}")
    plt.plot(train_losses)

    if os.path.isdir("reports/figures/{}".format(training_name)) == False:
        os.system("mkdir reports/figures/{}".format(training_name))
    plt.savefig("reports/figures/{}/training.png".format(training_name))

    if os.path.isdir("models/{}".format(training_name)) == False:
        os.system("mkdir models/{}".format(training_name))
    torch.save(model.state_dict(), "models/{}/checkpoint.pth".format(training_name))


cli.add_command(train)


if __name__ == "__main__":
    cli()
