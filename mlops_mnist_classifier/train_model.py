import pytorch_lightning as pl
import torch, click, matplotlib.pyplot as plt
from models.model import MyAwesomeModel
from data.dataset import MyDataset, TransformMnistProcessed
from datetime import datetime

import wandb


@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--ep", default=5, help="number of epochs to train for")
@click.argument("save_folder")
def train_manually(save_folder, lr, ep):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    print(ep)

    wandb.init(
        project="MLOPS CNN",
        notes="My first experiment",
        )

    training_start_time = datetime.now()
    print(f"Training started at: {training_start_time}")

    model = MyAwesomeModel()
    imgs, lbls = torch.load("data/processed/train_data.pt")
    train_set = MyDataset(imgs, lbls, transform=TransformMnistProcessed(imgs))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    test_set = MyDataset(imgs, lbls, transform=TransformMnistProcessed(imgs))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for e in range(ep):
        print(f"epoch {e+1}/{ep}")
        running_loss = 0
        for i, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()

            # pdb.set_trace()
            log_ps, _ = model(images.unsqueeze(1))
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 250 == 0:
                with torch.no_grad():
                        model.eval()
                        accuracy = []
                        errors, total = 0, 0
                        for images, labels in testloader:
                            log_ps, _ = model(images.unsqueeze(1))
                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            acc = torch.mean(equals.type(torch.FloatTensor))
                            accuracy.append(acc)
                        avg_acc = torch.mean(torch.Tensor(accuracy))
                        print(f"Accuracy: {avg_acc}%")
                        print(f"Errors/Total: {errors}/{total}")
                wandb.log({
                    "loss": running_loss,
                    "accuracy": avg_acc,
                    })
  

        losses.append(running_loss)

        plt.plot(list(range(len(losses))), losses)
        plt.savefig(f"reports/figures/loss_{training_start_time}.png")

        save_location = f"{save_folder}/MyAwesomeModel_i{e+1}_L{running_loss:.1f}.pt"
        torch.save(model, save_location)

        print(f"Trained model saved to {save_location}")

# @click.argument("save_folder")
@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--ep", default=5, help="number of epochs to train for")
def train(lr, ep):
    """Train a model on MNIST."""

    wandb.init(
        project="MLOPS CNN",
        notes="My first experiment [TorchLightning]",
        )

    training_start_time = datetime.now()
    print(f"Training started at: {training_start_time}")

    model = MyAwesomeModel(lr, ep)
    imgs, lbls = torch.load("data/processed/train_data.pt")
    train_set = MyDataset(imgs, lbls, transform=TransformMnistProcessed(imgs))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    test_set = MyDataset(imgs, lbls, transform=TransformMnistProcessed(imgs))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    trainer = pl.Trainer(max_epochs=5, logger=pl.loggers.WandbLogger(project='MNIST simple CNN'), precision="16-mixed")
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader)



cli.add_command(train_manually)
cli.add_command(train)

if __name__ == "__main__":
    cli()
