import pdb
import torch, click, matplotlib.pyplot as plt
from models.model import MyAwesomeModel
from data.dataset import MyDataset, TransformMnistProcessed
from datetime import datetime


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--ep", default=5, help="number of epochs to train for")
def train(lr, ep):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    print(ep)

    training_start_time = datetime.now()
    print(f"Training started at: {training_start_time}")

    model = MyAwesomeModel()
    imgs, lbls = torch.load("data/processed/train_data.pt")
    train_set = MyDataset(imgs, lbls, transform=TransformMnistProcessed(imgs))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

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

            if i % 64 == 0:
                print(f"loss @ {i}: {loss.item()}")
                with torch.no_grad():
                    model.eval()
                    accuracy = []
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    acc = torch.mean(equals.type(torch.FloatTensor))
                    accuracy.append(acc.item() * 100)
                    print(f"Accuracy: {torch.mean(torch.Tensor(accuracy))}%")
        losses.append(running_loss)

        plt.plot(list(range(len(losses))), losses)
        plt.savefig(f"reports/figures/loss_{training_start_time}.png")

        save_location = f"models/MyAwesomeModel_i{e+1}_L{running_loss:.1f}.pt"
        torch.save(model, save_location)

        print(f"Trained model saved to {save_location}")


cli.add_command(train)

if __name__ == "__main__":
    cli()
