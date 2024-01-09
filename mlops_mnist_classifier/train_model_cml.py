import pytorch_lightning as pl
import torch, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from models.model import MyAwesomeModel
from data.dataset import MyDataset, TransformMnistProcessed
from datetime import datetime

import wandb

def main(lr=1e-3, ep=5):
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

    preds, target = [], []
    for batch in trainloader:
        x, y = batch
        probs = model(x)
        preds.append(probs.argmax(dim=-1))
        target.append(y.detach())

    target = torch.cat(target, dim=0)
    preds = torch.cat(preds, dim=0)

    report = classification_report(target, preds)
    with open("classification_report.txt", 'w') as outfile:
        outfile.write(report)
    confmat = confusion_matrix(target, preds)
    disp = ConfusionMatrixDisplay(cm = confmat, )
    plt.savefig('confusion_matrix.png')




if __name__ == "__main__":
    main()