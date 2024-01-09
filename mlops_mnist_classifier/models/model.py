from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch import exp, mean, FloatTensor as T_FloatTensor
import wandb

class MyAwesomeModel(pl.LightningModule):
    """My awesome model."""

    def __init__(self, lr=1e-3, ep=5):
        super().__init__()

        self.learning_rate = lr
        self.n_epochs = ep

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.logsm = nn.LogSoftmax(dim=1)

        self.ffnn = nn.Sequential(self.fc1, self.relu, self.fc2, self.relu, self.fc3, self.logsm)

        self.criterion = nn.NLLLoss()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        ## add lr scheduler
        return optimizer
    
    def forward(self, x):
        conv_out_1 = self.conv1(x)
        conv_out_2 = self.conv2(conv_out_1)
        z = conv_out_2.view(x.size(0), -1)
        output = self.ffnn(z)
        return output, z

    def training_step(self, batch, batch_idx):
        images, labels = batch
        log_ps,_ = self(images.unsqueeze(1))
        train_loss = self.criterion(log_ps, labels)

        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.logger.experiment.log({'logits': wandb.Histogram(exp(log_ps).detach().cpu())})

        return train_loss

    def validation_step(self, batch, batch_idx):
        images,labels = batch
        log_ps, _ = self(images.unsqueeze(1))
        ps = exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        val_acc = mean(equals.type(T_FloatTensor))

        self.log("val_acc", val_acc)