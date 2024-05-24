
import os
import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from pytorch_lightning.plugins.environments.lightning_environment import LightningEnvironment
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule

import argparse

class LitClassifier(pl.LightningModule):
    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        acc = (logits.argmax(dim=-1) == y).float().mean()
        return acc

    def validation_epoch_end(self, outputs) -> None:

        self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True)

    def test_epoch_end(self, outputs) -> None:
        self.log("test_acc", torch.stack(outputs).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hosts", type=list, default=os.environ["SM_HOSTS"])
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--num-gpus", type=int, default=int(os.environ["SM_NUM_GPUS"]))

    parser.add_argument("--num_nodes", type=int, default = len(os.environ["SM_HOSTS"]))
           
    # num gpus is per node
    world_size = int(os.environ["SM_NUM_GPUS"]) * len(os.environ["SM_HOSTS"])
                 
    parser.add_argument("--world-size", type=int, default=world_size)
    
    parser.add_argument("--batch_size", type=int, default=int(os.environ["SM_HP_BATCH_SIZE"]))  
    
    parser.add_argument("--epochs", type=int, default=int(os.environ["SM_HP_EPOCHS"]))

    
    args = parser.parse_args()
    
    return args
    
    
if __name__ == "__main__":
        
    args = parse_args()
    
    cmd = 'ls {}'.format(args.train_dir)
    
    print ('Here is sample arbitrary csv train file!')
    
    os.system(cmd)
    
    dm = MNISTDataModule(batch_size=args.batch_size)
    
    model = LitClassifier()
    
    local_rank = os.environ["LOCAL_RANK"]
    torch.cuda.set_device(int(local_rank))
    
    num_nodes = args.num_nodes
    num_gpus = args.num_gpus
    
    env = LightningEnvironment()
    
    env.world_size = lambda: int(os.environ.get("WORLD_SIZE", 0))
    env.global_rank = lambda: int(os.environ.get("RANK", 0))
    
    ddp = DDPStrategy(cluster_environment=env, accelerator="gpu")
    
    trainer = pl.Trainer(max_epochs=args.epochs, strategy=ddp, devices=num_gpus, num_nodes=num_nodes, default_root_dir = args.model_dir)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    
