import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as ptl
# Local 
from model.model import Model
from data import CustomDataset
from hparams import hparams

class TrainingModule(ptl.LightningModule):
    def __init__(self, criterion, model):
        super().__init__()
        
        self.model = model
        self.criterion = criterion

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def train_dataloader(self):

        data = CustomDataset()
        dataloader = DataLoader(data, **hparams.dataloader)
        return dataloader
    
    def validation_dataloader(self):
        data = CustomDataset()
        dataloader = DataLoader(data, **hparams.dataloader)
        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = hparams.training['lr'])
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return optimizer #, scheduler

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        
        loss = self.criterion(outputs, targets) 
        
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)
        
        loss = self.criterion(outputs, targets) 
        
        return loss
  


model = Model()
criterion = nn.CrossEntropyLoss()

module = TrainingModule(criterion = criterion, model = model)

trainer = ptl.Trainer(**hparams.trainer)
trainer.fit(module)