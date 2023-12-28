import sys
import yaml
import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F

#class conv_model(nn.Module):
class conv_model(L.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.maxpool = nn.MaxPool1d(4)

        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=64,
                               kernel_size=33,
                               stride=1,
                               padding='valid')
        
        self.conv2 = nn.Conv1d(in_channels=64,
                               out_channels=32,
                               kernel_size=17,
                               stride=1,
                               padding='valid')

        self.conv3 = nn.Conv1d(in_channels=32,
                               out_channels=16,
                               kernel_size=9,
                               stride=1,
                               padding='valid')

        self.fc1 = nn.Linear(1936,32)
        self.fc2 = nn.Linear(32,3)    #6)


    def forward(self,audio_tensor):
        
        x = self.conv1(audio_tensor)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        #x = self.sigmoid(x)
        x = self.relu(x)

        return x

    def training_step(self, batch, batch_idx):
        
        #x, y1,y2,y3,y4,y5,y6 = batch
        x, y1,y2,y3 = batch
        y_pred = self.forward(x)
        y = torch.cat((y1.unsqueeze(dim=0),
                       y2.unsqueeze(dim=0),
                       y3.unsqueeze(dim=0)),0)
                       #y4.unsqueeze(dim=0),
                       #y5.unsqueeze(dim=0),
                       #y6.unsqueeze(dim=0)),0)
        
        y = torch.transpose(y, 0, 1)
        loss = F.mse_loss(y.float(),y_pred)
        self.log("train_loss",loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y1,y2,y3 = batch
        y_pred = self.forward(x)
        y = torch.cat((y1.unsqueeze(dim=0),
                       y2.unsqueeze(dim=0),
                       y3.unsqueeze(dim=0)),0)
        y = torch.transpose(y, 0, 1)
        val_loss = F.mse_loss(y.float(),y_pred)
        self.log("val_loss",val_loss)

    def test_step(self,batch, batch_idx):

        x, y1,y2,y3 = batch
        y_pred = self.forward(x)
        y = torch.cat((y1.unsqueeze(dim=0),
                       y2.unsqueeze(dim=0),
                       y3.unsqueeze(dim=0)),0)
        y = torch.transpose(y, 0, 1)
        test_loss = F.mse_loss(y.float(),y_pred)
        self.log("test_loss",test_loss)

    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer




