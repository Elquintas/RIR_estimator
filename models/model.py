import sys
import yaml
import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_lightning import  TrainResult


# Load config file
yaml_path = './configs/parameters.yaml'
with open(yaml_path, 'r') as file:
    cfg = yaml.safe_load(file)




##############################################################
#                                                            #
# Convolutional Model to predict the  mean absorption        #
# coefficients for each frequency band                       #
#                                                            #
##############################################################
class conv_model(L.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.save_hyperparameters()
        self.test_results = torch.empty((0, 12), dtype=torch.float32)

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
        self.fc2 = nn.Linear(32,6)


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
        x = self.sigmoid(x)
        #x = self.relu(x)

        return x

    def training_step(self, batch, batch_idx):
        
        x, y1,y2,y3,y4,y5,y6,_,_,_ = batch
        #x, y1,y2,y3 = batch
        y_pred = self.forward(x)
        y = torch.cat((y1.unsqueeze(dim=0),
                       y2.unsqueeze(dim=0),
                       y3.unsqueeze(dim=0),
                       y4.unsqueeze(dim=0),
                       y5.unsqueeze(dim=0),
                       y6.unsqueeze(dim=0)),0)
        
        y = torch.transpose(y, 0, 1)
        loss = F.mse_loss(y.float(),y_pred)
        self.log("train_loss",loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y1,y2,y3,y4,y5,y6,_,_,_ = batch
        y_pred = self.forward(x)
        y = torch.cat((y1.unsqueeze(dim=0),
                       y2.unsqueeze(dim=0),
                       y3.unsqueeze(dim=0),
                       y4.unsqueeze(dim=0),
                       y5.unsqueeze(dim=0),
                       y6.unsqueeze(dim=0)),0)
        y = torch.transpose(y, 0, 1)
        val_loss = F.mse_loss(y.float(),y_pred)
        self.log("val_loss",val_loss)

    def test_step(self,batch, batch_idx):

        x, y1,y2,y3,y4,y5,y6,_,_,_ = batch
        y_pred = self.forward(x)
        y = torch.cat((y1.unsqueeze(dim=0),
                       y2.unsqueeze(dim=0),
                       y3.unsqueeze(dim=0),
                       y4.unsqueeze(dim=0),
                       y5.unsqueeze(dim=0),
                       y6.unsqueeze(dim=0)),0)
        y = torch.transpose(y, 0, 1)
        test_loss = F.mse_loss(y.float(),y_pred)

        self.log("test_loss",test_loss)
    
        outputs = torch.cat((y.float(),y_pred),-1)

        self.test_results = torch.cat((self.test_results,outputs),0)




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg['lr'])
        return optimizer





##############################################################
#                                                            #
# Convolutional Model to predict in tandem the absorption    #
# coefficients for each band and the room geometry (x,y,z)   #
#                                                            #
##############################################################
class conv_model_multi_task(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.save_hyperparameters()
        self.test_step_outputs = []
    
        self.test_results = torch.empty((0, 18), dtype=torch.float32)

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

        self.fc1 = nn.Linear(1936,512)
        self.fc2 = nn.Linear(512,32)

        self.fc_absorptions = nn.Linear(32,6)
        self.fc_geometry = nn.Linear(32,3)

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
        x = self.relu(x)

        x_out1 = self.fc_absorptions(x)
        x_out1 = self.sigmoid(x_out1)

        x_out2 = self.fc_geometry(x)
        x_out2 = self.relu(x_out2)

        return x_out1, x_out2

    def training_step(self, batch, batch_idx):

        x, y1,y2,y3,y4,y5,y6, g_x,g_y,g_z = batch
        
        y_pred, g_pred = self.forward(x)
        y = torch.cat((y1.unsqueeze(dim=0),
                       y2.unsqueeze(dim=0),
                       y3.unsqueeze(dim=0),
                       y4.unsqueeze(dim=0),
                       y5.unsqueeze(dim=0),
                       y6.unsqueeze(dim=0)),0)
        
        g = torch.cat((g_x.unsqueeze(dim=0),
                       g_y.unsqueeze(dim=0),
                       g_z.unsqueeze(dim=0)),0)

        g = torch.transpose(g, 0, 1)
        y = torch.transpose(y, 0, 1)

        loss = cfg['loss_weight_abspt']*F.mse_loss(y.float(),y_pred) +\
               cfg['loss_weight_geom']*F.mse_loss(g.float(),g_pred)
        
        self.log("train_loss",loss)
                
        return loss
        

    def validation_step(self, batch, batch_idx):
        x, y1,y2,y3,y4,y5,y6, g_x,g_y,g_z = batch
        y_pred, g_pred = self.forward(x)
        y = torch.cat((y1.unsqueeze(dim=0),
                       y2.unsqueeze(dim=0),
                       y3.unsqueeze(dim=0),
                       y4.unsqueeze(dim=0),
                       y5.unsqueeze(dim=0),
                       y6.unsqueeze(dim=0)),0)

        g = torch.cat((g_x.unsqueeze(dim=0),
                       g_y.unsqueeze(dim=0),
                       g_z.unsqueeze(dim=0)),0)

        g = torch.transpose(g, 0, 1)
        y = torch.transpose(y, 0, 1)

        val_loss = cfg['loss_weight_abspt']*F.mse_loss(y.float(),y_pred) +\
                cfg['loss_weight_geom']*F.mse_loss(g.float(),g_pred)

        self.log("val_loss",val_loss)

    def test_step(self,batch, batch_idx):

        x, y1,y2,y3,y4,y5,y6, g_x,g_y,g_z = batch
        y_pred, g_pred = self.forward(x)
        y = torch.cat((y1.unsqueeze(dim=0),
                       y2.unsqueeze(dim=0),
                       y3.unsqueeze(dim=0),
                       y4.unsqueeze(dim=0),
                       y5.unsqueeze(dim=0),
                       y6.unsqueeze(dim=0)),0)

        g = torch.cat((g_x.unsqueeze(dim=0),
                       g_y.unsqueeze(dim=0),
                       g_z.unsqueeze(dim=0)),0)

        g = torch.transpose(g, 0, 1)
        y = torch.transpose(y, 0, 1)
        
        test_loss = cfg['loss_weight_abspt']*F.mse_loss(y.float(),y_pred) +\
                cfg['loss_weight_geom']*F.mse_loss(g.float(),g_pred)
        
        self.log("test_loss",test_loss)
        
        outputs = torch.cat((y.float(),y_pred,g.float(),g_pred),-1)
        
        self.test_results = torch.cat((self.test_results,outputs),0)       


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg['lr'])
        return optimizer



