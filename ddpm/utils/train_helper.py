import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm


class TrainerBase(nn.Module):
    def __init__(self,
                 epochs,
                 train_loader,
                 optimizer,
                 device,
                 **kwargs):
        super().__init__()
        self.epochs = epochs
        if self.epochs is None:
            raise ValueError("请传入训练总迭代次数")
        
        self.train_loader = train_loader
        if self.train_loader is None:
            raise ValueError("请传入train_loader")
        
        self.optimizer = optimizer
        if self.optimizer is None:
            raise ValueError("请传入optimizer")
        
        self.device = device
        if self.device is None:
            raise ValueError("请传入device")
        
    @staticmethod
    def save_best_model(model, path):
        torch.save(model.state_dict(), path + '/' +  'BestModel.pth')
        print("成功将此次训练模型存储(储存格式为.pth)至:" + str(path))
    
    def forward(self, model, *args, **kwargs):
        pass


class SimpleDiffusionTrainer(TrainerBase):
    def __init__(self, 
                 epochs=None, 
                 train_loader=None, 
                 optimizer=None, 
                 device=None, 
                 **kwargs):
        super().__init__(epochs, train_loader, optimizer, device, **kwargs)
        if "timesteps" in kwargs.keys():
            self.timesteps = kwargs["timesteps"]
        else:
            raise ValueError("扩散模型训练必须提供扩散步数参数")

    def forward(self, model,*args, **kwargs):
        for i in range(self.epochs):
            losses = []
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for step, (features, labels) in loop:
                features = features.to(self.device)
                batch_size = features.shape[0]

                t = torch.randint(0, self.timesteps, (batch_size,),device = self.device).long()

                loss = model(mode="train", x_start=features,t =t, loss_type='huber')
                losses.append(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                loop.set_description(f"Epoch [{i}/{self.epochs}]")
                loop.set_postfix(loss=loss.item())

        if "model_save_path" in kwargs.keys():
            self.save_best_model(model = model,path=kwargs["model_save_path"])
            
        return model