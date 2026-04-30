import torch
import numpy as np
from .sgdr_bit_variable_trainer import SGDR_Bit_Variable_Trainer

class Teacher_Pretrain_Trainer(SGDR_Bit_Variable_Trainer):
    def train_step(self, epoch):
        self.model.set_temperature(self.get_current_temp(epoch))
        step_losses = []
        
        for i, data in enumerate(self.loaders['train']):
            bits = torch.full((data.shape[0],), self.max_bits, dtype=torch.long)
            
            self.model.train()
            data = data.to(self.gpu)
            output = self.model(data, bits)
            
            step_loss = self.loss(output, data).mean()
            step_losses.append(step_loss.detach().cpu().float())
            
            self.optimizer.zero_grad()
            step_loss.backward()
            self.optimizer.step()
            self.scheduler.step(epoch + i / self.iters)
            
        self.eval_metric_history['train_loss'].append(np.mean(step_losses))