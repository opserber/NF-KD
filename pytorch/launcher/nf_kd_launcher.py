import torch
import wandb
from .base_launcher import BaseLauncher

class NF_KD_Launcher(BaseLauncher):
    def build_model(self):
        model = self.model_cls(**self.options['model_options'])
        wandb.watch(model)
        
        print("Initializing Teacher Model...")
        teacher_model = self.model_cls(**self.options['model_options'])
        
        ckpt_path = self.options.get('teacher_ckpt_path')
        if ckpt_path is None:
            raise ValueError("Error: 'teacher_ckpt_path'가 config(options)에 설정되지 않았습니다.")
        
        print(f"Loading Teacher weights from: {ckpt_path}")
        teacher_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        
        self.trainer = self.trainer_cls(
            model=model, 
            teacher_model=teacher_model,
            options=self.options, 
            **self.options['trainer_options']
        )
        
        self.trainer.set_loaders(self.loaders)
        self.trainer.build_optimizer()