"""
    ./sentinel_manager.py

        Contains the definition of the ModelManager, a class 
        made to handle our model and gather useful methods to train & test it.
    
    Author: William MAILLET
"""
import os
import time

import torch
import torch.nn.modules.loss as tloss
import torch.optim as topti
import numpy as np
from sklearn.metrics import f1_score 
from IPython.display import clear_output

from utils import get_progress_bar


class ModelManager(object):
    """
        A special custom object gathering functions 
        to manage the model : training, testing, saving, etc.

        Parameters:
        -----------
        model: torch.nn.Module
            The model to handle
        data_loaders: dict
            The data loaders to use (syntax: {'train': train_loader, 'test': test_loader, 'val': val_loader})
        saves_folder: str
            The folder to save the model weights
    """
    def __init__(self, s_model: torch.nn.Module, data_loaders: dict, saves_folder: str = './checkpoints'):
        self.model = s_model
        self.device = self.model.device
        self.saves_folder = saves_folder
        self.train_loader = data_loaders['train']
        self.test_loader = data_loaders['test']
        self.val_loader = data_loaders['val']

        # Training purposes
        self.hist = {
            'train_loss': [],
            'val_loss': [],
            'n_epochs': 0,
            'batch_size': self.train_loader.batch_size,
            'n_batches': len(self.train_loader),
        }

    def _checkpoint(self, epoch: str):
        """
            Function that saves the model weights

            Parameters:
            -----------
            epoch: int
                The current epoch
        """
        if not os.path.exists(self.saves_folder):
            os.mkdir(self.saves_folder)
        torch.save(self.model.state_dict(), f"{self.saves_folder}/model_{epoch}.pth")

    def _val_step(self, criterion):
        """
            A validation step of the training loop.

            Parameters:
            -----------
            criterion: torch.nn.modules.loss.Loss
                The loss function to use
            
            Returns:
            --------
            loss: float
                The loss of the validation step
        """
        val_loss = 0
        self.model.eval()
        with torch.no_grad():
            for b_idx, data in enumerate(self.val_loader):
                # Move data to GPU
                img, weather, target = data['image'].to(self.device), data['weather'].to(self.device), data['target'].to(self.device)
                
                # Forward pass
                scores = self.model(img, weather)
                loss = criterion(scores, target)
                self.hist['val_loss'].append(loss.item())
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def _trend_calculation(self, values):
        """
            Function that calculates the trend of a list of values.
            This function is used to calculate whether we should stop the training 
            of the model or not : if the validation loss has a trend of increasing we should
            stop the training.

            Parameters:
            -----------
            values: list
                The list of values to calculate the trend of
            
            Returns:
            --------
            str: str
                "up" if the trend is positive, "down" if it is negative
        """
        # Drop NaN values
        values = [v for v in values if not np.isnan(v)]
        coeffs = np.polyfit(list(range(len(values))), values, 1)
        slope = coeffs[-2]
        return "up" if slope > 0 else "down"

    def _log(self, mode: str, args):
        """
            This function is a logger, that is used to display the training status and a loading bar 
            when training, as well as printing when a checkpoint is saved.

            Parameters:
            -----------
            mode: str
                The logging mode (currently only 'train' and 'checkpoint' supported)
            args:
                The arguments to print (a dictioanry)
        """
        # Training log 
        if mode == 'train':
            epoch, b_idx, loss, val_loss, time = args['epoch'], args['b_idx'], args['loss'], args['val_loss'], args['time']
            # Getting progress bar 
            progress_bar = get_progress_bar(b_idx, self.hist['n_batches'])
            # Computing ETA
            eta = (self.hist['n_batches'] - b_idx) * time
            eta_min = int(eta/60)
            eta_sec = int(eta % 60)

            # Printing
            clear_output(wait=True)
            print(f"Epoch {epoch}/{self.hist['n_epochs']} (ETA: {eta_min:02d}:{eta_sec:02d} min.) | {progress_bar} | Loss {loss:.4f} | Val Loss: {val_loss:.4f}")
        # Checkpoint log 
        elif mode == 'checkpoint':
            epoch = args['epoch']
            print(f"[+] Checkpoint saved at epoch {epoch}")
    
    def save_model(self, save_name: str):
        """
            Saves the model state dict to a file 

            Parameters:
            -----------
            save_name: str
                The name of the file to save the model to
        """
        self._checkpoint(save_name)
    
    def load_model(self, path: str):
        """
            Method that loads a model from a file.
        
            Parameters:
            -----------
            path: str
                The path to the file to load the model from

            Returns:
            --------
            model: torch.nn.Module
                The loaded model
        """
        self.model.load_state_dict(torch.load(path))
        return self.model
    
    def train(self, criterion: tloss._Loss, optimizer: topti.Optimizer, lr_scheduler, n_epochs: int = 1, early_stop: bool = False, checkpoint: bool = True):
        """
            Function that trains the network.

            Parameters:
            -----------
            criterion: torch.nn.modules.loss.Loss
                The loss function to use
            optimizer: torch.optim.Optimizer
                The optimizer to use
            lr_scheduler: transformers.optimization.LRScheduler
                The learning rate scheduler to use (Cosine Warmup is recommended)
            n_epochs: int
                The number of epochs to train the model for
            early_stop: bool
                Whether to use early stopping or not
            checkpoint: bool
                Whether to save a checkpoint or not (Recommended)
        """
        # Log purposes
        self.hist['n_epochs'] = n_epochs
        # Pushing model to the device
        self.model.train()
        self.model.to(self.device)
        
        val_losses = []
        for epoch in range(n_epochs):
            for b_idx, data in enumerate(self.train_loader):
                t_start = time.time()
                
                # Resetting gradients
                optimizer.zero_grad()

                # Move data to GPU
                img, weather, target = data['image'].to(self.device), data['weather'].to(self.device), data['target'].to(self.device)

                # Forward pass 
                scores = self.model(img, weather)
                loss = criterion(scores, target)
                self.hist['train_loss'].append(loss.item())

                # Backward pass
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # Print loss 
                t_end = time.time()
                self._log('train', {
                    'epoch': epoch+1,
                    'b_idx': b_idx+1,
                    'loss': loss.item(),
                    'time': t_end - t_start,
                    'val_loss': val_losses[-1] if len(val_losses) > 0 else 0
                })

            # Validation step
            val_loss = self._val_step(criterion)
            val_losses.append(val_loss)
            self.hist['val_loss'].append(val_loss)

            # Detect upward trend in val loss
            if early_stop:
                if epoch > 60:
                    trend = self._trend_calculation(val_losses[-20:])
                else:
                    trend = "down"
            
            # Early stopping
            if early_stop:
                if trend == 'up':
                    print(f"[!] Early stopping triggered at epoch {epoch+1}")
                    # If checkpoint of the model is enabled load the model 5 epochs before the early stopping
                    if checkpoint:
                        self.load_model(f"{self.saves_folder}/model_{epoch-20}.pth")
                    break

            # Saving model
            if checkpoint:
                self._checkpoint(epoch+1)
                self._log('checkpoint', {'epoch': epoch+1})

        return self.hist

    def test(self, criterion: tloss._Loss) -> dict:
        """
            Function that tests the network. Calculates the loss, accuracy
            and the F1-Score of the model.

            Parameters:
            -----------
            criterion: torch.nn.modules.loss.Loss
                The loss function to use
            n_epochs: int
                The number of epochs to train the model
        """
        self.model.eval()
        self.model.to(self.device)
        
        correct = 0
        f1 = 0
        with torch.no_grad():
            for sample in self.test_loader:
                # Move data to GPU
                img, weather, target = sample['image'].to(self.device), sample['weather'].to(self.device), sample['target'].to(self.device)
                scores = self.model(img, weather)
                pred = scores.argmax(dim=1)
                correct += (pred == target.argmax(dim=1)).sum().item()
                f1 += f1_score(target.detach().cpu().numpy().argmax(axis=1), scores.detach().cpu().numpy().argmax(axis=1), labels=[0, 1])
                                
        self.hist['test_loss'] = criterion(scores, target).item()
        self.hist['accuracy'] = correct / len(self.test_loader.dataset)
        self.hist['f1_score'] = f1 / (len(self.test_loader.dataset) / self.test_loader.batch_size)

        return {'test_loss': self.hist['test_loss'], 'accuracy': self.hist['accuracy'], 'f1_score': self.hist['f1_score']}
    
    def get_model_size(self):
        """
            Function that returns the size of the model in MB.

            Returns:
            --------
            size: float
                The size of the model in MB
        """
        param_size, buff_size = 0, 0
        for p in self.model.parameters():
            param_size += p.nelement() * p.element_size()
        for p in self.model.buffers():
            buff_size += p.nelement() * p.element_size()

        return round((param_size + buff_size) / (1024**2), 3)

    def get_model_parameters(self):
        """
            Method that returns the number of parameters
            of the model if they require gradients.

            Returns:
            --------
            n_params: int
                The number of parameters
        """
        n_params = 0
        for p in self.model.parameters():
            if p.requires_grad:
                n_params += p.nelement()
        return n_params