from __future__ import annotations
from typing import Any, Tuple

import torch
import random
import numpy as np
import logging
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from torchvision import models

from automl.datasets import FeatureDataset, process_dataset

from automl.models import DARTS_Network, DARTS_Best
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AutoML:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._model: nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, dataset_class: Any) -> AutoML:
        """Fit function for the AutoML class using Bayesian Optimization."""
        
        # Set seed for pytorch training
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self._transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(30),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])
        ])

        batch_size = 64

        dataset = dataset_class(root="./data", split='train', download=True, transform=self._transform, rgb=True)
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        valid_dataset = dataset_class(root="./data", split='valid', download=True, transform=self._transform, rgb=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        ## FINETUNING WITH HYPERPARAMETER OPTIMIZATION

        # Define the search space for Bayesian Optimization
        space = [
            Real(1e-6, 1e-1, name='learning_rate', prior='log-uniform'),
            Real(0.1, 0.9, name='momentum'),
            Real(0, 1e-2, name='weight_decay', prior='uniform'),
            Categorical(['adam', 'sgd', 'rmsprop'], name='optimizer_type'),
            Integer(2, 4, name='cutoff_layer')
        ]

        @use_named_args(space)
        def objective(**params):
            learning_rate = params['learning_rate']
            weight_decay = params['weight_decay']
            momentum = params.get('momentum', 0.9)
            cutoff_layer = params.get('cutoff_layer', 4)
            optimizer_type = params.get('optimizer_type', 'adam')

            pretrained_model = models.resnet34(weights='DEFAULT')

            for param in pretrained_model.parameters():
                param.requires_grad = False
            
            if cutoff_layer == 2:
                for param in pretrained_model.layer2.parameters():
                    param.requires_grad = True
                for param in pretrained_model.layer3.parameters():
                    param.requires_grad = True
                for param in pretrained_model.layer4.parameters():
                    param.requires_grad = True
            elif cutoff_layer == 3:
                for param in pretrained_model.layer3.parameters():
                    param.requires_grad = True
                for param in pretrained_model.layer4.parameters():
                    param.requires_grad = True
            elif cutoff_layer == 4:
                for param in pretrained_model.layer4.parameters():
                    param.requires_grad = True
        
            in_features = pretrained_model.fc.in_features

            pretrained_model.fc = nn.Linear(in_features, dataset_class.num_classes)

            if optimizer_type == 'adam':
                optimizer = optim.Adam(pretrained_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_type == 'sgd':
                optimizer = optim.SGD(pretrained_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            elif optimizer_type == 'rmsprop':
                optimizer = optim.RMSprop(pretrained_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            best_val_loss, _ = self._train_and_validate(model=pretrained_model,
                                                    criterion=nn.CrossEntropyLoss(),
                                                    optimizer=optimizer,
                                                    train_loader=self.train_loader,
                                                    valid_loader=self.valid_loader,
                                                    num_epochs=1)
            
            return best_val_loss
        

        def final_training(learning_rate, weight_decay, momentum, optimizer_type, cutoff_layer):

            pretrained_model = models.resnet34(weights='DEFAULT')

            for param in pretrained_model.parameters():
                param.requires_grad = False
            
            if cutoff_layer == 2:
                for param in pretrained_model.layer2.parameters():
                    param.requires_grad = True
                for param in pretrained_model.layer3.parameters():
                    param.requires_grad = True
                for param in pretrained_model.layer4.parameters():
                    param.requires_grad = True
            elif cutoff_layer == 3:
                for param in pretrained_model.layer3.parameters():
                    param.requires_grad = True
                for param in pretrained_model.layer4.parameters():
                    param.requires_grad = True
            elif cutoff_layer == 4:
                for param in pretrained_model.layer4.parameters():
                    param.requires_grad = True
        
            in_features = pretrained_model.fc.in_features

            pretrained_model.fc = nn.Linear(in_features, dataset_class.num_classes)

            if optimizer_type == 'adam':
                optimizer = optim.Adam(pretrained_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_type == 'sgd':
                optimizer = optim.SGD(pretrained_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            elif optimizer_type == 'rmsprop':
                optimizer = optim.RMSprop(pretrained_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            best_val_loss, best_pretrained_model = self._train_and_validate(model=pretrained_model,
                                                    criterion=nn.CrossEntropyLoss(),
                                                    optimizer=optimizer,
                                                    train_loader=self.train_loader,
                                                    valid_loader=self.valid_loader)
        
            return best_pretrained_model 


        # load best parameters if there
        res = gp_minimize(objective, space, x0=[1e-2, 0.9, 0, 'sgd', 4], n_calls=30, random_state=self.seed)
        best_params = res.x
        best_params[-1] = int(best_params[-1])

        ## BEST MODEL TRAINING
        best_params = dict(zip(['learning_rate', 'momentum', 'weight_decay', 'optimizer_type', 'cutoff_layer'], best_params))

        pretrained_model = final_training(**best_params)
        in_features = pretrained_model.fc.in_features

        ## BUILD NEW DATASET FROM MODEL OUTPUT
        finetuned_without_fc = nn.Sequential(*list(pretrained_model.children())[:-1])

        train_features, train_labels = process_dataset(finetuned_without_fc, self.train_loader, self.device)
        valid_features, valid_labels = process_dataset(finetuned_without_fc, self.valid_loader, self.device)

        train_feature_dataset = FeatureDataset(train_features, train_labels)
        valid_feature_dataset = FeatureDataset(valid_features, valid_labels)
        
        self.train_loader = DataLoader(train_feature_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_feature_dataset, batch_size=batch_size, shuffle=False)

        ## ARCHITECTURE SEARCH
        darts_model = DARTS_Network(in_features=in_features, num_classes=dataset_class.num_classes)

        def extract_best_architecture(alphas):
            best_arch = []
            for alpha in alphas:
                best_op_index = alpha.argmax().item()
                best_arch.append(best_op_index)
            return best_arch

        darts_model = darts_model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(darts_model.parameters(), lr=3e-4)
        alpha_optimizer = optim.Adam(darts_model.alphas, lr=3e-4)

        self._train_and_validate(model=darts_model, 
                                            criterion=criterion, 
                                            optimizer=optimizer, 
                                            train_loader=self.train_loader, 
                                            alpha_optimizer=alpha_optimizer, 
                                            valid_loader=self.valid_loader)

        alphas = darts_model.alphas
        best_arch = extract_best_architecture(alphas)

        ## HYPERPARAMETER OPTIMIZATION

        # Define the search space for Bayesian Optimization
        space = [
            Real(1e-6, 1e-2, name='learning_rate', prior='log-uniform'),
            Real(0.1, 0.9, name='momentum'),
            Real(1e-6, 1e-2, name='weight_decay', prior='log-uniform'),
            Categorical(['adam', 'sgd', 'rmsprop'], name='optimizer_type'),
        ]

        @use_named_args(space)
        def objective(**params):
            learning_rate = params['learning_rate']
            weight_decay = params['weight_decay']
            momentum = params.get('momentum', 0.9)
            optimizer_type = params.get('optimizer_type', 'adam')

            best_mlp = DARTS_Best(model=darts_model, best_arch=best_arch)

            if optimizer_type == 'adam':
                optimizer = optim.Adam(best_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_type == 'sgd':
                optimizer = optim.SGD(best_mlp.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            elif optimizer_type == 'rmsprop':
                optimizer = optim.RMSprop(best_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)

            best_val_loss, best_mlp = self._train_and_validate(model=best_mlp,
                                                    criterion=nn.CrossEntropyLoss(),
                                                    optimizer=optimizer,
                                                    train_loader=self.train_loader,
                                                    valid_loader=self.valid_loader)
            
            return best_val_loss
        

        def final_training(learning_rate, weight_decay, momentum, optimizer_type):

            best_mlp = DARTS_Best(model=darts_model, best_arch=best_arch)

            if optimizer_type == 'adam':
                optimizer = optim.Adam(best_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_type == 'sgd':
                optimizer = optim.SGD(best_mlp.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            elif optimizer_type == 'rmsprop':
                optimizer = optim.RMSprop(best_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)

            best_val_loss, best_mlp = self._train_and_validate(model=best_mlp,
                                                    criterion=nn.CrossEntropyLoss(),
                                                    optimizer=optimizer,
                                                    train_loader=self.train_loader,
                                                    valid_loader=self.valid_loader)
        
            return best_mlp 


        # load best parameters if there
        res = gp_minimize(objective, space, n_calls=30, random_state=self.seed)
        best_params = res.x

        ## BEST MODEL TRAINING
        best_params = dict(zip(['learning_rate', 'momentum', 'weight_decay', 'optimizer_type'], best_params))
        best_mlp = final_training(**best_params)

        pretrained_model.fc = best_mlp
        self._model = pretrained_model

    def _train_and_validate(self, model, 
                        criterion, 
                        optimizer, 
                        train_loader, 
                        alpha_optimizer=None, 
                        valid_loader=None, 
                        num_epochs=8):
        
        model.to(self.device)
        best_val_loss = float('inf')
        best_model_weights = None
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if alpha_optimizer is not None:
                    # alpha optimization
                    inputs_valid, labels_valid = next(iter(valid_loader))
                    inputs_valid, labels_valid = inputs_valid.to(self.device), labels_valid.to(self.device)
                    alpha_optimizer.zero_grad()
                    outputs = model(inputs_valid)
                    loss = criterion(outputs, labels_valid)
                    loss.backward()
                    alpha_optimizer.step()

                # weight optimization
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            if valid_loader is None:
                continue

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in tqdm(valid_loader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(valid_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.state_dict()

        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
        return best_val_loss, model

    def predict(self, dataset_class) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction function for the AutoML class."""
        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                    [0.229, 0.224, 0.225])
        ])

        dataset = dataset_class(root="./data", split='test', download=True, transform=transform, rgb=True)
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        predictions = []
        labels = []
        self._model.eval()
        with torch.no_grad():
            for data, target in tqdm(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self._model(data)
                predicted = torch.argmax(output, 1)
                labels.append(target.cpu().numpy())
                predictions.append(predicted.cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        return predictions, labels
