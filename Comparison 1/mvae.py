import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

class ModifiedAutoEncoder(nn.Module):
    def __init__(self, atribucts_dict):
        super().__init__()
        
        for item, value in atribucts_dict.items():
            setattr(self, item, value)

        self.autoencoders_counter = 0

        if self.seed:
            self.fix_seed(self.seed)

        self.encoders, self.decoders, self.predictors = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.generate_autoencoders()
        self.generate_predictors()

    def fix_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def generate_autoencoders(self):
        for _ in range(self.n_features):
            self.add_autoencoder()
    
    def add_autoencoder(self):
        self.add_encoder()
        self.add_decoder()
        self.autoencoders_counter = self.autoencoders_counter + 1
        if self.is_an_appended_autoencoder():
            self.att_predictors()

    def is_an_appended_autoencoder(self):
        if self.n_features < self.autoencoders_counter:
            self.n_features = self.autoencoders_counter
            return True
        else:
            return False 

    def add_encoder(self):
        self.encoders.append(nn.Sequential(nn.Linear(self.inseqlen, self.growth * self.inseqlen),
                                getattr(nn, self.activation)(),
                                nn.Linear(self.growth * self.inseqlen, self.latent_space_dim)))

    def add_decoder(self):
        self.decoders.append(nn.Sequential(nn.Linear(self.latent_space_dim, self.growth * self.inseqlen),
                                getattr(nn, self.activation)(),
                                nn.Linear(self.growth * self.inseqlen, self.inseqlen)))

    def generate_predictors(self):
        for _ in range(self.n_targets):
            self.add_predictor()
    
    def add_predictor(self):
        self.predictors.append(nn.Sequential(nn.Linear(self.n_features * self.latent_space_dim, self.outseqlen)))

    def att_predictors(self):
        self.predictors = nn.ModuleList([nn.Sequential(nn.Linear(self.n_features * self.latent_space_dim, self.outseqlen)) for _ in range(self.n_targets)])

    def forward(self, batch):
        encoded_batch, decoded_batch = self.forward_autoencoders(batch)
        predict_batch = self.forward_predictors(encoded_batch)

        return decoded_batch, predict_batch
    
    def forward_autoencoders(self, batch):
        encoded_batch, decoded_batch = [], []
        for input in batch:
            encoded_batch.append([encoder(xs) for encoder, xs in zip(self.encoders, input)])
        # stack is used to transform a list of tensor in a unic tensor of tensor
        for enc in encoded_batch:
            decoded_batch.append(torch.stack([decoder(z) for decoder, z in zip(self.decoders, enc)]))
        return encoded_batch, torch.stack(decoded_batch)
        
    def forward_predictors(self, encoded_batch):
        predict_batch = []
        # stack is used to transform a list of tensor in a unic tensor of tensor
        for enc in encoded_batch:
            predict_batch.append(torch.stack([predictor(torch.cat(enc, dim=-1)) for predictor in self.predictors]))
        return torch.stack(predict_batch)

    def train_joined(self,
                     data_instance: DataManagement,
                     validation: bool = True
        ) -> None:

        self.data_train = data_instance.data_train
        self.validation = validation
        if self.validation:
            self.data_validation = data_instance.data_test
        
        epochs = range(self.epochs)

        self.loss_train, self.loss_val = [], []
        start = time.time()
        for epoch in epochs:
            for batch_train in self.data_train:
                inputs, targets = batch_train
                forward_output = self.forward(inputs)

                loss = self.joined_loss_calculation(batch_train, forward_output)
                self.weights_adjustment(loss)

            self.loss_train.append(loss.item())

            if self.validation:
                loss_validation = self.train_validation_joined()
                self.loss_val.append(loss_validation.item())

        end = time.time()
        self.elapsed_time = end - start
    
    def train_splitted_freezing_autoencoders_weights(self,
                                                     data_instance: DataManagement,
                                                     validation: bool = True):

        self.set_encoders_weights_unadjustable()
        self.train_splitted_freezing_decoders_weights(data_instance, validation)
        
    def train_splitted_freezing_decoders_weights(self,
                                                 data_instance: DataManagement,
                                                 validation: bool = True):
        self.data_train = data_instance.data_train
        if validation:
            self.data_validation = data_instance.data_test
        self.splitted_feature_data = data_instance.splited_data
        self.validation = validation

        epochs = range(self.epochs)
        start = time.time()

        self.train_all_autoencoders()
        self.set_decoders_weights_unadjustable()

        self.loss_train, self.loss_val = [], []
        for epoch in epochs:
            for batch in self.data_train:
                inputs, targets = batch
                forward_output = self.forward(inputs)

                loss = self.splitted_loss_calculation(batch, forward_output)
                self.weights_adjustment(loss)

            self.loss_train.append(loss.item())

            if self.validation:
                loss_validation = self.train_validation_splitted()
                self.loss_val.append(loss_validation.item())

        end = time.time()
        self.elapsed_time = end - start
    
    def train_additive(self, new_feature):
        autoencoder = [self.encoders[-1], self.decoders[-1]]
        self.train_autoencoder(new_feature, autoencoder)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.train_predictor()

    def train_all_autoencoders(self):
        for idx, feature in enumerate(self.splitted_feature_data):
            autoencoder = [self.encoders[idx], self.decoders[idx]]
            self.train_autoencoder(feature, autoencoder)

    def train_predictor(self):
        epochs = range(self.epochs)
        
        start = time.time()
        self.loss_train, self.loss_val = [], []
        for epoch in epochs:
            for batch in self.data_train:
                inputs, targets = batch
                forward_output = self.forward(inputs)

                loss = self.splitted_loss_calculation(batch, forward_output)
                self.weights_adjustment(loss)

            self.loss_train.append(loss.item())
        end = time.time()
        self.elapsed_time = end - start

    def train_autoencoder(self, feature, autoencoder):
        inner_optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        encoder = autoencoder[0]
        decoder = autoencoder[1]
        for epoch in range(self.epochs):
            for batch in feature:
                inputs, targets = batch
                encoded = encoder(inputs)
                decoded = decoder(encoded)

                loss = self.loss_function(decoded, targets)

                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()

    def train_validation_joined(self):
        with torch.no_grad():
            for batch_validation in self.data_validation:
                inputs_validation, targets_validation = batch_validation

                forward_output = self.forward(inputs_validation)
                
                joined_loss_validation = self.joined_loss_validation_calculation(batch_validation, forward_output)                
        return joined_loss_validation
    
    def train_validation_splitted(self):
        with torch.no_grad():
            for batch_validation in self.data_validation:
                inputs_validation, targets_validation = batch_validation

                forward_output = self.forward(inputs_validation)

                splitted_loss_validation = self.splitted_loss_validation_calculation(batch_validation, forward_output)
        return splitted_loss_validation

    def weights_adjustment(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def joined_loss_calculation(self, batch_validation, forward_output):
        inputs, targets = batch_validation
        decoded, predict = forward_output
        autoencoder_loss = self.loss_function(decoded, inputs)
        predicted_loss = self.loss_function(predict, targets)
        return autoencoder_loss + predicted_loss

    def splitted_loss_calculation(self, batch_validation, forward_output):
        inputs, targets = batch_validation
        decoded, predict = forward_output
        return self.loss_function(predict, targets)

    def joined_loss_validation_calculation(self, batch_validation, forward_output):
        with torch.no_grad():
            inputs, targets = batch_validation
            decoded, predict = forward_output
            autoencoder_loss = self.loss_function(decoded, inputs)
            predicted_loss = self.loss_function(predict, targets)
            return autoencoder_loss + predicted_loss

    def splitted_loss_validation_calculation(self, batch_validation, forward_output):
        with torch.no_grad():
            inputs, targets = batch_validation
            decoded, predict = forward_output
            return self.loss_function(predict, targets)
    
    def set_decoders_weights_unadjustable(self):
        for param in self.decoders.parameters():
            param.requires_grad = False

    def set_encoders_weights_unadjustable(self):
        for param in self.encoders.parameters():
            param.requires_grad = False

    def set_autoencoders_weights_unadjustable(self):
        self.set_decoders_weights_unadjustable()
        self.set_encoders_weights_unadjustable()

    def plot_loss(self):
        plt.title("Loss for train in " + str(self.elapsed_time) + " seconds")
        plt.plot(self.loss_train, label='loss train')
        if self.loss_val:
            plt.plot(self.loss_val, label='loss val')
        plt.legend(loc='best')
        plt.show()

    def metrics_calculate(self, pred, original, test_index):
        
        original_entire = original[:len(pred)]
        pred_entire = pred
        #Only train
        original_train = original_entire[:test_index]
        pred_train = pred_entire[:test_index]
        # Only test part
        original_test = original_entire[test_index:]
        pred_test = pred_entire[test_index:]

        # RMSE (root mean squared error) entire signal and only test part
        rmse_entire = np.sqrt(mean_squared_error(original_entire, pred_entire))
        rmse_train = np.sqrt(mean_squared_error(original_train, pred_train))
        rmse_test = np.sqrt(mean_squared_error(original_test, pred_test))
        # MAE (mean absolut error) entire signal and only test part
        mae_entire = mean_absolute_error(original_entire, pred_entire)
        mae_train = mean_absolute_error(original_train, pred_train)
        mae_test = mean_absolute_error(original_test, pred_test)
        # MEAE (median absolut error) entire signal and only test part
        meae_entire = median_absolute_error(original_entire, pred_entire)
        meae_train = median_absolute_error(original_train, pred_train)
        meae_test = median_absolute_error(original_test, pred_test)
        # MAPE (mean absolute percentage error) entire signal and only test part
        mape_entire = mean_absolute_percentage_error(original_entire, pred_entire)
        mape_train = mean_absolute_percentage_error(original_train, pred_train)
        mape_test = mean_absolute_percentage_error(original_test, pred_test)

        # Metrics dict        
        metrics_dict = {"rmse_entire": rmse_entire, "rmse_train": rmse_train, "rmse_test": rmse_test,
                        "mae_entire": mae_entire, "mae_train": mae_train, "mae_test": mae_test,
                        "meae_entire": meae_entire, "meae_train": meae_train, "meae_test": meae_test,
                        "mape_entire": mape_entire, "mape_train": mape_train, "mape_test": mape_test  
                    }
        return metrics_dict
