import json, sys

sys.path.append("../../")

import numpy as np
import pandas as pd
from mvae import ModifiedAutoEncoder, RMSELoss
from data_management import DataManagement
import torch
import matplotlib.pyplot as plt
df = pd.read_csv('../df_spsp_pred.csv', index_col=0)

df.index = pd.to_datetime(df.index)

window_size = 14
data = df
data = data[~data.index.duplicated(keep='first')]

# Features for the network nputs
features = ['deaths', 'aqi']
# Features target to predict
targets_features = ['deaths']
# Window lenght for the input data
input_window = 7
# Desire prediction output lenght
forward_len = 7
# Percent of train data 0.85 is equal to 85% of the data.
prct_to_train = 0.9

data_instance = DataManagement(data, features, targets_features, normalize_data=False)

#Creating the train and test datasets
data_instance.data_tensor_generate(input_window, forward_len)
# Train and tes split
data_instance.train_test_split(prct_to_train)

# Batch size for train
batch_s = 16
# Creating the data loaders for train and test
data_instance.dataloader_create(batch_s)
data_instance.data_split_by_feature()

learning_rate = 0.001

input_nn = {"inseqlen": input_window,
            "outseqlen": forward_len,
            "growth": 4,
            "latent_space_dim": 7,
            "n_features": len(features),
            "n_targets": len(targets_features),
            "activation": 'ReLU',
            "epochs": 150
            }

seeds = np.genfromtxt("../seeds.csv", delimiter=',')
save_json = {}
for idx, seed in enumerate(seeds):
    # Creating a model instance
    input_nn["seed"] = int(seed)
    model = ModifiedAutoEncoder(input_nn)
    
    # Setting model loss function and optimizer
    model.loss_function = RMSELoss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Epochs is fixed 
    model.train_splitted_freezing_autoencoders_weights(data_instance)
    
    # Creating data to test the model once
    data_instance.data_to_test_create(data)

    # Getting the trained model prediction
    with torch.no_grad():
        dec, pred = model(data_instance.data_to_test)
    # Getting the numpy array by the predicted torch Tensor
    pred_ = pred.view(-1).detach().numpy()
    # Getting the target features values from data
    deaths_values = data[targets_features].values

    # Getting the data test lenght for rmse calc 
    test_index = len(data_instance.X_train)

    # Getting metrics values (rmse of all data, and rmse for only test period)
    metrics = model.metrics_calculate(pred_, deaths_values, test_index)

    save_json["model_"+str(idx)] = {"seed": str(seed), "metrics": metrics}

with open("type#3(DA)_airq-data.json", 'w') as fp:
    json.dump(save_json, fp, indent=3)
