import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from model.BinaryClassification import BinaryClassification
np.random.seed(42)
torch.manual_seed(42)
#========================= Hyperparameters ======================
scaler = StandardScaler()
train_test_split = 0.8
batch_size = 64
learning_rate = 0.01
num_epochs = 300
num_features = 13
isTrain = True
isInference = True
#========================= Load data ============================
data = pd.read_csv('data/trainData.csv')
features = ['HomePlanet', 'Destination', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'VIP', 'CryoSleep', 'Deck', 'Room', 'Info']
X, Y = data[features], data['Transported']
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
# 检查数据是否有缺失
if torch.isnan(X).any():
    raise ValueError('Data contains NaN values')
#========================= Create DataLoader ============================
dataset = TensorDataset(X, Y)
train_size = int(train_test_split * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#=========================Model, Loss, and Optimizer ====================
model = BinaryClassification(input_dim=num_features)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train_losses = []
val_losses = []
best_val_loss = float('inf')
#========================= Training Loop ============================
if isTrain:
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, Y_batch.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                output = model(X_batch)
                loss = loss_fn(output, Y_batch.view(-1, 1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if isInference:
                torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch + 1}/{num_epochs}, train_loss: {train_loss}, val_loss: {val_loss}')
#========================= Plot Losses ============================
    print("Best Val Loss: ", min(val_losses))
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.legend()
    plt.show()
#========================= Inference ============================
if isInference:
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    test_data = pd.read_csv('data/testData.csv')
    X_test = test_data[features]
    X_test = scaler.transform(X_test)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    predictions = model(X_test).detach().numpy()
    
    probabilities = 1.0 / (1.0 + np.exp(-predictions))
    pred_labels = (probabilities >= 0.5).astype(int)


    passenger_ids = test_data['PassengerId']
    output = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': pred_labels.flatten()})
    output['Transported'] = output['Transported'].map({0: False, 1: True})
    output.to_csv('submissionFeb4.csv', index=False)
