import torch
import pandas as pd
from model.BinaryClassification import BinaryClassification

num_features = 13
model = BinaryClassification(input_dim=num_features)
model.load_state_dict(torch.load('best_model.pth'))

model.eval()  # 切换到推断模式
#========================= Inference ============================
test_data = pd.read_csv('data/testData.csv')
features = ['HomePlanet', 'Destination', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'VIP', 'CryoSleep', 'Deck', 'Room', 'Info']
X_test = test_data[features]
print(X_test)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# predictions = model(X_test).detach().numpy()
# print(predictions)
