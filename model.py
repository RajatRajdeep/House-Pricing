import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('home-data-for-ml-course/train.csv', index_col='Id')
test = pd.read_csv('home-data-for-ml-course/test.csv', index_col='Id')

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
y = train.SalePrice
X = train[features].copy()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

model = RandomForestRegressor(random_state=0)
model.fit(X_train, y_train)
pred = model.predict(X_valid)
print(mean_absolute_error(y_valid, pred))