from matplotlib import pyplot as plt
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
import torch
import torch.nn as nn
import torch.optim as optim



# The happiness rankings depend only on the respondents’ average Cantril ladder scores, 
# not on the values of the six variables (Explained by: GDP per capita,Explained by: Social support,
# Explained by: Healthy life expectancy,Explained by: Freedom to make life choices,Explained by: Generosity,
# Explained by: Perceptions of corruption) that we use to help account for the large differences we find. 

def load_data(dataloc):
    data = pd.read_csv(dataloc)
    return data

def format_train_data1(data):
    dropped_data = data.drop(columns=['Regional indicator', 'upperwhisker', 'lowerwhisker', 
                                      'Standard error of ladder score', 'Ladder score in Dystopia', 
                                      'Dystopia + residual', 'Logged GDP per capita', 
                                      'Social support', 'Healthy life expectancy', 
                                      'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'])
    renamed_data = dropped_data.rename(columns={
        'Country name': 'Country',
        'Ladder score': 'Happiness',
        'Explained by: Log GDP per capita': 'GDP per capita',
        'Explained by: Social support': 'Social support',
        'Explained by: Healthy life expectancy': 'Healthy life expectancy',
        'Explained by: Freedom to make life choices': 'Freedom to make choices',
        'Explained by: Generosity': 'Generosity',
        'Explained by: Perceptions of corruption': 'Perception of corruption'
    })
    return renamed_data

def format_train_data2(data):
    dropped_data = data.drop(columns=['RANK', 'Whisker-high', 'Whisker-low', 'Dystopia (1.83) + residual'])
    renamed_data = dropped_data.rename(columns={
        'Happiness score': 'Happiness',
        'Explained by: GDP per capita': 'GDP per capita',
        'Explained by: Social support': 'Social support',
        'Explained by: Healthy life expectancy': 'Healthy life expectancy',
        'Explained by: Freedom to make life choices': 'Freedom to make choices',
        'Explained by: Generosity': 'Generosity',
        'Explained by: Perceptions of corruption': 'Perception of corruption'
    })
    return renamed_data

def clean_data(data):
    clean_data = data.dropna(axis='columns')
    return clean_data


def extract_scale_features(data1, data2, feat_eng, conv_to_array):
    # Different features have different scales, which means different features
    # with different magnitudes could dominate the model's learning process,
    # so apply z-score scaling.

    # The standard score of a sample x is calculated as: z = (x - u) / s,
    # where u is the mean of the training samples, and s is the standard 
    # deviation of the training samples.

    # Scaling for "Happiness" is not needed because its own Cantil scale (1-10) is valid.

    data = pd.concat([data1, data2], ignore_index=True)
    print(data.shape)
    #data = data1
    scaler = StandardScaler()

    # Feature Engineering: Remove the two least important features
    # Removed GDP per capita and Generosity

    if feat_eng:
        feature_col = ['Social support', 'Healthy life expectancy',
                    'Freedom to make choices', 'Perception of corruption']
    else:
        feature_col = ['GDP per capita', 'Social support', 'Healthy life expectancy',
                    'Freedom to make choices', 'Generosity', 'Perception of corruption']
        
    if conv_to_array:
        data = data.values
        X = scaler.fit_transform(data[:,2:])
        Y = data[:,1]
    else:
        X = scaler.fit_transform(data[feature_col])
        Y = data['Happiness']

    return X, Y, feature_col

def fit_data(X, Y, feat_col):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    if len(feat_col) < 6:
        print("---------------------------------------------------------------------------------------")
        print("Feature Engieered Data: Removed GDP per capita and Generosity!")
        print("---------------------------------------------------------------------------------------\n")

    ### Basic SVR
    reg = SVR(kernel='rbf')
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)

    # plt.figure(figsize=(8, 6))
    # plt.scatter(y_test, y_pred)
    # plt.xlabel("Actual Happiness")
    # plt.ylabel("Predicted Happiness")
    # plt.title("Actual vs. Predicted Happiness")
    # plt.show()

    r2score = r2_score(y_test, y_pred) 
    mse = mean_squared_error(y_test, y_pred)
    y_train_pred = reg.predict(x_train)
    train_mse = mean_squared_error(y_train, y_train_pred)

    print("Basic SVR")
    print("----------------------")
    print(f"R2 Score: {r2score}")
    print(f"Training MSE: {train_mse}")
    print(f"Testing MSE: {mse}\n")
    
    ### Hyperparameter Tuning SVR

    param_dist = {
        'C': [0.1, 1, 10, 100],                 # Regularization
        'kernel': ['rbf'],                      # Kernel
        'epsilon': [0.01, 0.1, 0.2, 0.5],       # Margin Width
        'gamma': ['scale', 'auto', 0.1, 1, 10], # Boundary complexity
    }

    svr = SVR()
    rand_search = RandomizedSearchCV(svr, param_distributions=param_dist, n_iter=10, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, random_state=42)
    rand_search.fit(x_train, y_train)
    best_params = rand_search.best_params_
    best_svr = rand_search.best_estimator_
    y_pred = best_svr.predict(x_test)
    result = permutation_importance(best_svr, x_train, y_train, n_repeats=10, random_state=42)
    feat_imp = result.importances_mean
    feat_imp_dict = dict(zip(feat_col, feat_imp))

    y_train_pred = best_svr.predict(x_train)
    train_mse = mean_squared_error(y_train, y_train_pred)


    r2score = r2_score(y_test, y_pred) 
    mse = mean_squared_error(y_test, y_pred)


    print("Hyperparameter Tuned SVR")
    print("----------------------")
    print(f"Best Params: {best_params}")
    print(f"Important Features: {feat_imp_dict}")
    print(f"R2 Score: {r2score}")
    print(f"Training MSE: {train_mse}")
    print(f"Testing MSE: {mse}\n")
    

    # plt.figure(figsize=(8, 6))
    # plt.scatter(y_test, y_pred)
    # plt.xlabel("Actual Happiness")
    # plt.ylabel("Predicted Happiness")
    # plt.title("Actual vs. Predicted Happiness")
    # plt.show()

    ### Ensemble Learning 1: Random Forest Regressor
    for_reg = RandomForestRegressor(random_state=42)
    for_reg.fit(x_train, y_train)
    y_pred = for_reg.predict(x_test)

    y_train_pred = for_reg.predict(x_train)
    train_mse = mean_squared_error(y_train, y_train_pred)

    r2score = r2_score(y_test, y_pred) 
    mse = mean_squared_error(y_test, y_pred)
    print("Random Forest Regressor")
    print("----------------------")
    print(f"R2 Score: {r2score}")
    print(f"Training MSE: {train_mse}")
    print(f"Testing MSE: {mse}\n")
    

    ### Ensemble Learning 1.1: Random Forest Regressor with Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200, 300], # Number of trees
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }


    for_reg = RandomForestRegressor()
    rand_search = RandomizedSearchCV(for_reg, param_distributions=param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, random_state=42)
    rand_search.fit(x_train, y_train)
    best_params = rand_search.best_params_
    best_rfr = rand_search.best_estimator_
    feat_imp = best_rfr.feature_importances_
    feat_imp_dict = dict(zip(feat_col, feat_imp))
    y_pred = best_rfr.predict(x_test)

    y_train_pred = best_rfr.predict(x_train)
    train_mse = mean_squared_error(y_train, y_train_pred) 


    r2score = r2_score(y_test, y_pred) 
    mse = mean_squared_error(y_test, y_pred)
    print("Hyperparameter Tuned RFR")
    print("----------------------")
    print(f"Best Params: {best_params}")
    print(f"Important Features: {feat_imp_dict}")
    print(f"R2 Score: {r2score}")
    print(f"Training MSE: {train_mse}")
    print(f"Testing MSE: {mse}\n")

    ### Ensemble Learning 2: Gradient Boosting Regressor
    grad_reg = GradientBoostingRegressor(random_state=42)
    grad_reg.fit(x_train, y_train)
    y_pred = grad_reg.predict(x_test)

    y_train_pred = grad_reg.predict(x_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    
    r2score = r2_score(y_test, y_pred) 
    mse = mean_squared_error(y_test, y_pred)
    print("Gradient Boosting Regressor")
    print("----------------------")
    print(f"R2 Score: {r2score}")
    print(f"Training MSE: {train_mse}")
    print(f"Testing MSE: {mse}\n")

    ### Ensemble Learning 2.1: Gradient Boosting Regressor with Hyperparameter Tuning

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grad_reg = GradientBoostingRegressor()
    rand_search = RandomizedSearchCV(grad_reg, param_distributions=param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, random_state=42)
    rand_search.fit(x_train, y_train)
    best_params = rand_search.best_params_
    best_gbr = rand_search.best_estimator_
    feat_imp = best_gbr.feature_importances_
    feat_imp_dict = dict(zip(feat_col, feat_imp))
    y_pred = best_gbr.predict(x_test)

    y_train_pred = best_gbr.predict(x_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
   

    r2score = r2_score(y_test, y_pred) 
    mse = mean_squared_error(y_test, y_pred)
    print("Hyperparameter Tuned GBR")
    print("----------------------")
    print(f"Best Params: {best_params}")
    print(f"Important Features: {feat_imp_dict}")
    print(f"R2 Score: {r2score}")
    print(f"Training MSE: {train_mse}")
    print(f"Testing MSE: {mse}\n")

def nn_fit_data(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    x_train_tensor = torch.Tensor(x_train)
    y_train = y_train.astype('float32')
    y_train_tensor = torch.Tensor(y_train)
    y_train_tensor = y_train_tensor.view(-1, 1)
    x_test_tensor = torch.Tensor(x_test)
    y_test = y_test.astype('float32')
    y_test_tensor = torch.Tensor(y_test)
    y_test_tensor = y_test_tensor.view(-1, 1)

    class HappinessPredictor(nn.Module):
        def __init__(self, input_size):
            super(HappinessPredictor, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    input_size = x_train_tensor.shape[1]
    model = HappinessPredictor(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    model.eval()
    with torch.no_grad():
        predictions = model(x_test_tensor)
        test_loss = criterion(predictions, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')

def graph_MSER2():
    features = ['GDP per capita', 'Social support', 'Healthy life \nexpectancy', 'Freedom to \nmake choices', 'Generosity', 'Perception \nof corruption']
    feature_importance_svr = [0.0406, 0.3356, 0.3049, 0.2065, 0.0524, 0.0799]
    feature_importance_rfr = [0.0318, 0.4221, 0.3482, 0.0943, 0.0419, 0.0618]
    feature_importance_gbr = [0.0199, 0.5138, 0.2662, 0.1042, 0.0363, 0.0596]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(features, feature_importance_svr, color='blue', label='HT SVR', s=100, alpha=0.7)
    plt.scatter(features, feature_importance_rfr, color='green', label='HT Forest', s=100, alpha=0.7)
    plt.scatter(features, feature_importance_gbr, color='orange', label='HT Boosting', s=100, alpha=0.7)

    plt.title('Feature Importance Model Comparison')
    plt.xlabel('Features')
    plt.ylabel('Feature Importance')
    plt.legend()
    plt.savefig('feat_imp_fig.png')
    plt.show()

def graph_mses():
    models = ['SVR', 'HT SVR', 'Random \nForest', 'HT Random \nForest', 'Gradient \nBoost', 'HT Gradient \nBoost']
    train_mses = [0.2045, 0.2363, 0.0406, 0.1140, 0.0468, 0.000000250]
    test_mses = [0.2376, 0.2312, 0.2376, 0.2514, 0.2850, 0.3233]

    bar_width = 0.35
    index = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(index, train_mses, width=bar_width, label='Training MSE', color='blue')
    ax.bar(index + bar_width, test_mses, width=bar_width, label='Testing MSE', color='red')
    ax.set_xlabel('Models')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title('Training and Testing MSE for Different Models')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(models)
    ax.legend()
    plt.savefig('comp_train_test.png')
    plt.show()


    