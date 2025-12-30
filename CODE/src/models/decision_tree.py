from sklearn.tree import DecisionTreeRegressor

def train_decision_tree(X, y):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    return model