from sklearn.svm import SVR

def train_svr(X, y):
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X, y)
    return model