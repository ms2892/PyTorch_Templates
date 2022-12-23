# 1) Design Model (input,output size, forward pass)
# 2) Construct Loss and Optimizer
# 3) Training Loop
#     - forward pass: Compute the prediction
#     - backward pass: Computer the gradients
#     - update weights: using something like gradient descent etc


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



class LogisticRegression(nn.Module):
    
    def __init__(self,input_dim):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(input_dim,1)
        
    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

if __name__=='__main__':

    # 0) Prepare the dataset
    bc = datasets.load_breast_cancer()
    X,y = bc.data,bc.target

    n_samples,n_features = X.shape
    print(n_samples,n_features)

    # Split the dataset in the 80:20 ratio
    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2, random_state=1234)

    # Scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    Y_train = torch.from_numpy(Y_train.astype(np.float32))
    Y_test = torch.from_numpy(Y_test.astype(np.float32))

    Y_train = Y_train.view(Y_train.shape[0],1)
    Y_test = Y_test.view(Y_test.shape[0],1)

    # 1) Create The Model
    # f = wx + b, sigmoid at the end


    model = LogisticRegression(n_features)

    # 2) Loss and Optimizer
    learning_rate = 0.01

    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


    # 3) Training Loop
    num_epochs = 100

    for epoch in range(num_epochs):
        # Switch the model to Training Mode
        model.train()
        # forward Pass
        y_pred = model(X_train)
        loss = criterion(y_pred,Y_train)    
        
        # Backward Pass
        loss.backward()
        
        # Updates
        optimizer.step()
        
        optimizer.zero_grad()
        
        if (epoch+1)%10==0:
            # Switch to Evaluation Mode
            model.eval()
            y_test_pred = model(X_test)
            loss_test = criterion(y_test_pred,Y_test)
            
            y_pred = y_pred.round()
            tr_acc = y_pred.eq(Y_train).sum() / float(Y_train.shape[0])
            
            y_test_pred = y_test_pred.round()
            acc = y_test_pred.eq(Y_test).sum() / float(Y_test.shape[0])
            print(f'Epoch {epoch+1}: Accuracy = {tr_acc:.4f}, Loss = {loss:.4f}, Val Accuracy= {acc.item():.4f} Test Loss = {loss_test:.4f}')

    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = y_pred.round()
        acc = y_pred.eq(Y_test).sum() / float(Y_test.shape[0])
        
        print(f'Accuracy = {acc:.4f}')