"""
=================
Linear Regression
=================

:epkg:`scikit-learn` and :epkg:`torch` to train a linear regression.

data
====
"""

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch

X, y = make_regression(1000, n_features=5, noise=10.0, n_informative=2)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y)

################################
# scikit-learn: the simple regression
# ===================================
#
# .. math::
#
#       A^* = (X'X)^{-1}X'Y


clr = LinearRegression()
clr.fit(X_train, y_train)

print(f"coefficients: {clr.coef_}, {clr.intercept_}")

#############################
# Evaluation
# ==========

y_pred = clr.predict(X_test)
l2 = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"LinearRegression: l2={l2}, r2={r2}")

################################
# scikit-learn: SGD algorithm
# ===================================
#
# SGD = Stochastic Gradient Descent

clr = SGDRegressor(max_iter=5, verbose=1)
clr.fit(X_train, y_train)

print(f"coefficients: {clr.coef_}, {clr.intercept_}")

#############################
# Evaluation

y_pred = clr.predict(X_test)
sl2 = mean_squared_error(y_test, y_pred)
sr2 = r2_score(y_test, y_pred)
print(f"SGDRegressor: sl2={sl2}, sr2={sr2}")



###############################
# torch
# =====


class TorchLinearRegression(torch.nn.Module):
    def __init__(self, n_dims: int, n_targets: int):
        super(TorchLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(n_dims, n_targets)

    def forward(self, x):
        return self.linear(x)


def train_loop(dataloader, model, loss_fn, optimizer):
    total_loss = 0.0

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred.ravel(), y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # training loss
        total_loss += loss

    return total_loss


model = TorchLinearRegression(X_train.shape[1], 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

device = "cpu"
model = model.to(device)
dataset = torch.utils.data.TensorDataset(
    torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device)
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)


for i in range(5):
    loss = train_loop(dataloader, model, loss_fn, optimizer)
    print(f"iteration {i}, loss={loss}")

######################
# Let's check the error

y_pred = model(torch.Tensor(X_test)).detach().numpy()
tl2 = mean_squared_error(y_test, y_pred)
tr2 = r2_score(y_test, y_pred)
print(f"TorchLinearRegression: tl2={tl2}, tr2={tr2}")

###########################
# And the coefficients.

print("coefficients:")
for p in model.parameters():
    print(p)
