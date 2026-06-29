import numpy as np
import seaborn as sns

def unit_step_func(x):
  return np.where(x > 0, 1, 0)

class Perceptron:

  def __init__(self, learning_rate=0.01, n_iters=1000):
    self.lr = learning_rate
    self.n_iters = n_iters
    self.activation_func = unit_step_func
    self.weights = None
    self.bias = None

  def fit(self, x, y):
    n_samples, n_features = x.shape

    #init paramaters
    self.weights = np.random.randn(n_features) * 0.01
    self.bias = np.random.randn() * 0.01

    y_ = y

    # Learn weights
    for _ in range(self.n_iters):
      for idx, x_i in enumerate(x):
        linear_output = np.dot(x_i, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)

        # Perceptron update rule
        update = self.lr * (y_[idx] - y_predicted)
        self.weights += update * x_i
        self.bias += update



  def predict(self, x):
    linear_output = np.dot(x, self.weights) + self.bias
    y_predicted = self.activation_func(linear_output)
    return y_predicted

# Load the Titanic dataset
titanic = sns.load_dataset("titanic")
print(titanic.head(4))

# Keep only the columns we need
data = titanic[["survived", "sex", "pclass", "age", "alone"]].copy()

# Convert sex from words into numbers
# male = 0 and female = 1
data["sex"] = data["sex"].map({
    "male": 0,
    "female": 1
})

# Convert alone into numbers
data["alone"] = data["alone"].map({
    False: 0,
    True: 1
})

# Remove passengers whose age is missing
data = data.dropna()

# Create shuffled row numbers
np.random.seed(42)
indices = np.arange(len(x))
np.random.shuffle(indices)

# Put 80% in training and 20% in testing
split_index = int(len(x) * 0.5)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

# Split the input data
x_train = x[train_indices]
x_test = x[test_indices]

# Split the correct answers
y_train = y[train_indices]
y_test = y[test_indices]

print("Training rows:", len(x_train))
print("Testing rows:", len(x_test))

# Train the perceptron
model = Perceptron(learning_rate=0.01, n_iters=1000)

model.fit(x_train, y_train)



predictions = model.predict(x_test)

#Check accuracy 
accuracy = np.sum(predictions == y_test) / len(y_test)

print("Accuracy:", accuracy * 100, "%")






