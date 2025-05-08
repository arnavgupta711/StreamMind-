import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load your preprocessed data
df = pd.read_csv('../data/processed_twitch_reviews.csv')  # Adjust path if needed

# Ensure that NaN values in 'cleaned_content' are filled with an empty string
df['cleaned_content'] = df['cleaned_content'].fillna('')  # Fill NaN with empty string

# Create the labels - assuming `sentiment` column holds binary labels (0, 1)
df['sentiment'] = (df['reviewId'].apply(lambda x: 1 if isinstance(x, str) and len(x) % 2 == 0 else 0))  # Placeholder logic

# Prepare features and labels
X = df['cleaned_content'].apply(lambda x: len(x.split()))  # Feature: Length of cleaned content (as a placeholder feature)
y = df['sentiment']  # Labels: sentiment (binary classification)

# Convert features and labels to tensors
X_tensor = torch.tensor(X.values).float().view(-1, 1)  # Convert to tensor and reshape
y_tensor = torch.tensor(y.values).long()  # Labels must be Long for classification

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# Define a simple DNN model
class SimpleDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = self.fc2(x)  # Output layer
        return x

# Initialize the model, loss function, and optimizer
input_size = 1  # Since we're using the length of cleaned content as a feature
hidden_size = 64  # You can adjust this
output_size = 2  # Binary classification (sentiment: 0 or 1)

model = SimpleDNN(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()  # For multi-class classification

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        inputs, labels = batch
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}")

# Evaluate the model
model.eval()  # Set the model to evaluation mode
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels = batch
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
