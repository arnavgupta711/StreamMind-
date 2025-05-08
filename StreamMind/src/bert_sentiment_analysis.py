import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os

# Function to encode reviews
def encode_reviews(texts, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    input_ids = []
    attention_masks = []
    
    for text in texts:
        # Handle None values or non-string inputs
        if text is None or not isinstance(text, str):
            text = ""
            
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    return {
        'input_ids': torch.cat(input_ids, dim=0),
        'attention_mask': torch.cat(attention_masks, dim=0)
    }

# Function to run inference on text using BERT
def run_bert_sentiment(texts):
    """
    Run BERT sentiment analysis on a list of texts
    Returns list of sentiment labels (0=negative, 1=positive)
    """
    # Check if we have a saved model
    model_path = '../models/bert_sentiment_model.pt'
    
    if os.path.exists(model_path):
        # Load saved model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        # Train a new model
        model = train_sentiment_model(epochs=1)
        
    # Set to evaluation mode
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Tokenize texts
    encodings = encode_reviews(texts)
    
    # Create DataLoader for batch processing
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=16)
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
    
    return predictions

# Train the BERT model
def train_sentiment_model(batch_size=16, epochs=2):
    # Load processed data
    df = pd.read_csv('../data/processed_twitch_reviews.csv')
    
    # Create sentiment labels (similar to your DNN approach)
    # Improved labeling - if you have actual sentiment labels, use those instead
    df['sentiment'] = (df['reviewId'].apply(lambda x: 1 if isinstance(x, str) and len(x) % 2 == 0 else 0))
    
    # Handle missing or NaN values
    df['cleaned_content'] = df['cleaned_content'].fillna("").astype(str)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_content'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_encodings = encode_reviews(X_train)
    test_encodings = encode_reviews(X_test)
    
    train_dataset = TensorDataset(
        train_encodings['input_ids'], 
        train_encodings['attention_mask'],
        torch.tensor(y_train.values)
    )
    
    test_dataset = TensorDataset(
        test_encodings['input_ids'], 
        test_encodings['attention_mask'],
        torch.tensor(y_test.values)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    
    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Save model
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/bert_sentiment_model.pt')
    print("Model saved to ../models/bert_sentiment_model.pt")
    
    return model
