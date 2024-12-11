# Install required libraries

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
# Disable W&B integration
import os
os.environ["WANDB_DISABLED"] = "true"

# Rest of your imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# ... rest of your code


# Load the dataset
file_path = r'C:/Users/gokul/fakenews/fake_and_real_news.csv'  # Update the path to your dataset
df = pd.read_csv(file_path)

# Display dataset head
print("Dataset Head:")
print(df.head())

# Ensure the dataset has the required columns
text_column = 'Text'  # Update to match your dataset's text column
label_column = 'label'  # Update to match your dataset's label column

# Drop rows with missing values
df = df[[text_column, label_column]].dropna()

# Convert labels to numerical values
df[label_column] = df[label_column].map({'Fake': 0, 'Real': 1})

# Check class distribution
print("\nClass Distribution:")
print(df[label_column].value_counts())

# Step 1: Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df[text_column], df[label_column], test_size=0.2, random_state=42
)

# Step 2: Prepare the dataset for the transformer model
model_name = "bert-base-uncased"  # You can replace this with another model
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

train_encodings = tokenize_function(list(train_texts))
test_encodings = tokenize_function(list(test_texts))

# Convert labels to tensors
train_labels = torch.tensor(list(train_labels.values))
test_labels = torch.tensor(list(test_labels.values))

# Create Hugging Face Dataset objects
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# Step 3: Define the model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 4: Define the Trainer and Training Arguments
training_args = TrainingArguments(
    output_dir="./results",          # Output directory
    evaluation_strategy="epoch",    # Evaluate after each epoch
    learning_rate=2e-5,             # Learning rate
    per_device_train_batch_size=16, # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    num_train_epochs=3,             # Number of epochs
    weight_decay=0.01,              # Weight decay
    logging_dir="./logs",           # Log directory
    save_strategy="epoch",          # Save the model at each epoch
    load_best_model_at_end=True     # Load the best model at the end
)

# Custom evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model to your local system
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
print("Model and tokenizer saved successfully!")

# Step 5: Evaluate the model on the test set
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Metrics
print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels))

print("\nAccuracy Score:", accuracy_score(test_labels, predicted_labels))

# Step 6: Confusion Matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Visualization
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ['Fake', 'Real'])
plt.yticks([0, 1], ['Fake', 'Real'])

# Add numbers inside the confusion matrix
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')

plt.show()

