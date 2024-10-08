'''
NumPy:
Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585(7825), 357-362. https://doi.org/10.1038/s41586-020-2649-2
NumPy documentation: https://numpy.org/doc/stable/

Pandas:
McKinney, W. (2010). Data structures for statistical analysis. In Proceedings of the 9th Python in Science Conference (SciPy 2010) (pp. 51-56). https://doi.org/10.25080/Majora-92bf1922-00a
Pandas documentation: https://pandas.pydata.org/pandas-docs/stable/

PyTorch:
Paszke, A., Gross, S., & Massa, F. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in neural information processing systems (pp. 8024-8035). https://arxiv.org/abs/1912.01703
PyTorch documentation: https://pytorch.org/docs/stable/index.html

Transformers (Hugging Face):
Wolf, T., Sanh, V., Chaumond, J., & Delangue, C. (2020). Transformers: State-of-the-art natural language processing. https://doi.org/10.5281/zenodo.1207013
Transformers documentation: https://huggingface.co/transformers/

Scikit-learn:
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830. http://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf
Scikit-learn documentation: https://scikit-learn.org/stable/documentation.html

AdamW Optimizer:
Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR). https://arxiv.org/abs/1711.05101
This optimizer is included in PyTorch, so further details are available in the PyTorch documentation: https://pytorch.org/docs/stable/optim.html

AraBERT:
Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based model for Arabic language understanding. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020) (pp. 889-897). https://arxiv.org/abs/2003.00104
AraBERT documentation: https://huggingface.co/aubmindlab/bert-base-arabertv2
'''
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Define the f1_smart function as needed
def f1_smart(y_true, y_pred_probs):
    thresholds = np.arange(0.0, 1.1, 0.01)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_probs > threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    optimal_idx = np.argmax(f1_scores)
    optimal_f1 = f1_scores[optimal_idx]
    optimal_threshold = thresholds[optimal_idx]

    return optimal_f1, optimal_threshold

# Set random seeds
np.random.seed(726)
torch.manual_seed(726)

# Define paths directly
MODEL_PATH = "models/capsule_weights_best.pth"
PREDICTION_FILE = "arabert_predictions.csv"

# Create the directory if it does not exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Correct file path
file_path = 'data/arabic/training.csv'

# Read the CSV file
full = pd.read_csv(file_path, sep='\t', header=None, names=["id", "tweet", "label"], index_col=0)
print("Number of tweets in the dataset: ", full.shape[0])

# Split dataset
train, test = train_test_split(full, test_size=0.2)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

print("Completed reading")
print("Train shape : ", train.shape)
print("Test shape : ", test.shape)

# Variables
TEXT_COLUMN = "tweet"
LABEL_COLUMN = "label"

# Tokenizer and Encoding for AraBERT
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")

def tokenize_and_encode(texts, max_length=128):
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'  # Return PyTorch tensors
    )
    return encodings

# Tokenize and encode sequences in the training set
train_encodings = tokenize_and_encode(list(train[TEXT_COLUMN].values))
test_encodings = tokenize_and_encode(list(test[TEXT_COLUMN].values))

# Convert to PyTorch Datasets
def create_torch_dataset(encodings, labels):
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels, dtype=torch.long)
    )
    return dataset

# Prepare datasets
label_encoder = LabelEncoder()

# Fit the encoder on the training labels
Y_train = label_encoder.fit_transform(train[LABEL_COLUMN].values)

# Transform the test labels using the fitted encoder
Y_test = label_encoder.transform(test[LABEL_COLUMN].values)

train_dataset = create_torch_dataset(train_encodings, Y_train)
test_dataset = create_torch_dataset(test_encodings, Y_test)

# Define DataLoader
def create_dataloader(dataset, batch_size=64, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_dataloader = create_dataloader(train_dataset)
test_dataloader = create_dataloader(test_dataset, shuffle=False)

# Define the model
model = AutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv2", num_labels=len(np.unique(Y_train)))

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Define training loop
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Average Training Loss: {avg_loss:.4f}")

# Training the model
kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)

best_f1_scores = []
best_fold_metrics = {}
best_fold_conf_matrix = None
best_fold_index = -1
y_test = np.zeros((len(Y_test),))

for fold, (train_index, valid_index) in enumerate(kfold.split(train_encodings['input_ids'], Y_train)):
    print(f"Starting fold {fold+1}")

    X_train_encodings = {
        'input_ids': torch.tensor(np.array(train_encodings['input_ids'])[train_index]),
        'attention_mask': torch.tensor(np.array(train_encodings['attention_mask'])[train_index]),
    }
    X_val_encodings = {
        'input_ids': torch.tensor(np.array(train_encodings['input_ids'])[valid_index]),
        'attention_mask': torch.tensor(np.array(train_encodings['attention_mask'])[valid_index]),
    }
    Y_train_split = torch.tensor(Y_train[train_index], dtype=torch.long)
    Y_val_split = torch.tensor(Y_train[valid_index], dtype=torch.long)

    # Prepare the datasets for training and validation
    train_split_dataset = create_torch_dataset(X_train_encodings, Y_train_split)
    val_split_dataset = create_torch_dataset(X_val_encodings, Y_val_split)

    train_split_dataloader = create_dataloader(train_split_dataset)
    val_split_dataloader = create_dataloader(val_split_dataset, shuffle=False)

    # Training
    train_model(model, train_split_dataloader, optimizer, device)

    # Save the best model weights
    torch.save(model.state_dict(), MODEL_PATH)

    # Load the best model weights
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Predict and calculate F1 score
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_split_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate performance metrics
    f1, threshold = f1_smart(np.array(all_labels), np.array(all_preds))
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    print(f"Fold {fold+1} - F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    if not best_f1_scores or f1 > max(best_f1_scores):
        best_f1_scores.append(f1)
        best_fold_metrics = {
            'f1': f1,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
        }
        best_fold_conf_matrix = (tn, fp, fn, tp)
        best_fold_index = fold + 1

    # Prediction on the test set
    all_test_preds = []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, _ = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_test_preds.extend(preds.cpu().numpy())

    y_test_pred = np.array(all_test_preds)
    y_test += y_test_pred / kfold.n_splits

print('Finished Training')

# Final predictions on the test set
y_test = y_test.reshape((-1, 1))
pred_test_y = (y_test > np.mean(best_f1_scores)).astype(int)
test['predictions'] = pred_test_y

# Save predictions
test.to_csv(PREDICTION_FILE, sep='\t', encoding='utf-8')
print('Saved Predictions')

# Print the best fold metrics
print(f"Best Fold: {best_fold_index}")
print(f"Best F1: {best_fold_metrics['f1']:.4f}")
print(f"Best Accuracy: {best_fold_metrics['accuracy']:.4f}")
print(f"Best Precision: {best_fold_metrics['precision']:.4f}")
print(f"Best Recall: {best_fold_metrics['recall']:.4f}")
print("Best Confusion Matrix (tn, fp, fn, tp):", best_fold_conf_matrix)
