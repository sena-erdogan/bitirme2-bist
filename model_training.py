##################
#PRETRAINING CODE#
##################


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import re
import emoji
import string
import time
import logging
from zemberek import TurkishSpellChecker, TurkishSentenceNormalizer, TurkishMorphology, TurkishTokenizer

# function to clean the text
def cleaner(text):
    cleaned = re.sub(r'rt @\w+ :\s*', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'@\w+\s*', '', cleaned)
    cleaned = re.sub(r'http\S+|www\S+', '', cleaned)
    cleaned = ''.join([char for char in cleaned if char not in string.punctuation])
    cleaned = re.sub(r'\b\w+\d+\b', '', cleaned)
    cleaned = emoji.demojize(cleaned)
    cleaned = cleaned.strip().lower()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

# Load dataset
unlabeled_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/pretrain_spellchecked.csv')

unlabeled_data['content'] = unlabeled_data['content'].fillna('')
unlabeled_data = unlabeled_data.drop_duplicates(subset='content')

# Clean the preprocessed data
unlabeled_data['content'] = unlabeled_data['content'].apply(cleaner)

# MLM pretraining code
tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased', use_fast=True)
max_length = 250
unlabeled_texts = unlabeled_data['content'].tolist()
unlabeled_encodings = tokenizer(unlabeled_texts, truncation=True, padding=True, return_tensors="pt", max_length=max_length)

label_encoder = LabelEncoder()

batch_size = 64

mlm_model = AutoModelForMaskedLM.from_pretrained('dbmdz/bert-base-turkish-uncased')

mlm_optimizer = AdamW(mlm_model.parameters(), lr=1e-4, eps=1e-8)
mlm_epochs = 5
mlm_total_steps = len(unlabeled_texts) * mlm_epochs
mlm_scheduler = get_linear_schedule_with_warmup(mlm_optimizer, num_warmup_steps=0, num_training_steps=mlm_total_steps)

batch_size = 32
unlabeled_encodings = []
for i in range(0, len(unlabeled_texts), batch_size):
    batch_texts = unlabeled_texts[i:i+batch_size]
    batch_encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    unlabeled_encodings.append(batch_encodings)

mlm_model.train()
for epoch in range(mlm_epochs):
    total_mlm_loss = 0
    for batch_num, batch in enumerate(unlabeled_encodings):
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        labels = input_ids.clone()  # MLM labels are the same as the input_ids
        mlm_optimizer.zero_grad()
        outputs = mlm_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_mlm_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlm_model.parameters(), 1.0)
        mlm_optimizer.step()
        mlm_scheduler.step()
    avg_mlm_loss = total_mlm_loss / len(unlabeled_encodings)
    print(f'MLM Pretraining Epoch {epoch+1}: Loss = {avg_mlm_loss:.3f}')

# Save the MLM model
mlm_model_path = "/content/drive/MyDrive/ColabNotebookModels/mlmmodel.pth"
mlm_model.save_pretrained(mlm_model_path)


#################################
#MODEL TRAINING - FINETUNIG CODE#
#################################


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.stem.porter import PorterStemmer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns

train_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/train.csv')

test_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/test.csv')

train_data = train_data.dropna(axis=1, how='all')
test_data = test_data.dropna(axis=1, how='all')
train_data = train_data.dropna(axis=0, how='all')
test_data = test_data.dropna(axis=0, how='all')

train_data['content'] = train_data['content'].fillna('')
train_data = train_data.drop_duplicates(subset='content')

test_data['content'] = test_data['content'].fillna('')
test_data = test_data.drop_duplicates(subset='content')

train_texts = train_data['content'].tolist()
test_texts = test_data['content'].tolist()

tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased', use_fast=True)
max_length = 250
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt", max_length=max_length)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt", max_length=max_length)

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_data['label'])
test_labels = label_encoder.transform(test_data['label'])

train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))

mlm_model_path = "/content/drive/MyDrive/ColabNotebookModels/mlmmodel.pth"

model = AutoModelForSequenceClassification.from_pretrained(mlm_model_path, num_labels=3)

print('1')

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 10
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

print('2')

# Set the early stopping parameters
early_stopping_patience = 3
best_val_loss = float('inf')
patience = 0

print('3')

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

print('4')

# Fine-tune the model
for epoch in range(epochs):

    print('5')

    model.train()
    total_train_loss = 0
    total_train_accuracy = 0
    for batch_num, batch in enumerate(train_dataloader):
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip the gradients to prevent exploding gradients
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(outputs.logits, dim=1)
        total_train_accuracy += torch.sum(preds == labels).item()

        if (batch_num + 1) % 50 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_num+1}/{len(train_dataloader)}: Loss = {loss.item():.3f}')

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_accuracy = total_train_accuracy / len(train_dataset)

    # Evaluate on the validation set
    model.eval()
    total_val_loss = 0
    total_val_accuracy = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_val_accuracy += torch.sum(preds == labels).item()

    avg_val_loss = total_val_loss / len(test_dataloader)
    avg_val_accuracy = total_val_accuracy / len(test_dataset)

    print(f'Epoch {epoch+1}:')
    print(f'Training Loss: {avg_train_loss:.3f}')
    print(f'Training Accuracy: {avg_train_accuracy:.3f}')
    print(f'Validation Loss: {avg_val_loss:.3f}')
    print(f'Validation Accuracy: {avg_val_accuracy:.3f}')

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(avg_train_accuracy)
    val_accuracies.append(avg_val_accuracy)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience = 0
        torch.save(model.state_dict(), 'bozukmodel.pth')
    else:
        patience += 1

    if patience == early_stopping_patience:
        print(f'Early stopping after {epoch+1} epochs')
        break

# Load the best model
model.load_state_dict(torch.load('bozukmodel.pth'))

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Training Accuracy', color='blue')
plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()


##############################
#STARLANG SPELL CHECKING CODE#
##############################


import pandas as pd
from Corpus.Sentence import Sentence
from MorphologicalAnalysis.FsmMorphologicalAnalyzer import FsmMorphologicalAnalyzer
from NGram.NGram import NGram
from NGram.NoSmoothing import NoSmoothing
from SpellChecker.ContextBasedSpellChecker import ContextBasedSpellChecker
from SpellChecker.SpellCheckerParameter import SpellCheckerParameter

# Load your dataset
unlabeled_data = pd.read_csv('train.csv')

# Preprocessing steps
unlabeled_data['content'] = unlabeled_data['content'].fillna('')
unlabeled_data = unlabeled_data.drop_duplicates(subset='content')

# Initialize spell checker components
fsm = FsmMorphologicalAnalyzer()
nGram = NGram("../ngram.txt")
nGram.calculateNGramProbabilitiesSimple(NoSmoothing())
contextBasedSpellChecker = ContextBasedSpellChecker(fsm, nGram, SpellCheckerParameter())

total_count = len(unlabeled_data)

# Perform spell checking on each content and print progress
for index, row in unlabeled_data.iterrows():
    content = row['content']

    # Perform spell checking on the content
    original_sentence = Sentence(content)
    corrected_sentence = contextBasedSpellChecker.spellCheck(original_sentence)

    # Update the content with the corrected sentence
    unlabeled_data.at[index, 'content'] = corrected_sentence.toString()

    # Print progress
    current_count = index + 1
    remaining_count = total_count - current_count
    print("Processed content:", current_count, "Remaining:", remaining_count)

# Save the preprocessed and spell-checked data to a new CSV file
unlabeled_data.to_csv('train_spellchecked.csv', index=False)

print("Spell checking and preprocessing completed.     Preprocessed data saved in preprocessed_spellchecked.csv")

