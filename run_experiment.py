import configparser

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from numpy.random import seed
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import set_random_seed

from algo.nn.models import attention_capsule
from algo.nn.utility import f1_smart
from embeddings import get_emb_matrix
from preprocessing import clean_text, remove_names, remove_tags

if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed(726)
    set_random_seed(726)

    # Read the dataset
    print('Reading files')
    full = pd.read_csv("data/arabic/training.csv", sep='\t', header=None, names=["id", "tweet", "label"], index_col=0)
    print("Number of tweets in the dataset: ", full.shape[0])

    # Split the dataset into training and test sets
    train, test = train_test_split(full, test_size=0.2)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    print('Completed reading')

    print("Train shape : ", train.shape)
    print("Test shape : ", test.shape)

    # Define columns
    TEXT_COLUMN = "tweet"
    LABEL_COLUMN = "label"

    # Load configuration
    configParser = configparser.RawConfigParser()
    configFilePath = "config.txt"
    configParser.read(configFilePath)

    EMBEDDING_FILE = configParser.get('model-config', 'EMBEDDING_FILE')
    MODEL_PATH = configParser.get('model-config', 'MODEL_PATH')
    PREDICTION_FILE = configParser.get('model-config', 'PREDICTION_FILE')

    # Preprocess text data
    print("Removing usernames")
    train[TEXT_COLUMN] = train[TEXT_COLUMN].apply(lambda x: remove_names(x))
    test[TEXT_COLUMN] = test[TEXT_COLUMN].apply(lambda x: remove_names(x))

    print("Removing tags")
    train[TEXT_COLUMN] = train[TEXT_COLUMN].apply(lambda x: remove_tags(x))
    test[TEXT_COLUMN] = test[TEXT_COLUMN].apply(lambda x: remove_tags(x))

    # Clean text
    train[TEXT_COLUMN] = train[TEXT_COLUMN].apply(lambda x: clean_text(x))
    test[TEXT_COLUMN] = test[TEXT_COLUMN].apply(lambda x: clean_text(x))

    # Calculate document length and determine maximum sequence length
    train['doc_len'] = train[TEXT_COLUMN].apply(lambda words: len(words.split(" ")))
    max_seq_len = np.round(train['doc_len'].mean() + train['doc_len'].std()).astype(int)

    # Set embedding parameters
    embed_size = 300  # Size of each word vector
    max_features = None  # Number of unique words
    maxlen = max_seq_len  # Maximum number of words in a sequence

    # Fill missing values
    X = train[TEXT_COLUMN].fillna("_na_").values
    X_test = test[TEXT_COLUMN].fillna("_na_").values

    # Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features, filters='')
    tokenizer.fit_on_texts(list(X))

    X = tokenizer.texts_to_sequences(X)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Pad the sequences
    X = pad_sequences(X, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)

    # Get target values
    Y = train[LABEL_COLUMN].values

    word_index = tokenizer.word_index
    max_features = len(word_index) + 1

    # Load embeddings
    print('Loading Embeddings')
    embedding_matrix = get_emb_matrix(word_index, max_features, EMBEDDING_FILE)
    print('Finished loading Embeddings')
    print("Number of tweets in the dataset: ", full.shape[0])

    # Start training
    print('Start Training')
    kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
    best_score = -1
    best_fold_metrics = {}
    y_test = np.zeros((X_test.shape[0],))

    for i, (train_index, valid_index) in enumerate(kfold.split(X, Y)):
        X_train, X_val, Y_train, Y_val = X[train_index], X[valid_index], Y[train_index], Y[valid_index]
        filepath = MODEL_PATH

        # Define callbacks
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, min_lr=0.0001, verbose=2)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
        callbacks = [checkpoint, reduce_lr, earlystopping]

        # Initialize and train the model
        model = attention_capsule(maxlen, max_features, embed_size, embedding_matrix)
        if i == 0:
            print(model.summary())
        model.fit(X_train, Y_train, batch_size=64, epochs=20, validation_data=(X_val, Y_val), verbose=2, callbacks=callbacks)

        # Load the best model weights and make predictions
        model.load_weights(filepath)
        y_pred = model.predict([X_val], batch_size=64, verbose=2)
        y_test += np.squeeze(model.predict([X_test], batch_size=64, verbose=2)) / 5

        # Calculate F1 score and find optimal threshold
        f1, threshold = f1_smart(np.squeeze(Y_val), np.squeeze(y_pred))
        print(f'Fold {i + 1}: Optimal F1: {f1:.4f} at threshold: {threshold:.4f}')
        
        if f1 > best_score:
            best_score = f1
            best_fold_metrics = {
                'f1': f1,
                'threshold': threshold,
                'y_true': Y_val,
                'y_pred': np.squeeze(y_pred)
            }

    print('Finished Training')

    # Final predictions on the test set
    y_test = y_test.reshape((-1, 1))
    pred_test_y = (y_test > best_fold_metrics['threshold']).astype(int)
    test['predictions'] = pred_test_y

    # Save predictions
    file_path = PREDICTION_FILE
    test.to_csv(file_path, sep='\t', encoding='utf-8')
    print('Saved Predictions')

    # Post analysis based on the best fold
    tn, fp, fn, tp = confusion_matrix(best_fold_metrics['y_true'], (best_fold_metrics['y_pred'] > best_fold_metrics['threshold']).astype(int)).ravel()
    best_accuracy = accuracy_score(best_fold_metrics['y_true'], (best_fold_metrics['y_pred'] > best_fold_metrics['threshold']).astype(int))
    best_weighted_f1 = f1_score(best_fold_metrics['y_true'], (best_fold_metrics['y_pred'] > best_fold_metrics['threshold']).astype(int), average='weighted')
    best_weighted_recall = recall_score(best_fold_metrics['y_true'], (best_fold_metrics['y_pred'] > best_fold_metrics['threshold']).astype(int), average='weighted')
    best_weighted_precision = precision_score(best_fold_metrics['y_true'], (best_fold_metrics['y_pred'] > best_fold_metrics['threshold']).astype(int), average='weighted')

    print("Best Confusion Matrix (tn, fp, fn, tp): {} {} {} {}".format(tn, fp, fn, tp))
    print("Best Accuracy: ", best_accuracy)
    print("Best Weighted F1: ", best_weighted_f1)
    print("Best Weighted Recall: ", best_weighted_recall)
    print("Best Weighted Precision: ", best_weighted_precision)
