import os
import numpy as np
import time
import utils
import metrics
import matplotlib.pyplot as plt
from mobilenetv2 import _inverted_residual_block

from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix

def load_data(feat_folder, is_mono, fold):
   """Load training and testing data for the given fold."""
   feat_file = os.path.join(feat_folder, f'mbe_{"mon" if is_mono else "bin"}_fold{fold}.npz')
   data = np.load(feat_file)
   X_train, Y_train, X_test, Y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
   return X_train, Y_train, X_test, Y_test

def build_model(input_shape, output_shape, cnn_filters, cnn_pool_size, rnn_units, fc_units, dropout_rate):
   """Build the CRNN model."""
   inputs = layers.Input(shape=input_shape)
   x = inputs

   # Convolutional layers
   for pool_size in cnn_pool_size:
      x = layers.Conv2D(filters=cnn_filters, kernel_size=(3, 3), padding='same')(x)
      x = layers.BatchNormalization(axis=1)(x)
      x = layers.Activation('relu')(x)
      x = layers.Conv2D(filters=cnn_filters/2, kernel_size=(3, 3), padding='same')(x)
      x = layers.BatchNormalization(axis=1)(x)
      x = layers.Activation('relu')(x)
      x = layers.MaxPooling2D(pool_size=(1, pool_size))(x)
      x = layers.Dropout(dropout_rate)(x)
   
   x = _inverted_residual_block(x, 64, (1,3), 2, alpha=1.0, strides=1, n=1)
   x = _inverted_residual_block(x, 96, (1,3), 2, alpha=1.0, strides=1, n=2)
   x = _inverted_residual_block(x, 128, (1,3), 3, alpha=1.0, strides=1, n=3)
   x = _inverted_residual_block(x, 128, (1,3), 4, alpha=1.0, strides=1, n=6)
   x = _inverted_residual_block(x, 256, (1,3), 4, alpha=1.0, strides=1, n=6)
   
   # Reshape for RNN layers
   x = layers.Permute((2, 1, 3))(x)
   x = layers.Reshape((input_shape[1], -1))(x)

   # RNN layers
   for units in rnn_units:
      x = layers.Bidirectional(layers.GRU(units, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True), merge_mode='mul')(x)
   
   # Fully connected layers
   for units in fc_units:
      x = layers.TimeDistributed(layers.Dense(units))(x)
      x = layers.Dropout(dropout_rate)(x)
   
   # Output layer
   outputs = layers.TimeDistributed(layers.Dense(output_shape[-1]))(x)
   outputs = layers.Activation('sigmoid', name='strong_out')(outputs)
   
   model = models.Model(inputs=inputs, outputs=outputs)
   model.compile(optimizer='adam', loss='binary_crossentropy')
   model.summary()
   return model

def plot_training_curves(epochs, train_loss, val_loss, f1_scores, error_rates, fig_name):
   """Plot training and validation loss, and F1 and error rates."""
   plt.figure()

   plt.subplot(2, 1, 1)
   plt.plot(range(epochs), train_loss, label='Train Loss')
   plt.plot(range(epochs), val_loss, label='Validation Loss')
   plt.legend()
   plt.grid(True)

   plt.subplot(2, 1, 2)
   plt.plot(range(epochs), f1_scores, label='F1 Score')
   plt.plot(range(epochs), error_rates, label='Error Rate')
   plt.legend()
   plt.grid(True)

   plt.savefig(fig_name)
   plt.close()
   print(f'Figure saved as {fig_name}')

def preprocess_data(X, Y, X_test, Y_test, seq_len, nb_ch):
   """Preprocess data by splitting into sequences and handling multi-channel data."""
   X = utils.split_in_seqs(X, seq_len)
   Y = utils.split_in_seqs(Y, seq_len)
   X_test = utils.split_in_seqs(X_test, seq_len)
   Y_test = utils.split_in_seqs(Y_test, seq_len)

   X = utils.split_multi_channels(X, nb_ch)
   X_test = utils.split_multi_channels(X_test, nb_ch)
   return X, Y, X_test, Y_test

is_mono = False
feat_folder = 'data/feat/'
fig_name = f'{"mon" if is_mono else "bin"}_{time.strftime("%Y_%m_%d_%H_%M_%S")}'

nb_ch = 1 if is_mono else 2
batch_size = 256
seq_len = 256
nb_epoch = 500
# patience = int(0.25 * nb_epoch)
patience = int(0.2 * nb_epoch)

sr = 44100
nfft = 2048
frames_1_sec = int(sr / (nfft / 2.0))

print(f'\n\nUNIQUE ID: {fig_name}')
print(f'TRAINING PARAMETERS: nb_ch: {nb_ch}, seq_len: {seq_len}, batch_size: {batch_size}, nb_epoch: {nb_epoch}, frames_1_sec: {frames_1_sec}')

models_dir = 'models/'
utils.create_folder(models_dir)

# Model parameters
cnn_filters = 128
cnn_pool_size = [5, 2, 2, 2]
rnn_units = [64, 32, 16, 16]
fc_units = [64]
dropout_rate = 0.45
#######------------------------
# nb_cnn2d_filt=64
# pool_size=[5, 2, 2, 2]
# rnn_size=[128, 128]
# fnn_size=[128]
# dropout_rate = 0.5

# print(f'MODEL PARAMETERS:\n cnn_filters: {cnn_filters}, cnn_pool_size: {cnn_pool_size}, rnn_units: {rnn_units}, fc_units: {fc_units}, dropout_rate: {dropout_rate}')

X_train, Y_train, X_test, Y_test = load_data(feat_folder, is_mono, 1)
X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test, seq_len, nb_ch)

avg_er = []
avg_f1 = []

for fold in [1, 2, 3, 4]:
   print(f'\n\n----------------------------------------------')
   print(f'FOLD: {fold}')
   print(f'----------------------------------------------\n')

   X_train, Y_train, X_test, Y_test = load_data(feat_folder, is_mono, fold)
   X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test, seq_len, nb_ch)

   model = build_model(X_train.shape[1:], Y_train.shape[2:], cnn_filters, cnn_pool_size, rnn_units, fc_units, dropout_rate)
   # model = get_sedtcn_model(X_train.shape[1:], Y_train.shape[2:], dropout_rate, nb_cnn2d_filt, pool_size, fnn_size)

   best_epoch, patience_counter, best_er, f1_for_best_er, best_conf_mat = 0, 0, float('inf'), None, None
   train_loss, val_loss, f1_scores, error_rates = [], [], [], []

   for epoch in range(nb_epoch):
      print(f'Epoch: {epoch}', end=' ')
      history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test), epochs=1, verbose=2)

      train_loss.append(history.history['loss'][-1])
      val_loss.append(history.history['val_loss'][-1])

      pred = model.predict(X_test)
      pred_thresh = pred > 0.5
      scores = metrics.compute_scores(pred_thresh, Y_test, frames_in_1_sec=frames_1_sec)

      f1_scores.append(scores['f1_overall_1sec'])
      error_rates.append(scores['er_overall_1sec'])
      patience_counter += 1

      conf_mat = confusion_matrix(np.sum(Y_test, axis=2).reshape(-1), np.sum(pred_thresh, axis=2).reshape(-1))
      conf_mat = conf_mat / (np.sum(conf_mat, axis=1, keepdims=True) + 1e-7)

      if scores['er_overall_1sec'] < best_er:
         best_er = scores['er_overall_1sec']
         f1_for_best_er = scores['f1_overall_1sec']
         best_conf_mat = conf_mat
         best_epoch = epoch
         patience_counter = 0
         model.save(os.path.join(models_dir, f'{fig_name}_fold_{fold}_model.h5'))

      print(f'tr_loss: {train_loss[-1]:.4f}, val_loss: {val_loss[-1]:.4f}, F1: {f1_scores[-1]:.4f}, ER: {error_rates[-1]:.4f}, Best ER: {best_er:.4f} at epoch {best_epoch}')

      plot_training_curves(epoch + 1, train_loss, val_loss, f1_scores, error_rates, f'{models_dir}/{fig_name}_fold_{fold}.png')

      if patience_counter > patience:
         break

   avg_er.append(best_er)
   avg_f1.append(f1_for_best_er)
   print(f'Best model saved at epoch {best_epoch} with best ER: {best_er:.4f} and F1: {f1_for_best_er:.4f}')
   print(f'Best Confusion Matrix:\n{best_conf_mat}')
   print(f'Diagonal of Confusion Matrix:\n{np.diag(best_conf_mat)}')

print(f'\n\nMETRICS FOR ALL FOUR FOLDS:\n avg_er: {avg_er}\n avg_f1: {avg_f1}')
print(f'MODEL AVERAGE OVER FOUR FOLDS:\n avg_er: {np.mean(avg_er):.4f}\n avg_f1: {np.mean(avg_f1):.4f}')