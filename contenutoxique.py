import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub #importation des modele de tensorflowhub
import tensorflow_datasets as tfds
plt.rcParams['figure.figsize']=(12,8) 
from IPython import display 
import pathlib
import shutil
import tempfile
!pip install -q git+https://github.com/tensorflow/docs
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
print("Version: ", tf.__version__)
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

df=pd.read_csv('https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip',compression='zip',low_memory=False)
df.shape

df['target'].plot(kind='hist',title='Target distribution')

from sklearn.model_selection import train_test_split
train_df,remaining=train_test_split(df,random_state=42,train_size=0.01,stratify=df.target.values)
valid_df,_=train_test_split(remaining,random_state=42,train_size=0.001,stratify=remaining.target.values)
train_df.shape
train_df.target.head(15).values,train_df.question_text.head(15).values
module_url='https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'
def train_and_evaluate_model(module_url,embed_size,name,trainable=False):
  hub_layer=hub.KerasLayer(module_url,input_shape=[],output_shape=[embed_size],dtype=tf.string,trainable=trainable) #importation du module
  model=tf.keras.models.Sequential([                                #Constituion des couches Denses
      hub_layer,
      tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
  ])
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.losses.BinaryCrossentropy(),
              metrics=tf.metrics.BinaryAccuracy(name='accuracy'))
  history=model.fit(
                    train_df['question_text'],
                    train_df['target'],
                    epochs=100,
                    batch_size=32,
                    validation_data=(valid_df['question_text'],valid_df['target']),
                    callbacks=[tfdocs.modeling.EpochDots(),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min'),
                    tf.keras.callbacks.TensorBoard(logdir/name)],
                    verbose=0)
  return history 
  

