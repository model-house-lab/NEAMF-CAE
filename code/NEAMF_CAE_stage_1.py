import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from model_upload import *
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.callbacks import  EarlyStopping
import datetime

def CAE_train():
    positive_csv_data = pd.read_csv(r'..\\data\\data_image.csv')  # shape= (32,64,64,1)
    positive_data_path = positive_csv_data['File Path'].tolist()
    positive_data_path = sorted(positive_data_path, key=sort_key_32)


    positive_csv_math = pd.read_csv(r'..\\data\\data_math.csv')  # shape= (32,64,64,1)
    positive_math_path = positive_csv_math['File Path'].tolist()
    positive_math_path = sorted(positive_math_path, key=sort_key_32)


    positive_train_data_path, positive_val_data_path = train_test_split(positive_data_path, test_size=0.2, random_state=42)
    print("positive_train_data_path==", positive_train_data_path)


    batch_size = 8
    positive_train_data_path, positive_val_data_path = train_test_split(positive_data_path, test_size=0.2,random_state=42)
    print(positive_train_data_path[0])
    positive_train_math_path, positive_val_math_path = train_test_split(positive_math_path, test_size=0.2,random_state=42)
    print(positive_train_math_path[0])

    train_generator = DataGenerator_double(positive_train_data_path, positive_train_math_path, batch_size=batch_size,dim=(32, 64, 64, 1))
    val_generator = DataGenerator_double(positive_val_data_path, positive_val_math_path, batch_size=batch_size,dim=(32, 64, 64, 1))


    autoencoder = NEAMF_CAE_encoder((32, 64, 64, 1),batch_size=batch_size)
    autoencoder.summary(line_length=150,positions=[0.30,0.60,0.7,1.])

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='MSE')

    early_stopping = EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True)


    model_checkpoint = ModelCheckpoint('NEAMF_CAE.tf', monitor='val_loss', save_best_only=True, mode='min',overwrite=True,save_format="tf")


    history = autoencoder.fit(
        train_generator,
        epochs=30,
        batch_size=batch_size,
        validation_data=val_generator,
        callbacks=[early_stopping, model_checkpoint]
    )

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    now = datetime.datetime.now()

    plt.plot(epochs, train_loss, 'b', label='Training loss' )
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(r'..\\fig\\train_loss_{}_{}_{}_{}.png'.format(now.month, now.day, now.hour, now.minute))
    plt.show()

    plt.clf()
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(r'..\\fig\\train_loss_{}_{}_{}_{}.png'.format(now.month, now.day, now.hour, now.minute))
    plt.show()

CAE_train()