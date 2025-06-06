import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPool2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import matplotlib.pyplot as plt
import scipy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)

train_data = train_datagen.flow_from_directory('archive/raw-img',
                                              target_size = (256,256), 
                                              batch_size = 32,
                                              class_mode = 'categorical') 


test_data = test_datagen.flow_from_directory('archive/raw-img',
                                              target_size = (256,256),
                                              batch_size = 32,
                                              class_mode = 'categorical') 


early_stopping_callback = EarlyStopping(
    monitor='val_loss',         
    patience=10,                
    verbose=1,                  
    mode='min',                
    restore_best_weights=True  
)

saver = ModelCheckpoint(
    filepath='best_model_checkpoint.keras', 
    monitor='val_accuracy',                
    save_best_only=True,                   
    verbose=1,                              
    mode='max',                             
    save_weights_only=False                 
)
model = keras.Sequential()
model.add(keras.Input(shape=(256, 256, 3)))


model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', kernel_regularizer= l2(0.001)))

model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))


model.add(Dense(512, activation='relu', kernel_regularizer= l2(0.001)))
model.add(Dropout(0.4))


model.add(Dense(10, activation='softmax'))

print(model.summary())



model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, epochs = 20, validation_data=test_data, callbacks = [saver, early_stopping_callback])





acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(20) 

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1) 
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout() 
plt.show()

model.save('model_0.h5')