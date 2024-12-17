
# Emotion Classification 

This project utilizes Convolutional Neural Networks (CNN) for classifying emotions from images into two categories: Happy and Sad. The dataset consists of images from two classes, and the model is trained to predict the emotional state of a given image.

## Project Setup

To run this project, you need to install the following dependencies:

- TensorFlow
- OpenCV
- NumPy
- Matplotlib

You can install these dependencies using the following command:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

## Dataset

The dataset consists of images stored in different folders, where each folder represents a class:

- **happy**
- **sad**

The images are in various formats such as JPEG, PNG, BMP, etc. The dataset is assumed to be structured as follows:

```
/data
  /happy
  /sad
```

## Project Workflow

### 1. Import Libraries

We start by importing necessary libraries for image processing, data handling, and model creation.

```python
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
```

### 2. Data Cleaning

#### a) Remove Images Below 10KB

Images that are too small are likely corrupted or not relevant to our task, so we remove them from the dataset.

#### b) Filter Valid Image Extensions

We filter the images based on valid extensions (JPEG, JPG, BMP, PNG). Any image with an invalid extension is deleted.

```python
data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
```

### 3. Loading the Data

We use `tensorflow.keras.utils.image_dataset_from_directory` to load the images and create a dataset. This function automatically handles tasks such as resizing and shuffling.

```python
data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
```

### 4. Preprocessing the Data

To improve model training, we normalize the image data by scaling the pixel values between 0 and 1.

```python
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

data = data.map(normalize_img)
```

### 5. Splitting the Data

We split the data into training (70%), validation (20%), and test (10%) sets:

```python
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2) + 1
test_size = int(len(data) * 0.1) + 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)
```

### 6. Building the CNN Model

We construct a simple CNN model using TensorFlow Keras. The model consists of several convolutional layers followed by max-pooling layers, and finally, dense layers for classification.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(16, (3, 3), strides=1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), strides=1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), strides=1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
```

### 7. Training the Model

We use `TensorBoard` callbacks to visualize training progress. The model is trained for 20 epochs using the training and validation data.

```python
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
```

### 8. Plotting Training and Validation Results

After training the model, we visualize the loss and accuracy curves for both training and validation data.

```python
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()
```

### 9. Model Evaluation

Finally, the model is evaluated on the test data. We use precision, recall, and accuracy metrics to assess the model's performance.

```python
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(pre.result(), re.result(), acc.result())
```


## Conclusion

This emotion classification model leverages CNNs for image classification. It has been trained and evaluated on a dataset with images of happy and sad faces. The model's accuracy can be further improved by experimenting with different architectures, hyperparameters, or augmenting the dataset.

---

