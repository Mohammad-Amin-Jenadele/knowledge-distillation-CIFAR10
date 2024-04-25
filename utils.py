# Package import
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




# precision function
def precision(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision


# recall function
def recall(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

# Plotting confusion matrix
def plot_confusion_matrix(model , x , y ):

    y_pred = model.predict(x) 
    y_pred_plot=np.argmax(y_pred, axis=1)
    y_test_plot=np.argmax(y, axis=1)

    cm = confusion_matrix(y_test_plot, y_pred_plot)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
    

# Defining load_cifar10 function that gets val_ratio as input and returns (x_train, y_train), (x_test, y_test), (x_val, y_val)
def load_cifar10(val_ratio):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One hotting the labels
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Split training set into training and validation sets
    split_index = int(len(x_train) * (1 - val_ratio))
    x_val = x_train[split_index:]
    y_val = y_train[split_index:]
    x_train = x_train[:split_index]
    y_train = y_train[:split_index]

    return (x_train, y_train), (x_test, y_test), (x_val, y_val)

# Defining sample_plot function which randomly plot 10 images of dataset with their corresponding labels (y is one hotted)
def sample_plot(x , y , num_to_class_dic):
    plt.figure(figsize=(12, 6))
    random_number = np.random.randint(0 , len(x) , 10)
    for i in range(10):
        img_x = x[random_number[i]]
        ax = plt.subplot(2, 5 , i + 1)
        plt.imshow(img_x)
        plt.title(num_to_class_dic[np.argmax(y[random_number[i]])],fontsize = 15)
        plt.axis("off")
    plt.show()


# Defining MobileNetV2_model that gives MobileNetV2 model
def MobileNetV2_model(weights , input_shape):
    model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights = weights
    input_shape = input_shape
    )
    return model

# Defining ResNet50_model that gives ResNet50 model
def ResNet50_model():
    model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(224, 224, 3),
    weights="imagenet",
)
    return model