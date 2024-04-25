import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers, ops


class Distillator(keras.Model):
    def __init__(self, teacher: keras.Model, student: keras.Model, alpha: float, temperature: float):
        """
        Initializes the Distillator class which implements knowledge distillation from a teacher model to a student model.

        Args:
            teacher (keras.Model): The teacher model from which knowledge is distilled.
            student (keras.Model): The student model that learns from the teacher.
            alpha (float): The weighting factor for the student's loss relative to the distillation loss.
            temperature (float): The temperature factor used in softening probability distributions, facilitating distillation.
        """
        super().__init__()
        self.teacher = teacher 
        self.student = student
        self.alpha = alpha    
        self.temperature = temperature   

    def compile(self, optimizer: keras.optimizers.Optimizer, metrics: list, student_loss_: tf.keras.losses.Loss, distillator_loss: tf.keras.losses.Loss):
        """
        Configures the model for training by setting the optimizer, metrics, and loss functions.

        Args:
            optimizer (keras.optimizers.Optimizer): The optimizer to use during training.
            metrics (list): The list of metrics to be evaluated by the model during training and testing.
            student_loss_ (tf.keras.losses.Loss): The loss function that measures how well the student model is doing with respect to the true labels.
            distillator_loss (tf.keras.losses.Loss): The loss function used for distillation, comparing the softened outputs of the teacher and student.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_ = student_loss_
        self.distillator_loss = distillator_loss

    def calculate_loss(self, X: np.ndarray, Y: np.ndarray, y_prediction: np.ndarray) -> tf.Tensor:
        """
        Calculate the combined student and distillation losses.

        Args:
            X (np.ndarray): Input data to the teacher and student models.
            Y (np.ndarray): True labels corresponding to X for the student's learning.
            y_prediction (np.ndarray): The predictions made by the student model on X.

        Returns:
            tf.Tensor: The weighted sum of student and distillation losses.
        """
        teacher_prediction = self.teacher(X, training=False)
        student_loss = self.student_loss_(Y, y_prediction)
        
        distillation_loss = self.distillator_loss(ops.softmax(teacher_prediction / self.temperature),
                                    ops.softmax(y_prediction, self.temperature),
                                    ) * (self.temperature ** 2)
        
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        return loss 
            
        def call(self, x: tf.Tensor) -> tf.Tensor:
            """
            Forward pass for the Distillator model, utilizing only the student model.

            Args:
                x (tf.Tensor): Input tensor containing features.

            Returns:
                tf.Tensor: Output tensor from the student model after processing input 'x'.
            """
            return self.student(x)