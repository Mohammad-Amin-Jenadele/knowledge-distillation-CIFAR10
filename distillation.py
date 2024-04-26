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

    def compile(self, optimizer: keras.optimizers.Optimizer, metrics: list, student_loss: tf.keras.losses.Loss, distillation_loss: tf.keras.losses.Loss):
        """
        Configures the model for training by setting the optimizer, metrics, and loss functions.

        Args:
            optimizer (keras.optimizers.Optimizer): The optimizer to use during training.
            metrics (list): The list of metrics to be evaluated by the model during training and testing.
            student_loss (tf.keras.losses.Loss): The loss function that measures how well the student model is doing with respect to the true labels.
            distillation_loss (tf.keras.losses.Loss): The loss function used for distillation, comparing the softened outputs of the teacher and student.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss = student_loss
        self.distillation_loss = distillation_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            teacher_predictions = self.teacher(x, training=False)

            student_loss = self.student_loss(y, student_predictions)
            distillation_loss = self.distillation_loss(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1)
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, student_predictions)
        return {m.name: m.result() for m in self.metrics}
            
            