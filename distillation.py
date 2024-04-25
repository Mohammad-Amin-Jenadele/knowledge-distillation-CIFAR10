import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import ops

# ------ 


class Distillator(keras.Model):
    def __init__(self, teacher, student, alpha, temperature):
        """_summary_

        Args:
            teacher (_type_): _description_
            student (_type_): _description_
            alpha (_type_): _description_
            tau (_type_): _description_
        
        Exeption
        """
        super().__init__()
        self.teacher = teacher 
        self.student = student
        self.alpha = alpha    
        self.temperature = temperature   
        
        def compile(self,optimizer,
                    metrics,
                    student_loss,
                    distillator_loss,
                    ):
            """_summary_

            Args:
                optimizer (_type_): _description_
                metrics (_type_): _description_
                student_loss_fn (_type_): _description_
                distillator_loss_fn (_type_): _description_
            """
            
            super().compile(optimizer=optimizer, metrics=metrics)
            self.student_loss = student_loss
            self.distillator_loss = distillator_loss
            
        def calculate_loss(self, X, Y, y_prediction):
            """_summary_

            Args:
                X (_type_): _description_
                Y (_type_): _description_
                y_prediction (_type_): _description_

            Returns:
                _type_: _description_
            """
            teacher_prediction = self.teacher(X, training=False)
            student_loss = self.student_loss(Y, y_prediction)
            
            distillation_loss = self.distillator_loss(ops.softmax(teacher_prediction / self.temperature),
                                      ops.softmax(y_prediction, self.temperature),
                                      ) * (self.temperature ** 2)
            
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            
            return loss 
            
            