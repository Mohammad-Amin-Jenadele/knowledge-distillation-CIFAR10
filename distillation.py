import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import ops

# ------ 


class Distillator(keras.Model):
    def __init__(self, teacher, student, alpha, tau):
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
        
        def compile(self,optimizer,
                    metrics,
                    student_loss,
                    distillation_loss,
                    ):
            """_summary_

            Args:
                optimizer (_type_): _description_
                metrics (_type_): _description_
                student_loss_fn (_type_): _description_
                distillation_loss_fn (_type_): _description_
            """
            
            super().compile(optimizer=optimizer, metrics=metrics)
            self.student_loss = student_loss
            self.distillation_loss = distillation_loss