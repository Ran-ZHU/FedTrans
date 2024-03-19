import pandas as pd
import csv
import os
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from sklearn.preprocessing import StandardScaler

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import regularizers
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
from numpy import linalg as LA


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class nn_em:
    def __init__(self):
        print("VEM model initialized")

    def define_nn(self, m, n_neurons_1, n_neurons_2, learning_rate, nb_hidden_layer=2, hidden=True):
        classifier = Sequential()
        if hidden == True:
            # First Hidden Layer
            layer0 = Dense(n_neurons_1, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.03, seed=98765), input_dim=m)
            classifier.add(layer0)
            nb = 1
            while (nb < nb_hidden_layer):
                layer_nb = Dense(n_neurons_2, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.03, seed=98765))
                classifier.add(layer_nb)
                nb += 1
        # Output Layer
        layer1 = Dense(1, activation='sigmoid', kernel_initializer=initializers.random_normal(stddev=0.03, seed=98765), \
                       kernel_regularizer=regularizers.l2(0.5))
        classifier.add(layer1)
        # Compiling the neural network
        sgd = optimizers.gradient_descent_v2.SGD(learning_rate=learning_rate, clipvalue=0.5)
        classifier.compile(optimizer=sgd, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])


        return classifier

    def nn_pzi_test_val(self, classifier, social_features, prob_e_step, steps):
        classifier.fit(social_features, prob_e_step, epochs=steps, verbose=0)
        theta_i = classifier.predict(social_features)
        losses = classifier.evaluate(social_features, prob_e_step, verbose=0)
        return theta_i.astype(np.float32), losses, classifier

    def train_m_step(self, classifier, social_features, prob_e_step, steps, total_epochs):
        theta_i = prob_e_step.copy()  # 11x1
        # weights = np.array([])
        iter = 0
        old_theta_i = np.zeros((social_features.shape[0], 1))
        epsilon = 1e-3
        while (LA.norm(theta_i - old_theta_i) > epsilon) and (iter < total_epochs):
            old_theta_i = theta_i.copy()
            theta_i, losses, classifier = self.nn_pzi_test_val(classifier, social_features, prob_e_step, steps)
            iter += 1

        # print  ('>>>M-step | iter:', iter)

        return theta_i, classifier, iter




