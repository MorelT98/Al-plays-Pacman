import os
import tensorflow as tf
from keras.layers import Input, Conv2D, regularizers, BatchNormalization, LeakyReLU, add, Flatten, Dense
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint

from config import *
from environment import PacmanEnv

class Network(object):
    def __init__(self, name, input_dim=(HEIGHT, WIDTH, CHANNELS),
                 output_dim = NUM_ACTIONS, layers_metadata = NETWORK_METADATA,
                 reg_const = REG_CONST, learning_rate = LEARNING_RATE,
                 root_path = ROOT_PATH):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_metadata = layers_metadata
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.root_path = root_path
        self.graph = tf.get_default_graph()

        with self.graph.as_default():
            self.model = self._build_model()
            self.tensorboard = TensorBoard(log_dir='{}/logs/pacman_agent_v0'.format(self.root_path),
                                           histogram_freq=0,
                                           write_graph=None, write_images=None)
            self.checkpoint = ModelCheckpoint(
                filepath='{}/models/pacman_agent_v0_{}_model_chackpoint.h5'.format(root_path, self.name),
                save_best_only=True
            )

    def _add_conv_block(self, prev_block, filters, kernel_size):
        block = Conv2D(filters=filters, kernel_size=kernel_size, data_format='channels_last',
                       padding='same', use_bias=False, activation='linear',
                       kernel_regularizer=regularizers.l2(self.reg_const))(prev_block)
        block = BatchNormalization(axis=1)(block)
        block = LeakyReLU()(block)
        return block

    def _add_residual_block(self, prev_block, filters, kernel_size):
        block = self._add_conv_block(prev_block=prev_block, filters=filters, kernel_size=kernel_size)
        block = Conv2D(filters=filters, kernel_size=kernel_size, data_format='channels_last', padding='same',
                        use_bias=False, activation='linear',
                        kernel_regularizer=regularizers.l2(self.reg_const))(block)
        block = BatchNormalization(axis=1)(block)
        block = add([prev_block, block])
        block = LeakyReLU()(block)
        return block

    def _add_value_head(self, prev_block):
        block = Conv2D(filters=1, kernel_size=(1, 1), data_format='channels_last', padding='same',
                       use_bias=False, activation='linear',
                       kernel_regularizer=regularizers.l2(self.learning_rate))(prev_block)
        block = BatchNormalization(axis=1)(block)
        block = LeakyReLU()(block)
        block = Flatten()(block)
        block = Dense(units=20, use_bias=False, activation='linear',
                      kernel_regularizer=regularizers.l2(self.reg_const))(block)
        block = LeakyReLU()(block)
        block = Dense(units=1, use_bias=False, activation='tanh',
                      kernel_regularizer=regularizers.l2(self.reg_const), name='value_head')(block)
        return block

    def _add_policy_head(self, prev_block):
        block = Conv2D(filters=2, kernel_size=(1, 1), data_format='channels_last', padding='same',
                       use_bias=False, activation='linear',
                       kernel_regularizer=regularizers.l2(self.reg_const))(prev_block)
        block = BatchNormalization(axis=1)(block)
        block = LeakyReLU()(block)
        block = Flatten()(block)
        block = Dense(units=self.output_dim, use_bias=False, activation='softmax',
                      kernel_regularizer=regularizers.l2(self.reg_const), name='policy_head')(block)
        return block

    def _build_model(self):
        model = self.load()
        if model is None:
            main_input = Input(shape=self.input_dim, name='main_input')
            block = self._add_conv_block(prev_block=main_input, filters=self.layers_metadata[0]['filters'],
                                         kernel_size=self.layers_metadata[0]['kernel_size'])
            if len(self.layers_metadata) > 1:
                for metadata in self.layers_metadata[1:]:
                    block = self._add_residual_block(prev_block=block, filters=metadata['filters'],
                                                     kernel_size=self.layers_metadata[0]['kernel_size'])
            value_head = self._add_value_head(prev_block=block)
            policy_head = self._add_policy_head(prev_block=block)
            model = Model(inputs=main_input, outputs=[value_head, policy_head])
            model.compile(loss={'value_head':'mse', 'policy_head':'categorical_crossentropy'},
                          optimizer=SGD(lr=self.learning_rate), loss_weights={'value_head': 0.5, 'policy_head': 0.5},
                          metrics={'value_head': 'mse', 'policy_head': 'acc'})

        # The following 3 protected methods make the model prone to be shared by multithreads
        model._make_predict_function()
        model._make_train_function()
        model._make_test_function()

        return model

    def save(self):
        self.model.save(filepath='{}/models/pacman_agent_v0_{}_model.h5'.format(self.root_path, self.name),
                        include_optimizer=True, overwrite=True)

    def load(self):
        file_path = '{}/models/pacaman_agent_v0_{}_model.h5'.format(self.root_path, self.name)
        if os.path.exists(file_path):
            model = load_model(file_path)
        else:
            model = None
        return model

    def replace_by(self, another_network):
        with self.graph.as_default():
            self.model.set_weights(another_network.model.get_weights())

    def predict(self, inputs):
        with self.graph.as_default():
            return self.model.predict(x=inputs, steps=1)

    def fit(self, inputs, targets, epochs, batch_size, validation_split=0.0, validation_data=None):
        with self.graph.as_default():
            return self.model.fit(x=inputs, y=targets, epochs=epochs, batch_size=batch_size,
                                  shuffle=True, verbose=1, validation_split=validation_split,
                                  validation_data=validation_data,
                                  callbacks=[self.tensorboard, self.checkpoint])


def main():
    network = Network('test')
    env = PacmanEnv()
    state = env.get_current_state()
    v, pi = network.predict(state)
    print(v[0][0])
    print(pi[0])

if __name__ == '__main__':
    main()