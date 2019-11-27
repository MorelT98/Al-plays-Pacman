'''Trains an agent with Policy Gradients on Pong, using tensorflow/keras'''
import numpy as np
from environment import PacmanEnv
from config import *
import pickle

import tensorflow as tf

class PolicyModel:
    def __init__(self, K=NUM_ACTIONS, network_metadata=NETWORK_METADATA):
        '''Creates Model with input size D and output size K'''
        self.network_metadata = network_metadata # info about the hidden layers
        self.current_best_score = float('-inf')

        # inputs and targets
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, HEIGHT, WIDTH, CHANNELS), name='X')
        self.actions = tf.placeholder(dtype=tf.int32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(dtype=tf.float32, shape=(None,), name='advantages')

        # create hidden layers
        self.layers = []
        conv = self.X
        for metadata in self.network_metadata:
            conv = tf.layers.conv2d(
                inputs=conv,
                filters=metadata['filters'],
                kernel_size=metadata['kernel_size'],
                padding='same',
                activation=tf.nn.relu,
            )

        # final layers (dense layers)
        flat = tf.reshape(conv, [-1, DENSE_LAYER_SHAPE])
        dense = tf.layers.dense(inputs=flat, units=20, activation=tf.nn.relu)

        # logits layer
        logits = tf.layers.dense(inputs=dense, units=K)

        p_a_given_s = tf.nn.softmax(logits, name='probabilities')
        self.predict_op = p_a_given_s

        # get the probabilities of the actions selected
        # e.g.: p_a_given_s = [0.25, 0.4, 0.35], actinos = [1, 1, 0, 0, 2, 1]
        # p_a_given_s * tf.one_hot(actions, 3) =
        # [[0, 0.40, 0],
        #  [0, 0.40, 0],
        #  [0.25, 0, 0],
        #  [0.25, 0, 0],
        #  [0, 0, 0.35],
        #  [0, 0.40, 0]]
        # using tf.reduce_sum gives [0.4, 0.4, 0.25, 0.25, 0.35, 0.4]
        selected_probs = tf.log(
            tf.reduce_sum(
                p_a_given_s*tf.one_hot(self.actions, K),
                reduction_indices=[1]
            )
        )
        self.selected_probs = selected_probs

        # Policy gradient cost function formula
        cost = -tf.reduce_sum(self.advantages * selected_probs)
        self.cost = cost

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.current_learning_rate = LEARNING_RATE
        self.train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(cost)

        self.saver = tf.train.Saver()

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, actions, advantages):
        # size check
        X = np.atleast_3d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)

        # print(self.predict_op.eval(feed_dict={self.X: X, self.actions: actions, self.advantages:advantages}))
        self.session.run(
            self.train_op,
            feed_dict={self.X: X, self.actions: actions, self.advantages:advantages, self.learning_rate:self.current_learning_rate}
        )

    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X:X})

    def sample_action(self, X, possible_actions=ALL_ACTIONS):
        p = self.predict(X)[0]
        return np.random.choice(len(p), p=p)

    def redistribute_probabilities(self, p, possible_actions):
        '''Gives impossible actions a prob of 0 and redistributes their original
         probabilities into the possible actions'''
        if possible_actions == ALL_ACTIONS:
            return

        # get the original impossible probabilities
        for action in ALL_ACTIONS:
            if action not in possible_actions:
                p[action] = 0   # give impossible actions a probability of 0

        prob_left = np.sum(p)
        if prob_left > 0:
            # redistribute those probabilities proportionally into the possible actions
            for action in possible_actions:
                p[action] /= prob_left



    def save(self):
        save_path = self.saver.save(self.session, 'models/pacman_pg_model.ckpt')
        pickle.dump(self.current_best_score, open('best_score.pkl', 'wb'))

    def load(self):
        self.saver.restore(self.session, 'models/pacman_pg_model.ckpt')
        try:
            self.current_best_score = pickle.load(open('best_score.pkl', 'rb'))
            print(self.current_best_score)
        except:
            pass

def play_one_mc(env, model, render=False):

    env.reset()
    observation = env.get_current_state()
    done = False
    total_reward = 0

    reward = 0

    states, actions, rewards = [], [], []


    while not done:
        if render:
            env.print()


        possible_actions = env.get_valid_actions()
        try:
            assert possible_actions != []
        except:
            env.print()
            print('error: No possible action')
            exit()

        # predict probabilities, then sample action
        action = model.sample_action([observation], possible_actions)

        # save (s, a, r)
        states.append(observation)
        actions.append(action)
        rewards.append(reward)


        observation, reward, result = env.step(action)
        done = (result != 0)
        total_reward += reward

    # save the last (s, a, r)

    action = np.random.choice(range(NUM_ACTIONS))
    states.append(observation)
    actions.append(action)
    rewards.append(reward)


    # calculate returns and advantages
    returns = []
    advantages = []

    G = 0
    for s, r in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        advantages.append(G)
        G = r + DISCOUNT_FACTOR * G
    returns.reverse()
    advantages.reverse()

    # normalize advantages
    advantages = np.array(advantages)
    advantages -= np.mean(advantages)
    advantages /= np.std(advantages)

    model.partial_fit(states, actions, advantages)

    return total_reward

resume = True
test = False

def main():
    env = PacmanEnv()
    model = PolicyModel()
    session = tf.InteractiveSession()
    model.set_session(session)
    iter = 0

    if resume:
        print('loading model...')
        model.load()
        print('model loaded.')
    else:
        init = tf.global_variables_initializer()
        model.session.run(init)

    if test:
        play_one_mc(env, model, render=True)
        return

    episode_number = 0

    running_reward = None

    while True:
        iter += 1
        total_reward = play_one_mc(env, model, render=False)
        episode_number += 1

        if iter == BATCH_SIZE:
            iter = 0


        if episode_number % 10 == 0:
            print('The current network and the last saved network will compete.')
            current_model_reward = 0
            for i in range(20):
                current_model_reward += play_one_mc(env, model, render=False)
            print('End of competition. Current network score: {}. Last network score: {}'.format(current_model_reward, model.current_best_score), end=' ')
            if current_model_reward > model.current_best_score - 10:
                print('The current network wins!')
                print('Saving model...')
                model.current_best_score = current_model_reward
                model.save()
                print('Model saved.')
                running_reward = None
            else:
                print('The old network wins!')
                print('Slowing down learning...')
                model.load()
                model.current_learning_rate /= 10

        running_reward = total_reward if running_reward is None else running_reward * 0.99 + total_reward * 0.01
        print(episode_number, 'resetting env. episode reward total was %f. running mean: %f' % (total_reward, running_reward))

if __name__ == '__main__':
    main()







