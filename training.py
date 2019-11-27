import numpy as np
import random
import pickle
import time
import os

from config import *
from environment import PacmanEnv
from network import Network
from collections import deque
from threading import Thread

class Training(object):

    def __init__(self, best_network):
        print('Initializing/Loading network...')
        # init memory
        self.memory = deque(maxlen=MEMORY_SIZE)
        self._load_memory()
        # init the best network, loaded from disk
        self.best_network = best_network
        # init the current network for training and make it the same as the best network
        self.current_network = Network('Current')
        self.current_network.replace_by(self.best_network)
        # init the test network for comparison
        self.test_network = Network('Test')
        self.self_play_switch = True
        self.fit_switch = True
        self.comparison_switch = False

    def _load_memory(self):
        print('Loading memory...')
        filepath = '{}/models/pacman_agent_memory.dump'.format(ROOT_PATH)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as file_handler:
                self.memory = pickle.load(file_handler)
                file_handler.close()
            print('Loaded {} data into memory'.format(len(self.memory)))

    def _save_memory(self):
        filepath = '{}/models/pacman_agent_memory.dump'.format(ROOT_PATH)
        with open(filepath, 'wb+') as file_handler:
            pickle.dump(self.memory, file_handler)
            file_handler.close()

    def run_episode(self):
        steps = []
        env = PacmanEnv()
        state = env.get_current_state()
        reward = 0
        result = 0

        while True:
            # with the network, predict the value of the state and the
            # probability distribution of the actions from that state
            v, pi = self.current_network.predict(state)
            pi = pi[0]
            # add state, PI and placeholder for value in memory
            steps.append([state, pi, None])
            # choose an action pased on pi
            action = np.random.choice(len(pi), p=pi)
            # take the action
            state, reward, result = env.step(action)

            if result != 0:
                for i in range(len(steps)):
                    steps[i][2] = result
                break

        for step in steps:
            self.memory.append(step)

    def _prepare_training_data(self, samples):
        print('preparing training data...')
        inputs = []
        targets_w = []
        targets_pi = []
        for sample in samples:
            inputs.append(sample[0])
            targets_pi.append(sample[1])
            targets_w.append(sample[2])
        return np.vstack(inputs), [np.vstack(targets_w), np.vstack(targets_pi)]

    def compete_for_best_network(self, new_network, best_network):
        '''
            Each network is going to play a few games by itself, and the network
            with the highest accumulated score wins
        '''
        print('Competing for best network...')
        env = PacmanEnv()
        players = [new_network, best_network]
        players_scores = [0, 0]
        for i in range(len(players)):
            for _ in range(COMPETE_GAME_NUM):
                env.reset()
                state = env.get_current_state()

                while True:
                   v, pi = players[i].predict(state)
                   pi = pi[0]
                   action = np.random.choice(len(pi), p=pi)
                   state, reward, result = env.step(action)
                   players_scores[i] += reward

                   if result != 0:
                       break
        print('New network score:', players_scores[0])
        print('Best network score:', players_scores[1])
        if players_scores[0] > players_scores[1]:
            print('New network wins!')
        else:
            print('Best network stays champion!')
        return players_scores[0] > players_scores[1]

    def self_play(self):
        while self.self_play_switch:
            self.run_episode()
            # print('generated new samples. total sample number is {}'.format(len(self.memory)))
            self._save_memory()

    def fit(self):
        while self.fit_switch:
            if len(self.memory) >= MIN_MEMORY_SIZE_BEFORE_FIT:
                inputs, targets = self._prepare_training_data(random.sample(self.memory, SAMPLE_SIZE))
                val_inputs, val_targets = self._prepare_training_data(random.sample(self.memory, SAMPLE_SIZE))
                print('Fitting the network...')
                self.current_network.fit(
                    inputs=inputs, targets=targets, epochs=EPISODE_NUM, batch_size=BATCH_SIZE,
                    validation_data=(val_inputs, val_targets)
                )
                print('Fitted the network')
                self.comparison_switch = True
            time.sleep(FIT_INTERVAL)

    def comparison(self):
        print('Starting a new comparison...')
        while not self.comparison_switch:
            time.sleep(COMPARISON_INTERVAL)
        update_count = 0
        long_wait = COMPARISON_LONG_WAIT

        while self.comparison_switch:
            if update_count >= ITERATION_NUM:
                break
            elif len(self.memory) >= MIN_MEMORY_SIZE_BEFORE_FIT:
                if long_wait > 0:
                    print('Comparison thread will fall into a long sleep, to give fit thread time to fit network...')
                    time.sleep(long_wait)
                    long_wait = 0
                # copy the current network to the test network for comparison
                self.test_network.replace_by(self.current_network)
                is_update = self.compete_for_best_network(self.test_network, self.best_network)
                if is_update:
                    print('Updating best network...')
                    update_count += 1
                    self.best_network.save()
                    # clean up memory to get more precise data
                    for _ in range(int(len(self.memory) * MEMORY_CLEAN_RATE)):
                        self.memory.pop()
                    long_wait = COMPARISON_LONG_WAIT
                self.current_network.save()
            else:
                time.sleep(COMPARISON_INTERVAL)
            self.self_play_switch = False
            self.fit_switch = False

    def train(self):
        print('Initializing training...')
        self_play_thread = Thread(target=self.self_play)
        fit_thread = Thread(target=self.fit)
        comparison_thread = Thread(target=self.comparison)

        self_play_thread.setDaemon(True)
        fit_thread.setDaemon(True)
        comparison_thread.setDaemon(True)

        self_play_thread.start()
        fit_thread.start()
        comparison_thread.start()

        self_play_thread.join()
        fit_thread.join()
        comparison_thread.join()


if __name__ == '__main__':
    training_flag = str(input('Would you like to train the network before you test it? (answer Y or N): ')).upper() == 'Y'
    best_network = Network('Best')

    if training_flag:
        training = Training(best_network)
        training.train()

    env = PacmanEnv()
    total_score = 0

    while True:
        env.print()
        state = env.get_current_state()
        v, pi = best_network.predict(state)
        pi = pi[0]
        action = np.random.choice(len(pi), p=pi)
        print('action:', ACTION_DICT[action])
        print('\n')

        state, reward, result = env.step(action)
        total_score += reward

        if result != 0:
            break
    print('Game Over, Total Score:', total_score)

