import numpy as np
import os
from config import *
from levels.level import load_level
import random

class PacmanEnv:
    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height
        self.walls, self.food, self.entities, self.current_location, self.enemies, self.enemies_box, self.enemies_doors, self.special_passages = load_level()
        self.food_count = np.sum(self.food)
        self.steps = 0
        self.max_steps = MAX_STEPS
        print(self.food_count)

    def get_current_state(self):
        state = np.zeros((HEIGHT, WIDTH, CHANNELS), dtype=np.float32)
        state[:, :, 0] = self.food
        state[:, :, 1] = self.entities
        return state

    def step(self, action):
        if self.steps >= self.max_steps:
            return self.get_current_state(), -1, -1
        self.steps += 1

        prev_i = self.current_location[0]
        prev_j = self.current_location[1]
        i = prev_i
        j = prev_j

        in_special_location = False

        for passage in self.special_passages:
            if passage[0] == self.current_location and action == passage[1]:
                in_special_location = True
                i = passage[2][0]
                j = passage[2][1]
                break

        if not in_special_location:
            if action == LEFT:
                j -= 1
            elif action == RIGHT:
                j += 1
            elif action == UP:
                i -= 1
            elif action == DOWN:
                i += 1

        if i not in range(HEIGHT) or j not in range(WIDTH) or self.walls[i, j] == 1:
            reward = -1
            result = 0
            state = self.get_current_state()
            return state, reward, result

        # update entities
        self.entities[prev_i, prev_j] = 0
        if self.entities[i, j] != -1:
            self.entities[i, j] = 1
        self.move_enemies()

        # update food
        reward = self.food[i, j]
        self.food[i, j] = 0
        if self.food[prev_i, prev_j] != 0:
            self.food[prev_i, prev_j] = 0

        # update current location
        self.current_location = [i, j]

        # check if game over
        result = self.get_result()

        state = self.get_current_state()
        reward -= 0.25 # to make the agent not waste time

        return state, reward, result

    def _still_in_box(self, enemy_i, enemy_j):
        top_left = self.enemies_box['top left']
        width = self.enemies_box['width']
        height = self.enemies_box['height']
        return top_left[0] <= enemy_i < top_left[0] + height and top_left[1] <= enemy_j < top_left[1] + width

    def _get_possible_enemy_moves(self, enemy_i, enemy_j):
        possible_actions = []
        for action in self.get_all_next_actions():
            new_i = enemy_i
            new_j = enemy_j
            if action == LEFT:
                new_j -= 1
            elif action == RIGHT:
                new_j += 1
            elif action == UP:
                new_i -= 1
            elif action == DOWN:
                new_i += 1
            try:
                if new_i in range(HEIGHT) and new_j in range(WIDTH) and self.walls[new_i, new_j] != 1 and self.entities[new_i, new_j] != -1:
                    possible_actions.append(action)
            except Exception:
                print('error:', new_i, new_j)
                exit()
        return possible_actions


    def move_enemies(self):
        '''Moves the enemies in random possible locations'''
        for i in range(len(self.enemies)):
            enemy = self.enemies[i]
            if self._still_in_box(enemy[0], enemy[1]):
                # Check if there is available space at the door
                for door in self.enemies_doors:
                    if self.entities[door[0], door[1]] != -1:
                        # If the is space at the door, move the enemy to that space
                        self.entities[enemy[0], enemy[1]] = 0

                        self.enemies[i][0] = door[0]
                        self.enemies[i][1] = door[1]

                        # update entities
                        self.entities[door[0], door[1]] = -1
                        break
            else:
                # Get the possible moves that the enemy can make
                possible_actions = self._get_possible_enemy_moves(enemy[0], enemy[1])
                if len(possible_actions) > 0:
                    action = random.choice(possible_actions)
                    new_i = enemy[0]
                    new_j = enemy[1]
                    if action == LEFT:
                        new_j -= 1
                    elif action == RIGHT:
                        new_j += 1
                    elif action == UP:
                        new_i -= 1
                    elif action == DOWN:
                        new_i += 1
                    # update enemy location and entities
                    self.entities[enemy[0], enemy[1]] = 0
                    self.enemies[i][0] = new_i
                    self.enemies[i][1] = new_j
                    self.entities[new_i, new_j] = -1

    def get_result(self):
        '''Returns 1 if you win the game, -1 if you lose, and 0 otherwise'''
        result = 0

        # Check if you win, meaning there is no food to eat
        if np.sum(self.food) == 0:
            result = 1
        # If you lose, you are in the same location as an enemy
        elif self.entities[self.current_location[0], self.current_location[1]] == -1:
            result = -1
        else:
            result = 0
        return result

    def is_valid(self, action, state=None):
        if state is None:
            state = self.get_current_state()
        walls = state[:, :, 0]
        entities = state[:, :, 2]

        current_location = np.where(entities == 1)
        i = current_location[0]
        j = current_location[1]

        if action == LEFT:
            j -= 1
        elif action == RIGHT:
            j += 1
        elif action == UP:
            i -= 1
        elif action == DOWN:
            i += 1

        if i not in range(HEIGHT) or j not in range(WIDTH) or walls[i, j] == 1:
            return False
        else:
            return True

    def get_all_next_actions(self):
        return [LEFT, RIGHT, UP, DOWN]

    def get_valid_actions(self):
        current_location = self.current_location


        actions = []
        for action in self.get_all_next_actions():
            try:
                i = self.current_location[0]
                j = self.current_location[1]

                if action == LEFT:
                    j -= 1
                elif action == RIGHT:
                    j += 1
                elif action == UP:
                    i -= 1
                elif action == DOWN:
                    i += 1

                if i in range(HEIGHT) and j in range(WIDTH) and self.walls[i, j] != 1:
                    actions.append(action)
            except:
                print('error:', current_location)
                self.print()
                exit()
        return actions

    def reset(self):
        self.walls, self.food, self.entities, self.current_location, self.enemies, self.enemies_box, self.enemies_doors, self.special_passages = load_level()
        self.steps = 0



    def to_str(self):
        string = ''
        for i in range(self.height):
            for j in range(self.width):
                if self.entities[i, j] == -1:
                    string += 'e'
                elif self.entities[i, j] == 1:
                    string += 'p'
                elif self.walls[i, j] == 1:
                    string += '*'
                elif self.food[i, j] == SMALL_FOOD_VALUE:
                    string += '.'
                elif self.food[i, j] == BIG_FOOD_VALUE:
                    string += 'F'
                else:
                    string += ' '
                string += '\t'
            string += os.linesep

        return string

    def print(self):
        print(self.to_str())



def main():
    env = PacmanEnv()
    for _ in range(7):
        env.print()
        print('\n')
        state, reward, result = env.step(LEFT)
    for _ in range(9):
        env.print()
        print('\n')
        state, reward, result = env.step(UP)
    for _ in range(10):
        env.print()
        print('\n')
        state, reward, result = env.step(LEFT)
    for _ in range(10):
        env.print()
        print('\n')
        state, reward, result = env.step(RIGHT)

if __name__ == '__main__':
    main()