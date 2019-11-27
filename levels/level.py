import json
import pygame
import numpy as np
from config import *

def swap(array, i, j):
    temp = array[i]
    array[i] = array[j]
    array[j] = temp

def load_level(level_name='level-1.json'):
    with open('./levels/{}'.format(level_name)) as jsonData:
        data = json.load(jsonData)
        width = data['width']
        height = data['height']

        walls = np.full((height, width), 0, dtype=np.float32)
        food = np.full((height, width), 0, dtype=np.float32)
        entities = np.full((height, width), 0, dtype=np.float32)

        # load walls
        for wall_box in data['walls']['wall boxes']:
            top_left = wall_box[0]
            wall_width = wall_box[1]
            wall_height = wall_box[2]
            for j in range(top_left[0], top_left[0] + wall_width):
                for i in range(top_left[1], top_left[1] + wall_height):
                    walls[i, j] = 1

        for wall_row in data['walls']['wall rows']:
            start = wall_row[0]
            end = wall_row[1]
            i = start[1]
            for j in range(start[0], end[0] + 1):
                walls[i, j] = 1

        for wall_column in data['walls']['wall columns']:
            start = wall_column[0]
            end = wall_column[1]
            j = start[0]
            for i in range(start[1], end[1] + 1):
                walls[i, j] = 1

        # load food
        for food_row in data['food']['food rows']:
            start = food_row[0]
            end = food_row[1]
            i = food_row[0][1]
            for j in range(start[0], end[0] + 1):
                food[i, j] = 2

        for food_row in data['food']['food columns']:
            start = food_row[0]
            end = food_row[1]
            j = food_row[0][0]
            for i in range(start[1], end[1] + 1):
                food[i, j] = 2

        # load big food
        for big_food in data['food']['big food']:
            j = big_food[0]
            i = big_food[1]
            food[i, j] = 1

        # load enemies
        enemies = []
        for enemy in data['enemies']:
            j = enemy[0]
            i = enemy[1]
            enemies.append([i, j])
            entities[i, j] = -1

        # load enemy box
        enemies_box = {}
        j = data['enemies box'][0][0]
        i = data['enemies box'][0][1]
        enemies_box['top left'] = [i, j]
        enemies_box['width'] = data['enemies box'][1]
        enemies_box['height'] = data['enemies box'][2]

        # load enemies doors
        enemies_doors = []
        for door in data['enemies doors']:
            swap(door, 0, 1)
            enemies_doors.append(door)

        # load special passages
        special_passages = []
        for passage in data['special passages']:
            swap(passage[0], 0, 1)
            swap(passage[2], 0, 1)
            special_passages.append(passage)

        # load pacman
        j = data['pacman'][0]
        i = data['pacman'][1]
        pacman = [i, j]
        entities[i, j] = 1

    return walls, food, entities, pacman, enemies, enemies_box, enemies_doors, special_passages
