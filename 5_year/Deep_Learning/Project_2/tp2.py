# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:40:05 2023

@author: Asus
"""

import shutil
import os
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/content/drive/MyDrive/AP_2')

import pickle
from snake_game import SnakeGame
import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import deque
import random
from PIL import Image
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, BatchNormalization
from tensorflow.keras import regularizers
import math

def training_plot(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.title('Training and validation accuracy ')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.title('Training and validation loss ')
    plt.legend()
    plt.show()
  

def plot_board(file_name,board,text=None):
    file_name = "ztest_" + file_name + ".png"
    plt.figure(figsize=(10,10))
    plt.imshow(board)
    plt.axis('off')
    if text is not None:
        plt.gca().text(3, 3, text, fontsize=45,color = 'yellow')
    fig = plt.gcf()
    plt.show()
    #fig.savefig(file_name,bbox_inches='tight')
    plt.close()

#heuristica 2
def choose_action_h2(snake_head, apple, direction, game):
    apple_y, apple_x = apple[0]
    head_y, head_x = snake_head

    # Calculate distances between head and apple
    distance_x = apple_x - head_x
    distance_y = apple_y - head_y
    
    if direction == 0:
        if apple_y >= head_y:
            # Apple is not up
            if apple_x > head_x:
                action = 1
                snake_head_new = (head_y, head_x + action)
                if snake_head_new in game.snake:
                    #print("1 colisao 0 ")
                    return 0
                else:
                    return 1
            elif apple_x <= head_x:
                action = -1
                snake_head_new = (head_y, head_x + action)
                if snake_head_new in game.snake:
                    #print("2 colisao 0 ")
                    return 0
                else:
                    return -1
        else:
            if abs(distance_y) >= abs(distance_x):
                return 0
            else:
                action = np.sign(distance_x)
                snake_head_new = (head_y, head_x + action)
                if snake_head_new in game.snake:
                    #print("3 colisao 0 ")
                    return 0
                else:
                    print
                    return action
    
    elif direction == 1:
        if apple_x <= head_x:
            # Apple is not right
            if apple_y < head_y:
                action = -1
                snake_head_new = (head_y + action, head_x)
                if snake_head_new in game.snake:
                    #print("1 colisao 1")
                    return 0
                else:
                    return -1
            elif apple_y >= head_y:
                action = 1
                snake_head_new = (head_y + action, head_x)
                if snake_head_new in game.snake:
                    #print("2 colisao 1")
                    return 0
                else:
                    return 1
        else:
            if abs(distance_x) >= abs(distance_y):
                return 0
            else:
                action = np.sign(distance_y)
                snake_head_new = (head_y + action, head_x)
                if snake_head_new in game.snake:
                    #print("3 colisao 1")
                    return 0
                else:
                    return action

    elif direction == 2:
        if apple_y <= head_y:
            # Apple is not down
            if apple_x > head_x:
                action = -1
                snake_head_new = (head_y, head_x - action)
                if snake_head_new in game.snake:
                    #print("1 colisao 2")
                    return 0
                else:
                    return -1
            elif apple_x <= head_x:
               action = 1
               snake_head_new = (head_y, head_x - action)
               if snake_head_new in game.snake:
                   #print("2 colisao 2")
                   return 0
               else:
                   return 1
        else:
            if abs(distance_y) >= abs(distance_x):
                return 0
            else:
                action = np.sign(distance_x)*-1
                snake_head_new = (head_y, head_x - action)
                if snake_head_new in game.snake:
                    #print("3 colisao 2")
                    return 0
                else:
                    return action
    
    else:
        if apple_x >= head_x:
            # Apple is not left
            if apple_y < head_y:
                action = 1
                snake_head_new = (head_y - action, head_x)
                if snake_head_new in game.snake:
                    #print("1 colisao 3")
                    return 0
                else:
                    return 1
            elif apple_y >= head_y:
                action = -1
                snake_head_new = (head_y - action, head_x)
                if snake_head_new in game.snake:
                    #print("2 colisao 3")
                    return 0
                else:
                    return -1
        else:
            if abs(distance_x) >= abs(distance_y):
                return 0
            else:
                action = np.sign(distance_y)*-1
                snake_head_new = (head_y - action, head_x)
                if snake_head_new in game.snake:
                    #print("2 colisao 3")
                    return 0
                else:
                    return action
    
#heuristica 1
def choose_action_h1(snake_head, apple, direction, game):
    apple_y, apple_x = apple[0]
    head_y, head_x = snake_head

    # Calculate distances between head and apple
    distance_x = apple_x - head_x
    distance_y = apple_y - head_y
    action = 0
    
    if direction == 0:
        if apple_y >= head_y:
            # Apple is not up
            if apple_x > head_x:
                action = 1
                snake_head_new = (head_y, head_x + action)
                if snake_head_new in game.snake:
                    #print("1 colisao 0 ")
                    return 0
                else:
                    return 1
            elif apple_x <= head_x:
                action = -1
                snake_head_new = (head_y, head_x + action)
                if snake_head_new in game.snake:
                    #print("2 colisao 0 ")
                    return 0
                else:
                    return -1
    
    elif direction == 1:
        if apple_x <= head_x:
            # Apple is not right
            if apple_y < head_y:
                action = -1
                snake_head_new = (head_y + action, head_x)
                if snake_head_new in game.snake:
                    #print("1 colisao 1")
                    return 0
                else:
                    return -1
            elif apple_y >= head_y:
                action = 1
                snake_head_new = (head_y + action, head_x)
                if snake_head_new in game.snake:
                    #print("2 colisao 1")
                    return 0
                else:
                    return 1

    elif direction == 2:
        if apple_y <= head_y:
            # Apple is not down
            if apple_x > head_x:
                action = -1
                snake_head_new = (head_y, head_x - action)
                if snake_head_new in game.snake:
                    #print("1 colisao 2")
                    return 0
                else:
                    return -1
            elif apple_x <= head_x:
               action = 1
               snake_head_new = (head_y, head_x - action)
               if snake_head_new in game.snake:
                   #print("2 colisao 2")
                   return 0
               else:
                   return 1
    
    else:
        if apple_x >= head_x:
            # Apple is not left
            if apple_y < head_y:
                action = 1
                snake_head_new = (head_y - action, head_x)
                if snake_head_new in game.snake:
                    #print("1 colisao 3")
                    return 0
                else:
                    return 1
            elif apple_y >= head_y:
                action = -1
                snake_head_new = (head_y - action, head_x)
                if snake_head_new in game.snake:
                    #print("2 colisao 3")
                    return 0
                else:
                    return -1
                
    return action
            
        
# gerar imagens com heuristica 1
def snake_examples_images_h1(number_of_games):
    file = open('labels.txt', 'w')
    images = []
    game = SnakeGame(20, 20, border=1)
    step = 0
    for i in range(number_of_games):
        board, reward, done, info = game.reset()
        #plot_board(str(step), board)
        images.append(board)
        done = False
        frame = 0
        score = 0
        while not done:
            step += 1
            score, apple, head, tail, direction = game.get_state()
            action = choose_action_h1(head, apple, direction, game)  # Use the improved heuristic
            #print(action)
            new_board, reward, done, info = game.step(action)
            images.append(new_board)
            board = new_board
            #plot_board(str(step), board)
            done_binary = 0
            if done:
                done_binary = 1
            file.write(str(action) + " " + str(reward) + " " + str(done_binary) + "\n")
            #print('final reward: ', reward)
        print('final score: ', score)
        step = 0
    file.close()
    np.savez_compressed('data.npz', images)

# gerar imagens com heuristica 2
def snake_examples_images_h2(number_of_games):
    file = open('labels.txt', 'w')
    images = []
    game = SnakeGame(20, 20, border=1)
    step = 0
    for i in range(number_of_games):
        board, reward, done, info = game.reset()
        #plot_board(str(step), board)
        images.append(board)
        done = False
        frame = 0
        score = 0
        while not done:
            step += 1
            score, apple, head, tail, direction = game.get_state()
            action = choose_action_h2(head, apple, direction, game)  # Use the improved heuristic
            #print(action)
            new_board, reward, done, info = game.step(action)
            images.append(new_board)
            board = new_board
            #plot_board(str(step), board)
            done_binary = 0
            if done:
                done_binary = 1
            file.write(str(action) + " " + str(reward) + " " + str(done_binary) + "\n")
            #print('final reward: ', reward)
        print('final score: ', score)
        step = 0
    file.close()
    np.savez_compressed('data.npz', images)    

# gerar imagens com heuristica 1 e 2
def snake_examples_images_h1_h2(nr_h1, nr_h2):
    file = open('labels.txt', 'w')
    images = []
    game = SnakeGame(20, 20, border=1)
    step = 0
    for i in range(nr_h1):
        board, reward, done, info = game.reset()
        #plot_board(str(step), board)
        images.append(board)
        done = False
        frame = 0
        score = 0
        while not done:
            step += 1
            score, apple, head, tail, direction = game.get_state()
            action = choose_action_h1(head, apple, direction, game)  # Use the improved heuristic
            #print(action)
            new_board, reward, done, info = game.step(action)
            images.append(new_board)
            board = new_board
            #plot_board(str(step), board)
            done_binary = 0
            if done:
                done_binary = 1
            file.write(str(action) + " " + str(reward) + " " + str(done_binary) + "\n")
            #print('final reward: ', reward)
        print('final score: ', score)
        step = 0
    for i in range(nr_h2):
        board, reward, done, info = game.reset()
        #plot_board(str(step), board)
        images.append(board)
        done = False
        frame = 0
        score = 0
        while not done:
            step += 1
            score, apple, head, tail, direction = game.get_state()
            action = choose_action_h2(head, apple, direction, game)  # Use the improved heuristic
            #print(action)
            new_board, reward, done, info = game.step(action)
            images.append(new_board)
            board = new_board
            #plot_board(str(step), board)
            done_binary = 0
            if done:
                done_binary = 1
            file.write(str(action) + " " + str(reward) + " " + str(done_binary) + "\n")
            #print('final reward: ', reward)
        print('final score: ', score)
        step = 0
    file.close()
    np.savez_compressed('data.npz', images)
    
# agente normal
def agent(state_shape, action_shape):
    learning_rate= 0.001;
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    print(state_shape.shape[0], state_shape.shape[1], state_shape.shape[2])
    model.add(keras.layers.Input(
        shape=(state_shape.shape[0], state_shape.shape[1], state_shape.shape[2])))
    

    model.add(Conv2D(32, (2, 2), padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(32, (2, 2), padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(keras.layers.Flatten(name="features"))
    
    model.add(Dense(256, activation='relu', kernel_initializer=init))
    
    model.add(Dense(128, activation='relu', kernel_initializer=init))
    
    model.add(Dense(64, activation='relu', kernel_initializer=init))

    model.add(Dense(action_shape, activation='linear', kernel_initializer=init))

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    
    model.summary()
    
    return model

# agente dueling
def agent_duel(state_shape, action_shape):
    learning_rate= 0.001;
    init = tf.keras.initializers.HeUniform()
    
    print(state_shape.shape[0], state_shape.shape[1], state_shape.shape[2])
    input_net = (keras.layers.Input(
        shape=(state_shape.shape[0], state_shape.shape[1], state_shape.shape[2])))
    

    conv1 = Conv2D(32, (2, 2), padding="same", activation="relu")(input_net)
    
    conv2 = Conv2D(32, (2, 2), padding="same", activation="relu")(conv1)
    
    conv3 = Conv2D(64, (2, 2), padding="same", activation="relu")(conv2)
    


    pool =MaxPooling2D(pool_size=(2, 2))(conv3)
    
    flatten = keras.layers.Flatten(name="features")(pool)
    
    value = Dense(256, activation='relu', kernel_initializer=init)(flatten)
    
    value = Dense(128, activation='relu', kernel_initializer=init)(value)
    
    value = Dense(64, activation='relu', kernel_initializer=init)(value)

    value = Dense(1, activation='linear', kernel_initializer=init)(value)




    adv = Dense(256, activation='relu', kernel_initializer=init)(flatten)
    
    adv = Dense(128, activation='relu', kernel_initializer=init)(adv)
    
    adv = Dense(64, activation='relu', kernel_initializer=init)(adv)

    adv = Dense(action_shape, activation='linear', kernel_initializer=init)(adv)
    
    outputs = keras.layers.Add()([adv, value])
    
    model = keras.Model(inputs=input_net, outputs=outputs)
    
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    return model


#load dos exemplos
def getQueue():
    data = np.loadtxt('labels.txt')
    replay_memory = deque(maxlen=55000)
    counter = 0
    images = np.load('data.npz')['arr_0']

    array = images[0]
    print(len(images))
    skip = False
    for i in range(len(images) - 1):
        if i == 0 or skip:
            skip = False
            continue;
        new_array = images[i]
        action = data[counter, 0]
        reward = data[counter, 1]
        done = data[counter, 2]
        if done == 1:
            array = images[i + 1]
            skip = True
        #Add +1 to action because outputs are 0,1,2
        replay_memory.append([array, int(action + 1), reward, new_array, done])

        counter +=1
        array = new_array
    return replay_memory

# treino normal
def train(env, replay_memory, model, target_model, done, batch, batch_size ,number_epochs):

    discount_factor = 0.95
    
    mini_batch = random.sample(replay_memory, batch)
    current_states = np.array([i[0] for i in mini_batch])
    actions = np.array([i[1] for i in mini_batch])
    rewards = np.array([i[2] for i in mini_batch])
    next_states = np.array([i[3] for i in mini_batch])
    dones = np.array([i[4] for i in mini_batch])
        
    q_values_next = model.predict_on_batch(next_states)
    best_actions = np.argmax(q_values_next, axis=1)
    q_values_target_next = target_model.predict_on_batch(next_states)
    


    targets = rewards + discount_factor*(np.amax(target_model.predict_on_batch(next_states), axis=1))*(1-dones)
    
    targets_full = model.predict_on_batch(current_states)
    
    targets_full[np.arange(batch), actions] = targets
    
    model.fit(current_states, targets_full, verbose=0, epochs = number_epochs,batch_size=batch_size)

# treino double
def train_double(env, replay_memory, model, target_model, done, batch, batch_size ,number_epochs):
        
    discount_factor = 0.95
    
    mini_batch = random.sample(replay_memory, batch)
    current_states = np.array([i[0] for i in mini_batch])
    actions = np.array([i[1] for i in mini_batch])
    rewards = np.array([i[2] for i in mini_batch])
    next_states = np.array([i[3] for i in mini_batch])
    dones = np.array([i[4] for i in mini_batch])
        
    
    # Double DQN update
    q_values_next = model.predict_on_batch(next_states)
    best_actions = np.argmax(q_values_next, axis=1)
    q_values_target_next = target_model.predict_on_batch(next_states)
    
    
    targets = rewards + discount_factor * q_values_target_next[np.arange(batch), best_actions] * (1 - dones)


    targets_full = model.predict_on_batch(current_states)
    
    targets_full[np.arange(batch), actions] = targets
    
    model.fit(current_states, targets_full, verbose=0, epochs = number_epochs,batch_size=batch_size)



RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
train_episodes = 2000
max_steps = 10000
create_new_examples = True
train_examples_number = 300
batch = 1000
number_epochs = 5
batch_size = int(batch/number_epochs)

"""batch = 4000
batch_size = 64
number_epochs = int(batch/batch_size + 12)"""


results = []

def train_agent():
    game = SnakeGame(20,20, border=1)
    model = agent(game.board_state(), 3)
    target_model = agent(game.board_state(), 3)
    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    decay = 0.999
    target_model.set_weights(model.get_weights())
    steps_to_update_target_model = 0
    
    
    
    #Fill queue with examples
    if create_new_examples:
        snake_examples_images_h1(350)
    replay_memory = getQueue()
    
    
    
    sum_of_rewards = []
    
    update = False
    
    
    for episode in range(train_episodes):
        
        #get initial configuration
        board,reward,done,info = game.reset()
        
        #Convert image to array
        observation = []
        observation.append(board)
        observation = np.array(observation)
         
        score = 0
        
        for i in range(max_steps):
            
            steps_to_update_target_model += 1
            
            random_number = np.random.rand()
            if random_number <= epsilon:
                action = random.randint(0,2)
            else:      
                predicted = model.predict(observation).flatten()
                #print(predicted)
                action = np.argmax(predicted)
                
            new_board,reward,done,info = game.step(action - 1)
            
            score += reward
            
            new_observation = []
            new_observation.append(new_board)
            new_observation = np.array(new_observation)
            
            observation = new_observation
            
            replay_memory.append([board, action, reward, new_board, done])
            
            board = new_board
            
            if (steps_to_update_target_model % 4 == 0 or done):
                train_double(game, replay_memory, model, target_model, done, batch, batch_size ,number_epochs)
            
            
            if epsilon > min_epsilon:
              epsilon *= decay
            
                
            #print(f'epsilon: {epsilon}, replay_memory: {len(replay_memory)} step: {i+1}')
            
            if done:
                if steps_to_update_target_model >= 100:
                    print('update model heights')
                    target_model.set_weights(model.get_weights())
                    #model.save_weights(str(episode) + '.h5')
                    steps_to_update_target_model = 0
                print(f'episode: {episode+1}/{train_episodes}, epsilon: {epsilon}, score: {score}, steps: {i+1}')
                results.append([episode+1, score,i+1])
                break
        sum_of_rewards.append(score)
        
        
    model.save_weights('agent.h5')
    
    file=open("agent.obj","wb")
    pickle.dump(replay_memory,file)
    file.close()
    
    result = np.array(results)

    plt.figure().set_figwidth(20)

    plt.plot(result[:,0],result[:,2], "ob")
    plt.show() 
    
    plt.figure().set_figwidth(20)
    plt.plot(result[:,0],result[:,1], "or")

    plt.show() 
    

train_agent()