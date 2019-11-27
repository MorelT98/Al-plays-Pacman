# Al-plays-Pacman
Attempt to use reinforcement learning to teach an AI to play Pacman

**Pacman Version:** On the console (See /levels and environment.py for implementation)<br/>
**Technology:** Python, Tensorflow and Keras<br/>
**Models:** Policy gradient model and Monte Carlo<br/>

## How to run
Install Python<br/>
Download the repository<br/>
Run the training.py file<br/>

## How it works
Pacman has 4 possible directions: UP, DOWN, LEFT, RIGHT. To choose where to go, Pacman has an Neural Network that, given the state of the game (the whole grid), predicts the probabilities of choosing each direction. Each action leads to a specific reward. Bumping into a wall for example would lead to a negative reward.<br/> 
At the end of each game, the states, actions and rewards are collected as training data. We then use policy gradients to train the network and improve the agent.<br/>
Once in a while, the current agent competes with the last best network, and the one with the greater reward over an amount of games is saved.

## Files descriptions
**- /levels:** The implementation of a level in the console<br/>
**- /logs/pacman_agent_v0:** Tensorflow parameters saved for the best model<br/>
**- /models:** The saved Tensorflow models<br/>
**- best_score.pkl:** The best score obtained so far, saved with the pickle library<br/>
**- config.py:** The network and Policy gradient parameters<br/>
**- eds.py:** Implementation of a Stack and Queue<br/>
**- environment.py:** Implementation of the Pacman environment (State and actions)<br/>
**- network.py:** Implementation of a Convolutional Neural Network with a two headed output: The policy head and the value head<br/>
**- policy_gradient_model.py:** Creates/Loads the network and implements policy gradient and Monte Carlo algorithms<br/>
**- training.py:** Trains the model<br/>

## Results
After a day of training, the model was able to avoid walls, but still ran into enemies without avoiding them. Tuning the network parameters (Number of layers, learning rate, etc) and the Policy Gradients parameters (number of games before collecting data, number of iterations before competing, etc) might lead to better results. Stay tuned!
