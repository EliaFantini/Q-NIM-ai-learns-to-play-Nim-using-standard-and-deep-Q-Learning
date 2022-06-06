# CS-456-Artificial-Neural-Networks-Project
Q-Learning and Deep Q-Learning to train artificial agents that can play the famous game of Nim.

2022 miniproject of the Artificial Neural Networks course at EPFL (code CS-456), given by the very well-known Wulfram Gerstner and Johanni Brea.

## File architecture
* **q_learning.ipynb** : Jupyter Notebook containing the Q-Learning part of the project (questions 1 to 10)
* **q_learning_results.pkl** : ```pickle``` file to reload all results of the previous notebook without having to re-run the main game function (```q_learning```)
* **deep_q_learning.ipynb** : Jupyter Notebook containing the Deep Q-Learning part of the project, as well as the comparison between Q-Learning and Deep Q-Learning (questions 11 to 21)
* **deep_q_learning_results.pkl** : ```pickle``` file to reload all results of the previous notebook without having to re-run the main game function (```deep_q_learning```)
* **nim_env.py** : Python file defining the environment of the game of Nim

## Hardware

- GPU: GeForce RTX 3070
- CPU: Ryzen 7 3700x
- RAM: 16 GB

## Required libraries
* **torch**
* **tqdm**
* **random**
* **numpy**
* **collections**
* **plotly.express**
* **pandas**
* **multiprocessing**
* **joblib**
* **pickle**
* **time**

## How to run the code ?

First, clone the repository. Then, you can run each of the cells of both notebooks to test any game configuration you like. The notebooks follow the order of the questions in ```MP_Nim.pdf```.

## Implementation details

In each approach, it is possible to assess the performance of our learning agent against optimal and random players by setting the ```test``` variable to ```True```.
It is also possible to make our agent play against itself, i.e. by changing the instance of the adversary player with the boolean ```self_learning```.

## Authors
* Elia Fantini
* Félix Klein
