<p align="center">
  <img alt="🦾Q-Nim" src="https://user-images.githubusercontent.com/62103572/183402410-5bd9bdb6-f020-487d-a81b-a36937470c5a.png">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/EliaFantini/Q-NIM-ai-learns-to-play-Nim-using-standard-and-deep-Q-Learning">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/EliaFantini/Q-NIM-ai-learns-to-play-Nim-using-standard-and-deep-Q-Learning">
  <img alt="GitHub code size" src="https://img.shields.io/github/languages/code-size/EliaFantini/Q-NIM-ai-learns-to-play-Nim-using-standard-and-deep-Q-Learning">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/EliaFantini/Q-NIM-ai-learns-to-play-Nim-using-standard-and-deep-Q-Learning">
  <img alt="GitHub follow" src="https://img.shields.io/github/followers/EliaFantini?label=Follow">
  <img alt="GitHub fork" src="https://img.shields.io/github/forks/EliaFantini/Q-NIM-ai-learns-to-play-Nim-using-standard-and-deep-Q-Learning?label=Fork">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/EliaFantini/Q-NIM-ai-learns-to-play-Nim-using-standard-and-deep-Q-Learning?label=Watch">
  <img alt="GitHub star" src="https://img.shields.io/github/stars/EliaFantini/Q-NIM-ai-learns-to-play-Nim-using-standard-and-deep-Q-Learning?style=social">
</p>

This project implements the training of different intelligent agents to play the famous game of Nim. To do so, we used the Reinforcement Learning techniques of Q-Learning and Deep Q-Learning (DQN). With Deep Q-Learning , the intelligent agent's behaviour is modeled by a fully connected neural network. 

Interestingly, the optimal policy for Nim is known, so our aim is to answer these three questions:
1. Can RL algorithms learn to play Nim by playing against optimal policy?
2. Can an RL algorithm learn to play Nim only by playing against itself (self-learning)?
3. What are the pros and cons of Deep Q-Learning compared to (Tabular) Q-Learning?

This project was made as an assignment of the EPFL course [CS-456 Artificial Neural Networks](https://lcnwww.epfl.ch/gerstner/VideoLecturesANN-Gerstner.html) (2021/2022), given by Wulfram Gerstner and Johanni Brea.

<p align="center">
  <img height="300" alt="NimGame" src="https://user-images.githubusercontent.com/62103572/183406503-c3a05e71-2574-4eac-8f9c-80f7b29e2b9a.svg">
  <img height="300" alt="iStock" src="https://user-images.githubusercontent.com/62103572/183406507-09021b97-d602-422c-a726-cd87dc1944b6.jpg">
  
</p>

## Authors
-  [Elia Fantini](https://github.com/EliaFantini)
- [Félix Klein](https://github.com/felixkln)

## Results
Through a lot of different game configurations (playing against good and bad experts, by self-learning, using decreasing exploration, modifying the batch size for Deep Q-Learning), our agents learned how to play the game of Nim more or less well. With the appropriate *epsilon* and *n_star* though, some agents actually converged
towards the optimal policy. 

Among the two methods, Q-Learning turned out to be preferable for learning such a simple game, since it manages to achieve similar results in much less time. In fact, not only the number of games to learn the optimal policy is much less, but also the computation time: one Q-Learning training takes 10
seconds on a Ryzen 7 3700x CPU, whereas one Deep Q-Learning training takes roughly 10 minutes on a RTX 3070 GPU (4 minutes
without testing and self-learning). On the other hand, if time is not a problem, for sure opting for a Deep Q-Learning solution on
40’000 games with self-learning and decreasing epsilon will give the most robust model, since it will have explored much more
states and scenarios than the simpler Q-Learning method.

The following images show results with standard Q-Learning. On the left plot we see different intelligent agents (with different values of *n_star*, which is an hyperparameter of the training) playing 500 matches against the Optimal Player, who always does the mathematically best move. This is done every 250 matches played training, and we can see that the score obtained goes closer and closer to 0: when the M_opt score is 0, the intelligent agents wins half of the times against the optimal_player. Since it's impossible to do better, this means the agent has learned to play. As a proof , we have the right image. The game is in the case where there's 3,4,7 sticks in the 3 rows and the mathematically best action is to
remove 1 stick from heap 3. Such action is coded as action 7, which is indeed the one with the highest Q-value (brigther color, white), meaning that it is the action the intelligent agent will choose.
<p align="center">
<img height="250" alt="1" src="https://user-images.githubusercontent.com/62103572/183409263-898cdfe4-26dd-4850-b31f-a8d5d16a73eb.png">
<img height="250" alt="2" src="https://user-images.githubusercontent.com/62103572/183409259-dd62e03c-41e3-4e20-b70a-6283a4794dbc.png">


</p>

The following images show results with Deep Q-Learning. The explanation is similar, again on the left we see the brightest cell is the 14th, which indeed corresponds to the best action that can be chosen.

<p align="center">
<img height="250" alt="Immagine 2022-08-08 133446" src="https://user-images.githubusercontent.com/62103572/183409347-e4d654f6-8c30-4393-b10c-5964ad5d4b00.png">
<img height="250" alt="4" src="https://user-images.githubusercontent.com/62103572/183409352-48a711bb-457b-4c64-bbf4-40d11e457b2c.png">

</p>

To read a more detailed explanation and see all the experiments we did,  we encourage you to read the **report.pdf**. 

## How to install and reproduce results
Download this repository as a zip file and extract it into a folder. The easiest way to run the code is to install Anaconda 3 distribution (available for Windows, macOS and Linux). To do so, follow the guidelines from the official
website (select python of version 3): https://www.anaconda.com/download/

Additional package required are: 
- pytorch
- tqdm
- plotly
- pandas
- multiprocessing
- joblib


To install them write the following command on Anaconda Prompt (anaconda3):
```shell
cd *THE_FOLDER_PATH_WHERE_YOU_DOWNLOADED_AND_EXTRACTED_THIS_REPOSITORY*
```
Then write for each of the mentioned packages:
```shell
conda install *PACKAGE_NAME*
```
Some packages might require more complex installation procedures (especially [pytorch](https://pytorch.org/)). If the above command doesn't work for a package, just google "How to install *PACKAGE_NAME* on *YOUR_MACHINE'S_OS*" and follow those guides.

Then, you can run each of the cells of both jupyter notebooks  **q_learning.ipynb** and  **deep_q_learning.ipynb** to test any game configuration you like. The notebooks follow the order of the questions in ```MP_Nim.pdf```.


## Files description
* **MP_NIM.pdf** : pdf containing the rules and questions to answer for the assignment of the EPFL course.
* **report.pdf** : pdf containing answers,plots and conclusions for course's assignment.
* **q_learning.ipynb** : Jupyter Notebook containing the Q-Learning part of the project (questions 1 to 10)
* **q_learning_results.pkl** : ```pickle``` file to reload all results of the previous notebook without having to re-run the main game function (```q_learning```)
* **deep_q_learning.ipynb** : Jupyter Notebook containing the Deep Q-Learning part of the project, as well as the comparison between Q-Learning and Deep Q-Learning (questions 11 to 21)
* **deep_q_learning_results.pkl** : ```pickle``` file to reload all results of the previous notebook without having to re-run the main game function (```deep_q_learning```)
* **nim_env.py** : code implementation of the NIM game environment

## Hardware used

- GPU: GeForce RTX 3070
- CPU: Ryzen 7 3700x
- RAM: 16 GB


## Implementation details

In each approach, it is possible to assess the performance of our learning agent against optimal and random players by setting the ```test``` variable to ```True```.
It is also possible to make our agent play against itself, i.e. by changing the instance of the adversary player with the boolean ```self_learning```.

## 🛠 Skills
Python, Pytorch, Matplotlib, Pandas, Plotly, Multiprocessing and Joblib. Machine learning, Reinforcement Learning, Q-Learning and Deep Q-Learning, self-learning, DQN.

## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/EliaFantini/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/-elia-fantini/)
