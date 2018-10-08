# DQN

The DQN algorithm used for solving Gym's cartpole environment.    
I changed the reward function:    
```
reward = np.cos(2*next_state[3]) 
```
## Requirements

- Tensorflow  
- Numpy   
- Gym 

## Run 

There is a constant: 
```
DEVICE = '/gpu:0'
```
Set it to '/cpu:0' if you don't have one. 

And then run as: 

```
$ python main.py

```


