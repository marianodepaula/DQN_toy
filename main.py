
import tensorflow as tf
import numpy as np
import time
from replay_buffer import ReplayBuffer
from q_network import Network
from image import imageGrabber
import gym
import cv2
from robots import gym_environment

DEVICE = '/cpu:0'

# Base learning rate 
LEARNING_RATE = 1e-3
# Soft target update param
TAU = 1.
RANDOM_SEED = 11543521#1234
EXPLORE = 1000000


N_ACTIONS = 4
SIZE_FRAME = 84

def trainer(epochs=1000, MINIBATCH_SIZE=64, GAMMA = 0.99,save=1, save_image=1, epsilon=1.0, min_epsilon=0.05, BUFFER_SIZE=15000, train_indicator=True, render = True):
    with tf.Session() as sess:

        # configuring the random processes
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        # set evironment

        env = gym.make('CartPole-v1') 
        print('action ', env.action_space)
        print('obs ', env.observation_space)
        observation_space = 4
        action_space = 2
        agent = Network(sess,observation_space, action_space,LEARNING_RATE,TAU,DEVICE)
        
        # TENSORFLOW init seession
        sess.run(tf.global_variables_initializer())
               
        # Initialize target network weights
        agent.update_target_network()
        # Initialize replay memory
        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
        replay_buffer.load()
        print('buffer size is now',replay_buffer.count)
        # this is for loading the net  
        if save:
            try:
                agent.recover()
                print('********************************')
                print('models restored succesfully')
                print('********************************')
            except:
                print('********************************')
                print('Failed to restore models')
                print('********************************')
        
        winner_used = 0
        for i in range(epochs):

            if (i%500 == 0) and (i != 0): 
                print('*************************')
                print('now we save the model')
                agent.save()
                #replay_buffer.save()
                print('model saved succesfuly')
                print('*************************')
            
            if i%100 == 0: 
                 agent.update_target_network()
                 
            
            state = env.reset()
            
            q0 = np.zeros(action_space)
            ep_reward = 0.
            done = False
            step = 0
            
            while not done:
                
                epsilon -= 0.000001
                epsilon = np.maximum(min_epsilon,epsilon)
                
                # 1. get action with e greedy
                
                if np.random.random_sample() < epsilon:
                    #Explore!
                    action = np.random.randint(0,action_space)
                    # print('random', action)
                else:
                    # Just stick to what you know bro
                    q0 = agent.predict(np.reshape(state,(1,observation_space)) ) 
                    action = np.argmax(q0)
                    # print('action', action)

                next_state, reward, done, info = env.step(action)
                #print('next_state',next_state)
                # env.render()
                               
                if train_indicator:
                    
                   # Keep adding experience to the memory until
                    # there are at least minibatch size samples
                    if replay_buffer.size() > MINIBATCH_SIZE:

                        
                        # 4. sample random minibatch of transitions: 
                        s_batch, a_batch, r_batch, t_batch, s2_batch= replay_buffer.sample_batch(MINIBATCH_SIZE)

                        
                        q_eval = agent.predict_target(np.reshape(s2_batch,(MINIBATCH_SIZE,observation_space)))

                        #q_target = np.zeros(MINIBATCH_SIZE)
                        q_target = q_eval.copy()
                        for k in range(MINIBATCH_SIZE):
                            if t_batch[k]:
                                q_target[k][a_batch[k]] = r_batch[k]
                            else:
                                # TODO check that we are grabbing the max of the triplet
                                q_target[k][a_batch[k]] = r_batch[k] + GAMMA * np.max(q_eval[k])

                           #print(q_target)
                        #5.3 Train agent! 
                        #print(a_batch)
                        #print(q_target)
                        #print(np.reshape(a_batch,(-1,action_space)))
                        #print(np.reshape(q_target,(MINIBATCH_SIZE, action_space)))
                        #print(np.reshape(s_batch,(MINIBATCH_SIZE,observation_space)) )
                        agent.train(np.reshape(a_batch,(MINIBATCH_SIZE,1)),np.reshape(q_target,(MINIBATCH_SIZE, action_space)), np.reshape(s_batch,(MINIBATCH_SIZE,observation_space)) )
                        
                        

                # 3. Save in replay buffer:
                replay_buffer.add(state,action,reward,done,next_state) 
                
                # prepare for next state
                state = next_state
                ep_reward = ep_reward + reward
                step +=1
                
                
                #end2 = time.time()
                #print(step, action, q0, round(epsilon,3), round(reward,3))#, round(loop_time,3), nseconds)#'epsilon',epsilon_to_print )
                   #print(end-start, end2 - start)
                 
            
            
            print('th',i+1,'Step', step,'Reward:',ep_reward,'epsilon', round(epsilon,3) )
            #print('the reward at the end of the episode,', reward)
          
                        

            

            #time.sleep(15)

        print('*************************')
        print('now we save the model')
        agent.save()
        #replay_buffer.save()
        print('model saved succesfuly')
        print('*************************')
        
        


if __name__ == '__main__':
    trainer(epochs=20000 ,save_image = False, epsilon= 0.01, train_indicator = True)
