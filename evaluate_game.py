from deep_rl import *
import numpy as np



def evaluate_game(agent,log,name):
    agent.load(log)
    print(agent.network)

    state = agent.task.reset()
    memory = np.asarray(state)
    ##initalise the game
    task = Task(name, frame_stack = 4, episode_life =False)
    #env = task().get_env() 
    state =task.reset()
    count=0
    while True:
        task.render()
        time.sleep(0.03)
        time.sleep(10)
        #print(state.size())
        #value = agent.network(agent.config.state_normalizer(state)).flatten()
        value = agent.network(agent.config.state_normalizer(state))
        value = to_np(value).flatten()
        action = np.argmax(value)
        print(value)
        observation, reward, done, info = task.step([action])
        print(done)
        if done:
            count=count+1
        if count > 1:
            print("game is done! quitting")
            time.sleep(2)
            break
        state = observation
        
