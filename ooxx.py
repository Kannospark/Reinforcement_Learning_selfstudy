import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

class Agent():

    def __init__(self, OOXX_index, Epsilon=0.1, LearningRate = 0.1):
        self.value = np.zeros([3, 3, 3, 3, 3, 3, 3, 3, 3])
        self.currentState = np.zeros(9)
        self.previousState = np.zeros(9)
        self.index = OOXX_index
        self.epsilon = Epsilon
        self.alpha = LearningRate

    def reset(self):
        self.currentState = np.zeros(9)
        self.previousState = np.zeros(9)

    def actionTake(self, State):
        state = State.copy()
        avaliable = np.where(state == 0)[0]
        length = len(avaliable)
        if length == 0:
            return state
        else:
            random = np.random.rand()
            if random < self.epsilon:
                choose = np.random.randint(length)
                state[avaliable[choose]] = self.index
            else:
                tempValue = np.zeros(length)
                for i in range(length):
                    tempState = state.copy()
                    tempState[avaliable[i]] = self.index
                    tempValue[i] = self.value[tuple(tempState.astype(int))]
                choose = np.where(tempValue == np.max(tempValue))[0]
                chooseIndex = np.random.randint(len(choose))
                state[avaliable[choose[chooseIndex]]] = self.index
            if args.debug:
                print(state)
            return state
        
    def valueUpdate(self, State, Reward=0):
        self.currentState = State.copy()
        self.value[tuple(self.previousState.astype(int))] = self.value[tuple(self.previousState.astype(int))] + self.alpha * (self.value[tuple(self.currentState.astype(int))] - self.value[tuple(self.previousState.astype(int))] + Reward)
        self.previousState = self.currentState.copy()
        

    # 判断是否胜利
    def isWin(self, State):
        state = State.copy()
        state = state.reshape([3, 3])
        for i in range(3):
            if (state[i,0] == self.index and state[i,1] == self.index and state[i,2] == self.index) or (state[0,i] == self.index and state[1,i] == self.index and state[2,i] == self.index):
                return True
        if (state[0,0] == self.index and state[1,1] == self.index and state[2,2] == self.index) or (state[0,2] == self.index and state[1,1] == self.index and state[2,0] == self.index):
            return True
        return False

def test(agent1, agent2, e1, e2, times=1000):
    times = times
    agent1.epsilon = e1
    agent2.epsilon = e2
    competition_result = np.zeros(times)

    for i in range(times):
        # 假设agent1先手
        state = np.zeros(9)
        # print('Game:', i)
        while True:
            state = agent1.actionTake(state)
            # 判断Agent1是否胜利
            if agent1.isWin(state):
                competition_result[i] = 1
                agent1.valueUpdate(state, Reward=1)
                agent2.valueUpdate(agent2.currentState, Reward=-1)
                if args.debug:
                    print('Agent1 Win')
                break
            else:
                agent1.valueUpdate(state, Reward=0)
            # 判断游戏是否平局
            if np.where(state == 0)[0].size == 0:
                agent1.valueUpdate(state, Reward=0)
                agent2.valueUpdate(agent2.currentState, Reward=0)
                competition_result[i] = 0
                if args.debug:
                    print('Draw')
                break

            state = agent2.actionTake(state)
            if agent2.isWin(state):
                competition_result[i] = -1
                agent1.valueUpdate(agent1.currentState, Reward=-1)
                agent2.valueUpdate(state, Reward=1)
                if args.debug:
                    print('Agent2 Win')
                break
            else:
                agent2.valueUpdate(state, Reward=0)
                
        agent1.reset()
        agent2.reset()

    agent1_wins = 0
    agent1_wins_proportion = np.zeros(times)
    for i in range(times):
        if competition_result[i] == 1:
            agent1_wins += 1
        agent1_wins_proportion[i] = agent1_wins / (i + 1)
    if args.plot:
        plt.plot(agent1_wins_proportion)
        plt.show()

def play(agent, index=-1):
    # 玩家与训练好的Agent对战
    agent.epsilon = 0
    agent_player = Agent(index)
    state = np.zeros(9)
    while True:
        state = agent.actionTake(state)
        print(state.reshape(3,3))
        if agent.isWin(state):
            print('Agent Win')
            break
        if np.where(state == 0)[0].size == 0:
            print('Draw')
            break
        
        position = input('Please input the position: ')
        state[int(position)] = -1
        if agent_player.isWin(state):
            print('You Win')
            break


if __name__ == '__main__':
    
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='Example script for debug flag.')

    # 添加一个可选的命令行参数--debug
    parser.add_argument('-debug')
    parser.add_argument('-plot')

    # 解析命令行参数
    args = parser.parse_args()
    
    times = 30000
    agent1 = Agent(1, Epsilon=0.2, LearningRate=0.2)
    agent2 = Agent(-1, Epsilon=0.2, LearningRate=0.2)

    test(agent1, agent2, 0.2, 0.2, times=times)

    if args.plot:
        test(agent1, agent2, 0, 0, times=times)
        
    else: play(agent1)
