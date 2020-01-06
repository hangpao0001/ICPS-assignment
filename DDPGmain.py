import numpy as np
import matplotlib.pyplot as plt
from Dynamic.Dynamicenv import ArmEnv
from DDPG.DDPGrl import DDPG

MAX_EPISODES = 1000
MAX_EP_STEPS = 300
ON_TRAIN = True
GLO_B = []
# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            env.render()

            a = rl.choose_action(s)

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                if len(GLO_B) == 0:
                    GLO_B.append(ep_r)
                else:
                    GLO_B.append(0.9 * GLO_B[-1] + 0.1 * ep_r)
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
    rl.save()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        s = env.reset()
        for _ in range(200):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()
plt.plot(np.arange(len(GLO_B)),GLO_B)
plt.xlabel('Episode');
plt.ylabel('Moving reward');
plt.show()