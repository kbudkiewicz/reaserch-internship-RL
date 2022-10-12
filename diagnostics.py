import time
import numpy as np
import matplotlib.pyplot as mpl
from collections import deque

def diagnose(list_scores, list_loss, episodes):
    x1 = np.linspace(0, episodes, len(list_scores))

    # get the moving mean of scores
    m1 = []
    d = deque(maxlen=100)
    for i in list_scores:
        d.append(i)
        if len(d) == 100:
            m1.append( np.mean(d) )
    mx1 = np.linspace( 100, episodes, len(m1) )

    scores_plot = mpl.figure(1)
    mpl.title('Scores')
    mpl.xlabel('Episode #')
    mpl.ylabel('Score @ Episode')
    # mpl.legend( ['Score @ Episode','Moving mean of 100 scores'], loc='lower right' )
    mpl.plot(x1, list_scores, label='Score @ Episode')
    mpl.plot(mx1,m1, label='Moving mean of 100 scores')

    x2 = x1
    # get the moving mean of scores
    l1 = []
    d = deque(maxlen=100)
    for i in list_loss:
        d.append(i)
        if len(d) == 100:
            l1.append( np.mean(d) )
    lx1 = np.linspace( 100, episodes, len(l1) )

    loss_plot = mpl.figure(2)
    mpl.title('Loss')
    mpl.xlabel('Episode #')
    mpl.ylabel('Loss @ Episode')
    mpl.plot(x2,list_loss)
    mpl.plot(lx1,l1)

    # get current time in local format
    year = time.localtime(time.time()).tm_year
    month = time.localtime(time.time()).tm_mon
    day = time.localtime(time.time()).tm_mday
    hour = time.localtime(time.time()).tm_hour
    minute = time.localtime(time.time()).tm_min
    current_time = '%s_%s_%s@%s_%s' % (day, month, year, hour, minute)

    # save the figures with the time of creation
    scores_plot.savefig('C:/Users/kryst/Documents/GitHub/research-internship-RL/Diagnostics/Scores_%s.png' % current_time)
    loss_plot.savefig('C:/Users/kryst/Documents/GitHub/research-internship-RL/Diagnostics/Loss_%s.png' % current_time)

# def make_rapport(*args):
    # input: device, episodes, play_time, layer_size, net_update, memory_size, batch_size, gamma, tau, lr, eps, eps_end, eps_decay, solved_in
    # parameters = dict{
    #     1:device
    #     2:episodes
    #     3:play_time
    #     4:layer_size
    #     5:memory_size
    #     6:batch_size
    #     7:gamma
    #     8:tau
    #     9:lr
    #     10:eps
    #     11:eps_end
    #     12:eps_decay
    #     13:solved_in
    # }

    # f = open('C:/Users/kryst/Documents/GitHub/research-internship-RL/Rapport@%s' % (current_time), 'w')
    # for line in f:
    #     if line == 0:
    #         line = f.write('Time: %s.%s.%s, %s:%s' % (day, month, year, hour, minute))
    #     else:
    #         line = f.write('%s = %.4f' % (dict{line}, dict{line}))
    # f.close()