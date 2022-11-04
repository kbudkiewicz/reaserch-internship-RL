import math
import numpy as np
import data1 as d
import data2 as d2
from matplotlib import pyplot as mpl
from collections import deque

def lin_var_plot(matrices, labels, ylab, title='', legend_loc='lower right',smooth=True):
    # input an iterable of matrices to diagnose
    mpl.figure()
    for matrix in matrices:
        if type(matrices) == list:
            matrix_width = matrix.shape[1]
        else:
            matrix_width = matrices.shape[1]
            matrix = matrices
        x = np.arange(matrix_width)

        # extract data and calculate means and standard deviations at each column
        means = []
        sdevs = []
        temp1 = []
        temp2 = []
        for i in range(matrix_width):
            for j in range(5):
                if math.isnan(matrix[j,i]) == True:
                    pass
                else:
                    temp1.append(matrix[j,i])
                    temp2.append(matrix[j,i])
            means.append( np.mean(temp1) )
            sdevs.append( np.sqrt(np.var(temp2)))
            temp1 = []
            temp2 = []

        # smooths the plot with moving average
        if smooth == True:
            # redefine means as a moving average of last 100 values
            moving_avg = deque(maxlen=10)
            for i in range(matrix_width):
                moving_avg.append(means[i])
                means[i] = np.mean(moving_avg)

            # redefine standard deviation (sdevs) as moving average of last 100 values
            moving_sdev = deque(maxlen=10)
            for i in range(matrix_width):
                moving_sdev.append(sdevs[i])
                sdevs[i] = np.mean(moving_sdev)
        else:
            pass

        # define the position of the contours
        top = []
        bottom = []
        for i in range(matrix_width):
            top.append( means[i]+sdevs[i] )
            bottom.append( means[i]-sdevs[i] )

        mpl.fill_between(x, top, bottom, alpha=0.3, label='_nolegend_')
        mpl.plot(x, means)

    mpl.title(title)
    mpl.xlabel('Episode #')
    mpl.ylabel(ylab)
    mpl.legend(labels=labels, loc=legend_loc)
    mpl.show()

### plots of exponential eps-decay
# comparison untrained vs imported state_dict
# labels = ('Untrained','Trained')
# lin_var_plot( [d.M_n2_s, d.M_sd_s], labels, legend_loc='lower right')
# lin_var_plot( [d.M_n2_l, d.M_sd_l], labels, legend_loc='upper right')

# amount of hidden layers comparison
# labels = ('3 hidden layers', '4 hidden layers', '5 hidden layers')
# lin_var_plot( [d.M_n2_s, d.M_hl4_s, d.M_hl5_s], labels, title='Moving score average with different hidden layer sizes', legend_loc='lower right')
# lin_var_plot( [d.M_n2_l, d.M_hl4_l, d.M_hl5_l], labels, title='Moving loss average with different hidden layer sizes', legend_loc='upper right')

# amount of neurons comparison
# labels = ('32 neurons', '64 neurons', '128 neurons', '256 neurons')
# lin_var_plot([d.M_n1_s, d.M_n2_s, d.M_n3_s, d.M_n4_s], labels, title='Moving score average across different neuron amount', legend_loc='lower right')
# lin_var_plot([d.M_n1_l, d.M_n2_l, d.M_n3_l, d.M_n4_l], labels, title='Moving loss average across different neuron amount', legend_loc='upper right')

# epsilon comparison
# labels = ('eps_decay = 0.995','eps_decay = 0.9975','eps_decay = 0.999')
# lin_var_plot([d.M_e995_s, d.M_e9975_s, d.M_e999_s], labels, legend_loc='lower right')
# lin_var_plot([d.M_e995_l, d.M_e9975_l, d.M_e999_l], labels, legend_loc='upper right')

# learning rate comparison
# labels = ('lr = 1e-4','lr = 2.5e-4', 'lr = 5e-4')
# lin_var_plot([d.M_lr1_s, d.M_n2_s, d.M_lr5_s], labels, legend_loc='lower right')
# lin_var_plot([d.M_lr1_l, d.M_n2_l, d.M_lr5_l], labels, legend_loc='upper right')

### plots of linear eps-decay
# eps start comparison
# labels = ('eps_start = 1', 'eps_start = 0.9', 'eps_start = 0.8', 'eps_start = 0.7')
# lin_var_plot([d2.M_lin1_s], labels, legend_loc='lower right')
# lin_var_plot([d2.M_lin1_l], labels, legend_loc='upper right')

# eps end comparison
# labels = ('eps_end = 0.01', 'eps_end = 0.025', 'eps_end = 0.05', 'eps_end = 0.075', 'eps_end = 0.1')
# lin_var_plot( [], labels, legend_loc='lower right' )
# lin_var_plot( [], labels, legend_loc='upper right' )

# eps term comparison
labels = ('Termination at episode 800', 'Termination at episode 1000', 'Termination at episode 1200', 'Termination at episode 1400')
lin_var_plot( [d2.M_et800_s, d2.M_lin1_s, d2.M_et1200_s, d2.M_et1400_s], labels, ylab='Averaged score at given episode', legend_loc='lower right' )
lin_var_plot( [d2.M_et800_l, d2.M_lin1_l, d2.M_et1200_l, d2.M_et1400_l], labels, ylab='Averaged loss at given episode',legend_loc='upper right' )

# learning rate comparison
# labels = ('lr = 1e-3', 'lr = 5e-4','lr = 2.5e-4','lr = 1e-4', 'lr = 1e-5')
# lin_var_plot([d2.M_lr13_s, d2.M_lr54_s, d2.M_lin1_s, d2.M_lr14_s, d2.M_lr15_s], labels, ylab='Averaged score at given episode', legend_loc='lower right')
# lin_var_plot([d2.M_lr13_l, d2.M_lr54_l, d2.M_lin1_l, d2.M_lr14_l, d2.M_lr15_l], labels, ylab='Averaged loss at given episode', legend_loc='upper right')
