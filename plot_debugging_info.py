import numpy as np
from matplotlib import pyplot as plt
import os

def combine_results(big_matrix,file_no,DIR_file):
    episode_xtick_position=[]
    episode_xtick_position.append(1)
    episode_xtick_labels=[]
    episode_xtick_labels.append(1)
    for i in range(2,int(file_no/7)+1):
        y=np.load(DIR_file.format(i))
        big_matrix=np.concatenate((big_matrix,y))
        #episode_legend_positions_and_lables
        episode_xtick_position.append(episode_xtick_position[-1]+y.shape[0])
        episode_xtick_labels.append(i)
    return big_matrix,episode_xtick_position,episode_xtick_labels

def lineplotCI(x_data, y_data, sorted_x, low_CI, upper_CI, x_label, y_label, title):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    ax.plot(x_data, y_data, lw = 1, color = '#539caf', alpha = 1, label = 'Mean of weights')
    # Shade the confidence interval
    ax.fill_between(sorted_x, low_CI, upper_CI, color = '#539caf', alpha = 0.4, label = '1 standard deviation of weights')
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Display legend
    ax.legend(loc = 'best')

#count number of directories
DIR="./log/dqn_pa_gs_4_d_2_bf_2_cmaxl_1_cminl_1_CTMaze-v0/190426-143036apnn/debug_matrix"
file_no=len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
#initialise matrix for loading
DIR_big_matrix_y_apnn_in=DIR+"/y_apnn_in_stack_{0}000.npy"
big_matrix_y_apnn_in=np.load(DIR_big_matrix_y_apnn_in.format(1))

DIR_big_matrix_y_apnn_out=DIR+"/y_apnn_out_stack_{0}000.npy"
big_matrix_y_apnn_out=np.load(DIR_big_matrix_y_apnn_out.format(1))

DIR_big_matrix_y_dqn_out=DIR+"/y_dqn_out_stack_{0}000.npy"
big_matrix_y_dqn_out=np.load(DIR_big_matrix_y_dqn_out.format(1))

DIR_big_matrix_apnn_weights=DIR+"/apnn_weights_{0}000.npy"
big_matrix_apnn_weights=np.load(DIR_big_matrix_apnn_weights.format(1))

DIR_big_matrix_conv_layers_out=DIR+"/y_conv_layers_stack_{0}000.npy"
big_matrix_conv_layers_out=np.load(DIR_big_matrix_conv_layers_out.format(1))

DIR_big_matrix_elgibility_traces=DIR+"/Eligbility_traces_stack_{0}000.npy"

DIR_big_matrix_reward=DIR+"/reward_{0}000.npy"
big_matrix_reward=np.load(DIR_big_matrix_reward.format(1))
print(big_matrix_reward)
#load each big_matrix
big_matrix_y_apnn_in,episode_xtick_position,episode_xtick_labels=combine_results(big_matrix_y_apnn_in,file_no,DIR_big_matrix_y_apnn_in)
big_matrix_y_apnn_out,destroy,destroy_2=combine_results(big_matrix_y_apnn_out,file_no,DIR_big_matrix_y_apnn_out)
big_matrix_y_dqn_out,destroy,destroy_2=combine_results(big_matrix_y_dqn_out,file_no,DIR_big_matrix_y_dqn_out)
big_matrix_conv_layers_out,destroy,destroy_2=combine_results(big_matrix_conv_layers_out,file_no,DIR_big_matrix_conv_layers_out)
#v-stack weightes
for i in range(2,int(file_no/7)+1):
    y=np.load(DIR_big_matrix_apnn_weights.format(i))
    big_matrix_apnn_weights=np.dstack((big_matrix_apnn_weights,y))
    y2=np.load(DIR_big_matrix_reward.format(i))
    big_matrix_reward=np.dstack((big_matrix_reward,y2))

#set_xticks
episode_xtick_position=np.asarray(episode_xtick_position)[0::50]
x_values=episode_xtick_labels
episode_xtick_labels=np.asarray(episode_xtick_labels)[0::50]
# plot all episodes
f=plt.figure(1)
imgplot=plt.imshow(big_matrix_y_apnn_in.transpose(), interpolation='none',aspect=150)
plt.title("y_apnn_in")
plt.colorbar(imgplot)
plt.xticks(episode_xtick_position,episode_xtick_labels)
plt.xlabel('thousands of epsiodes')
plt.ylabel('apnn input')
g=plt.figure(2)
imgplot=plt.imshow(big_matrix_y_apnn_out.transpose(), interpolation='none',aspect=150)
plt.title("y_apnn_out")
plt.colorbar(imgplot)
plt.xticks(episode_xtick_position,episode_xtick_labels)
plt.xlabel('thousands of epsiodes')
plt.ylabel('apnn output')
e=plt.figure(3)
imgplot=plt.imshow(big_matrix_y_dqn_out.transpose(), interpolation='none',aspect=150)
plt.title("y_dqn_out")
plt.colorbar(imgplot)
plt.xticks(episode_xtick_position,episode_xtick_labels)
plt.xlabel('thousands of epsiodes')
plt.ylabel('dqn output')

d=plt.figure(22)
imgplot=plt.imshow(big_matrix_conv_layers_out.transpose(), interpolation='none',aspect=1)
plt.title("y_dqn_out")
plt.colorbar(imgplot)
plt.xticks(episode_xtick_position,episode_xtick_labels)
plt.xlabel('thousands of epsiodes')
plt.ylabel('dqn output')

#weights plots
apnn_weights_mean=np.mean(big_matrix_apnn_weights,axis=0)
apnn_weights_var=np.var(apnn_weights_mean,axis=0)
apnn_weights_mean=np.mean(apnn_weights_mean,axis=0)
lower_CI=apnn_weights_mean+apnn_weights_var
upper_CI=apnn_weights_mean-apnn_weights_var
z=plt.figure(8)
#imgplot=plt.plot(apnn_weights_mean)
lineplotCI(x_values, apnn_weights_mean, x_values, lower_CI, upper_CI, 'thousands of epsiodes', 'mean weight value', "apnn weights")
#plt.xticks(episode_xtick_position,episode_xtick_labels)

#show weights per output
for i in range(0,3):
    plt.figure(10+i)
    plot=plt.plot(x_values,np.transpose(big_matrix_apnn_weights[i,:,:]))
    plot=plt.plot(x_values,np.transpose(big_matrix_reward[0,:,:]))
    plt.title("y_apnn_out")
    plt.xlabel('thousands of epsiodes')
    plt.ylabel('weights for action {0}'.format(i))


#plot single episodes
episode_number=144 #in thousands of episodes
single_episode_y_apnn_in=np.load(DIR_big_matrix_y_apnn_in.format(episode_number))
single_episode_y_apnn_out=np.load(DIR_big_matrix_y_apnn_out.format(episode_number))
single_episode_y_dqn_out=np.load(DIR_big_matrix_y_dqn_out.format(episode_number))
single_elgibility_traces=np.load(DIR_big_matrix_elgibility_traces.format(episode_number))
single_episode_conv_layers_out=np.load(DIR_big_matrix_conv_layers_out.format(episode_number))
print(single_elgibility_traces.shape)
f=plt.figure(4)
imgplot=plt.imshow(single_episode_y_apnn_in.transpose(), interpolation='none',aspect=0.25)
plt.title("y_apnn_in_single_episode")
plt.colorbar(imgplot)
plt.xlabel('steps')
plt.ylabel('apnn input')
g=plt.figure(5)
imgplot=plt.imshow(single_episode_y_apnn_out.transpose(), interpolation='none',aspect=1.5)
plt.title("y_apnn_out_single_episode")
plt.colorbar(imgplot)
plt.xlabel('steps')
plt.ylabel('apnn output')
e=plt.figure(6)
imgplot=plt.imshow(single_episode_y_dqn_out.transpose(), interpolation='none',aspect=1.5)
plt.title("y_dqn_out_single_episode")
plt.colorbar(imgplot)
plt.xlabel('steps')
plt.ylabel('dqn output')
e=plt.figure(20)
imgplot=plt.imshow(single_episode_conv_layers_out.transpose(), interpolation='none',aspect=0.02)
plt.title("conv_layers_out_single_episode")
plt.colorbar(imgplot)
plt.xlabel('steps')
plt.ylabel('dqn output')
g=plt.figure(7)
imgplot=plt.imshow(single_episode_y_apnn_out.transpose()+single_episode_y_dqn_out.transpose(), interpolation='none',aspect=1.5)
plt.title("taken action")
plt.colorbar(imgplot)
plt.xlabel('steps')
plt.ylabel('network output')



#plot Eligbility_traces_singles
for i in range(0,3):
    plt.figure(14+i)
    plot=plt.plot(np.transpose(single_elgibility_traces[i,:,:]))
    plt.title("elgibility traces")
    plt.xlabel('steps')
    plt.ylabel('trace for action {0}'.format(i))



plt.show()
