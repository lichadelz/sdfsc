import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
def sample(score_all):
    length=len(score_all)
    sample_indices = np.linspace(0, length - 1, num=20, dtype=int)
    score_all=score_all[sample_indices]
    print(score_all.shape)
    return score_all
def plot_opttraj():  

    y1 = np.load('plan_data/score_all_rrtstar_simple.npy') 
    y2 = np.load('plan_data/score_all_rrtopt.npy')
    y3= np.load('plan_data/score_all_rrtcon_simple.npy')
    y2=sample(y2)
    y3=sample(y3)
    x = np.arange(len(y1))

    x_new = np.linspace(x[0], x[-1], 300)  
    y1_smooth = make_interp_spline(x, y1, k=2)(x_new)
    y2_smooth = make_interp_spline(x, y2, k=2)(x_new)
    y3_smooth = make_interp_spline(x, y3, k=2)(x_new)
    print(y1_smooth.shape)
    print(y2_smooth.shape)
    print(y3_smooth.shape)

    fig, ax1 = plt.subplots()
    ax1.plot(x_new/5, y1_smooth, label='RRT*', antialiased=True)
    ax1.plot(x_new/5, y3_smooth, label='RRT-connect', antialiased=True)
    ax1.plot(x_new/5, y2_smooth, label='Ours', antialiased=True)
    ax1.set_xlabel('Time(s)', fontsize=11)
    ax1.set_ylabel('Collision Distance(m)', fontsize=11)
    ax1.set_ylim(-0.05, 0.4)

    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(color='b', linestyle='--', linewidth=0.5)
    # plt.title('Comparison-FCL/SDF-SS')
    plt.savefig('plan_data/opt_comparion.png', dpi=1000,bbox_inches='tight')  
    plt.show()

if __name__ == '__main__':

    plot_opttraj()