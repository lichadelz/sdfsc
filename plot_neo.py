import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_neo():  

    y1 = np.load('plan_data/d_no_neo.npy')
    y2 = np.load('plan_data/d_neo_sdfsc.npy')
    y3 = np.load('plan_data/d_neo_origin.npy')
    y4= np.zeros(700)
    x = np.arange(700)

    fig, ax1 = plt.subplots()
    ax1.plot(x[:len(y1)]/100, y1, label='No Control', antialiased=True)
    ax1.plot(x[:700]/100, y2[:700], label='NEO-SS', antialiased=True)
    ax1.plot(x[:700]/100, y3[:700] , label='NEO', antialiased=True)
    ax1.plot(x[:700]/100, y4[:700] , color='red', linestyle='--', antialiased=True)

    ax1.set_xlabel('Time(s)', fontsize=11)
    ax1.set_ylabel('Distance(m)', fontsize=11)
    ax1.set_ylim(-0.05, 0.25)
    ax1.set_xlim(0,7)

    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(color='b', linestyle='--', linewidth=0.5)
    plt.savefig('plan_data/Reactive Control.png', dpi=1000,bbox_inches='tight')  
    plt.show()

if __name__ == '__main__':
    plot_neo()