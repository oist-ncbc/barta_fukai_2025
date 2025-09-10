import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['font.size'] = 9
mpl.rcParams['savefig.dpi'] = 300

hist_params = {
    'histtype': 'step',
    'density' : True,
    'linewidth': 1.5
}

def format_p(p):
    if p >= 0.01:
        return f"{p:.2f}"
    elif p < 1e-100:
        x = 100
        return f"$<10^{{-{x}}}$"
    else:
        # find smallest integer x such that p < 10^-x
        x = int(np.floor(-np.log10(p)))
        return f"$<10^{{-{x}}}$"

def log_ticks(ax):
    xticklabels = ax.get_xticklabels()
    new_ticks = [fr'$10^{{{x._text}}}$' for x in xticklabels]
    ax.set_xticklabels(new_ticks)

def rule_marker(rule):
    if rule == 'GSF' or rule == 'rate':
        return ('o')
    elif rule == 'sHIP' or rule == 'hebb_smooth_rate':
        return ('v')
    elif rule == 'fHIP' or rule == 'hebb':
        return ('s')
    else:
        raise Exception("Unknown rule") 

def rule_name(rule):
    if rule == 'GSF' or rule == 'rate':
        return ('GHP')
    elif rule == 'sHIP' or rule == 'hebb_smooth_rate':
        return ('sLHP')
    elif rule == 'fHIP' or rule == 'hebb':
        return ('fLHP')
    else:
        return rule

def despine_ax(ax, where=None, remove_ticks=None):
    if where is None:
        where = 'trlb'
    if remove_ticks is None:
        remove_ticks = where

    if remove_ticks is not None:
        if 'b' in where:
            # ax.set_xticks([])
            ax.set_xticklabels([])
            for tick in ax.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
        if 'l' in where:
            # ax.set_yticks([])
            ax.set_yticklabels([])

            for tick in ax.yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

    to_despine = []

    if 'r' in where:
        to_despine.append('right')
    if 't' in where:
        to_despine.append('top')
    if 'l' in where:
        to_despine.append('left')
    if 'b' in where:
        to_despine.append('bottom')

    for side in to_despine:
        ax.spines[side].set_visible(False)

# def rule_color(rule):
#     if rule == 'GSF' or rule == 'rate':
#         return ('#0684a5')
#     elif rule == 'sHIP' or rule == 'hebb_smooth_rate':
#         return ('#6f4e7c')
#     elif rule == 'fHIP' or rule == 'hebb':
#         return ('#ca472f')
#     else:
#         raise Exception("Unknown rule") 

def rule_color(rule):
    if rule == 'GSF' or rule == 'rate':
        return ('#0078d2')
    elif rule == 'sHIP' or rule == 'hebb_smooth_rate':
        return ('#43C091')
    elif rule == 'fHIP' or rule == 'hebb':
        return ('#e646a0')
    elif rule == 'uniform':
        return 'gray'
    else:
        raise Exception("Unknown rule")
    
def plot_table(ax, nix_list, fontsize=None, colors=None):
    shared_neurons = []
    
    for nix1 in nix_list:
        shared_neurons.append([])
        for nix2 in nix_list:
            shared_neurons[-1].append(np.isin(nix1, nix2).sum())
    
    shared_neurons = np.array(shared_neurons)

    table = ax.table(shared_neurons, bbox=[0,0,1,1])
    
    ax.axis('tight')
    # ax.axis('off')
    
    for ii in range(len(nix_list)):
        if colors is None:
            table[(ii, ii)].set_facecolor(f'C{ii}')
        else:
            table[(ii, ii)].set_facecolor(colors[ii])
            
        table[(ii, ii)].set_alpha(0.3)

    if fontsize is not None:
        for key, cell in table.get_celld().items():
            cell.get_text().set_fontsize(fontsize)

    despine_ax(ax)
    
type_color = {
    'exc': 'black',
    'inh': 'gray'
}