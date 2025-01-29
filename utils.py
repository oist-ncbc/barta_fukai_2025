def despine_ax(ax, where=None, remove_ticks=None):
    if where is None:
        where = 'trlb'
    if remove_ticks is None:
        remove_ticks = where

    if remove_ticks is not None:
        if 'b' in where:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if 'l' in where:
            ax.set_yticks([])
            ax.set_yticklabels([])

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

def underscore(text):
    if len(text) > 0:
        return '_' + text
    else:
        return text