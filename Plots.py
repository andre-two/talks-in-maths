import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
# GRAFICO DE FREQUENCIA OBSERVADA E ESTIMADA POR DECIL DE VARIAVEL CONTINUA

from pandas._libs.lib import is_integer

def weighted_qcut(values, weights, q, **kwargs):
    'Return weighted quantile cuts from a given series, values.'
    if is_integer(q):
        quantiles = np.linspace(0, 1, q + 1)
    else:
        quantiles = q
    order = weights.iloc[values.argsort()].cumsum()
    bins = pd.cut(order / order.iloc[-1], quantiles, **kwargs)

    return bins.sort_index()



def grafico10(df, target, variable , weight, estimate = '', weight_label = '', 
              target_label = '', estimate_label= '', fill_na = -1, title = '', n_bins = 10):    
    # import matplotlib.pyplot as plt
    # import plotly.graph_objects as go
    # from plotly.subplots import make_subplots
    # import pandas as pd
    # import numpy as np
    
    
    X = df.copy()
    
    # WEIGHT
    if weight != '' :
        X['COUNT'] = X[weight]
    else: 
        X['COUNT'] = 1

    # FILLNA 
    X.fillna(value = {variable : fill_na}, inplace = True)
        
        

    # TESTANDO FUNCAO PARAR CRIAR OS CORTES EM QUANTILES PONDERADOS

    # X['bins'] = pd.qcut(X[variable] , n_bins, labels= False) 

    X['bins'] = weighted_qcut(X[variable], X['COUNT'], n_bins, labels=False)


    
    if estimate == '' :
        x_plot = X[['bins', 'COUNT', target]].groupby('bins')\
            .agg({'COUNT': 'sum', target : 'sum'}).reset_index()    
        x_plot.rename(columns = {target : 'SUM_EVENTS'}, inplace = True)
        x_plot[target] = x_plot['SUM_EVENTS']/x_plot['COUNT']
        x_plot['base'] = X[target].sum()/X['COUNT'].sum()
        
    else:
        X['estimate_weight'] = X[estimate]*X[weight]

        x_plot = X[['bins', 'COUNT', target, 'estimate_weight']].groupby('bins')\
            .agg({'COUNT': 'sum', target : 'sum', 'estimate_weight': 'sum'}).reset_index()    
        
        x_plot.rename(columns = {target : 'SUM_EVENTS', 'estimate_weight': 'SUM_ESTIMATE'}, inplace = True)

        x_plot[target] = x_plot['SUM_EVENTS']/x_plot['COUNT']
        x_plot[estimate] = x_plot['SUM_ESTIMATE']/x_plot['COUNT']

        x_plot['base'] = X[target].sum()/X['COUNT'].sum()

        x_plot[estimate + 'base'] = X['estimate_weight'].sum()/X['COUNT'].sum()
        
#     x_plot['base'] = X[target].mean()    
#     print(x_plot)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    

    fig.add_trace(
        go.Scatter(x=x_plot['bins'], y=x_plot[target], name= (target_label 
                                                                if target_label != '' else target),
                   line = dict(color='firebrick')),
        secondary_y=True
    )
    
    
    fig.add_trace(
        go.Scatter(x=x_plot['bins'], y=x_plot['base'], name=(target_label 
                                                                if target_label != '' else target) + "_base",
                   mode='lines',
                   line = dict(color='gray', dash = 'dash' )),
        secondary_y=True,
    )  
    
    
    if estimate != '' : 
        fig.add_trace(
            go.Scatter(x=x_plot['bins'], y=x_plot[estimate], name= (estimate_label 
                                                                    if estimate_label != '' else estimate),
                       line = dict(color='royalblue')),
            secondary_y=True
        )

        fig.add_trace(
            go.Scatter(x=x_plot['bins'], y=x_plot[estimate+'base'], name= (estimate_label 
                                                                        if estimate_label != '' else estimate)+"_base",
                       mode='lines',
                       line = dict(color='rgb(100, 100, 100)', dash = 'dash' )),
            secondary_y=True,
        )


    # Add traces
    fig.add_trace(
        go.Bar(x=x_plot['bins'] , y=x_plot['COUNT'], name= (weight_label 
                                                                if weight_label != '' else weight), 
            #    marker_color = 'rgb(12, 123, 50)', 
            #    marker_line_color = 'rgb(0, 51, 3)',
               marker_color = 'rgba(53,53,53,0.2)', 
               marker_line_color = 'rgba(53,53,53,1)',
               marker_line_width = 1,
               opacity= 1
              ),
        secondary_y=False,
    )

    fig.update_xaxes(title_text="DECIS por " + " " + variable)
    
    # fig.show()

    fig.update_layout(
        plot_bgcolor='white',
        autosize=True,
        width=700,
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        title = title
    )
    fig.update_yaxes(tickformat=".1%", secondary_y= True )


    
    return x_plot, fig



# GRAFICO DE FREQUENCIA OBSERVADA POR NIVEL DE VARIAVEL CATEGORICA
def graficoCat(df, target, variable , weight, estimate = '', weight_label = '', fill_na = -1, 
               target_label = '', var2 = '', var2_label = '', title = '' ):
    # import matplotlib.pyplot as plt
    # import plotly.graph_objects as go
    # from plotly.subplots import make_subplots
    # import pandas as pd
    # import numpy as np

    # WEIGHT
    X = df.copy()
    if weight != '' :
        X['COUNT'] = X[weight]
    else: 
        X['COUNT'] = 1

    
    # FILLNA 
    X.fillna(value = {variable : fill_na}, inplace = True)
    
    # AGREGAR VARIAVEIS
    if var2 == '':
        x_plot = X[[variable, 'COUNT', target]].groupby(variable, dropna = False)\
            .agg({'COUNT': 'sum', target : 'sum'}).reset_index()
        
        x_plot.rename(columns = {target : 'SUM_EVENTS'}, inplace = True)
        x_plot[target] = x_plot['SUM_EVENTS']/x_plot['COUNT']
        x_plot['base'] = X[target].sum()/X['COUNT'].sum()
    else:
        x_plot = X[[variable, 'COUNT', target, var2]].groupby([variable], dropna = False)\
            .agg({'COUNT': 'sum', target : 'sum', var2: 'sum'}).reset_index()
        
        x_plot.rename(columns = {target : 'SUM_EVENTS', var2: 'SUM_VAR2'}, inplace = True)
        x_plot[target] = x_plot['SUM_EVENTS']/x_plot['COUNT']
        x_plot[var2] = x_plot['SUM_VAR2']/x_plot['COUNT']
        x_plot['base'] = X[target].sum()/X['COUNT'].sum()
        x_plot[var2+'base'] = X[var2].sum()/X['COUNT'].sum()


    if estimate == '':
        x_plot = X[[variable, 'COUNT', target]].groupby(variable, dropna = False)\
            .agg({'COUNT': 'sum', target : 'sum'}).reset_index()
        
        x_plot.rename(columns = {target : 'SUM_EVENTS'}, inplace = True)
        x_plot[target] = x_plot['SUM_EVENTS']/x_plot['COUNT']
        x_plot['base'] = X[target].sum()/X['COUNT'].sum()
    else:

        X['estimate_weight'] = X[estimate]*X[weight]

        x_plot = X[[variable, 'COUNT', target, 'estimate_weight']].groupby([variable], dropna = False)\
            .agg({'COUNT': 'sum', target : 'sum', 'estimate_weight': 'sum'}).reset_index()
        
        x_plot.rename(columns = {target : 'SUM_EVENTS', 'estimate_weight': 'SUM_estimate'}, inplace = True)
        x_plot[target] = x_plot['SUM_EVENTS']/x_plot['COUNT']
        x_plot[estimate] = x_plot['SUM_estimate']/x_plot['COUNT']
        x_plot['base'] = X[target].sum()/X['COUNT'].sum()
        x_plot[estimate+'base'] = X['estimate_weight'].sum()/X['COUNT'].sum()
    
#     print(x_plot)
    
    # PLOT
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=x_plot[variable], y=x_plot[target], name= (target_label 
                                                                if target_label != '' else target),
                   line = dict(color='firebrick')),
        secondary_y=True
    )
    
    
    fig.add_trace(
        go.Scatter(x=x_plot[variable], y=x_plot['base'], name= (target_label 
                                                                if target_label != '' else target)+"_base",
                   mode='lines',
                   line = dict(color='gray', dash = 'dash' )),
        secondary_y=True,
    )

    # Add traces
    if var2 != '' :
        fig.add_trace(
            go.Scatter(x=x_plot[variable], y=x_plot[var2], name= (var2_label 
                                                                if var2_label != '' else var2),
                       line = dict(color='royalblue')),
            secondary_y=False
        )

        fig.add_trace(
        go.Scatter(x=x_plot[variable], y=x_plot[var2+'base'], name= (var2_label 
                                                                if var2_label != '' else var2)+"_base",
                   mode='lines',
                   line = dict(color='rgb(100, 100, 100)', dash = 'dash' )),
        secondary_y=False,
        )
    else:
        fig.add_trace(
            go.Bar(x=x_plot[variable] , y=x_plot['COUNT'], name=(weight_label 
                                                                if weight_label != '' else weight), 
                # marker_color = 'rgb(12, 123, 50)', 
                # marker_line_color = 'rgb(0, 51, 3)',
               marker_color = 'rgba(53,53,53,0.2)', 
               marker_line_color = 'rgba(53,53,53,1)',
               marker_line_width = 1,
               opacity= 1
            ),
            secondary_y=False,
        )
    
    # Add traces
    if estimate != '' :
        fig.add_trace(
            go.Scatter(x=x_plot[variable], y=x_plot[estimate], name= (var2_label 
                                                                if var2_label != '' else estimate),
                       line = dict(color='royalblue')),
            secondary_y=True
        )

        fig.add_trace(
        go.Scatter(x=x_plot[variable], y=x_plot[estimate+'base'], name= (var2_label 
                                                                if var2_label != '' else estimate)+"_base",
                   mode='lines',
                   line = dict(color='rgb(100, 100, 100)', dash = 'dash' )),
        secondary_y=True,
        )
    # else:
    #     fig.add_trace(
    #         go.Bar(x=x_plot[variable] , y=x_plot['COUNT'], name=(weight_label 
    #                                                             if weight_label != '' else weight), 
    #             # marker_color = 'rgb(12, 123, 50)', 
    #             # marker_line_color = 'rgb(0, 51, 3)',
    #            marker_color = 'rgba(53,53,53,0.2)', 
    #            marker_line_color = 'rgba(53,53,53,1)',
    #            marker_line_width = 1,
    #             opacity= 0.4
    #             ),
    #         secondary_y=False,
    #     )

    fig.update_xaxes(title_text=f"{target} por {variable}")
    
    fig.update_layout(
        plot_bgcolor='white',
        autosize=True,
        width=700,
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        title = title
    )

    fig.update_yaxes(tickformat=".1%", secondary_y= True )


    # fig.show()
    
    return fig



"""
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# from https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
def heatmap(x, y, **kwargs):
    
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 


def corrplot(data, size_scale=500, marker='s'):
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )

    """