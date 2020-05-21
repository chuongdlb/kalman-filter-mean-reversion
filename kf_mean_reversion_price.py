
import datetime as dt
import pandas as pd
from pykalman import KalmanFilter
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import numpy as np
from matplotlib import style 
import json
from pandas.io.json import json_normalize

def draw_date_coloured_scatterplot(etfs, prices):
    """
    Create a scatterplot of the two ETF prices, which is
    coloured by the date of the price to indicate the 
    changing relationship between the sets of prices    
    """
    # Create a yellow-to-red colourmap where yellow indicates
    # early dates and red indicates later dates
    plen = len(prices)
    colour_map = plt.cm.get_cmap('YlOrRd')    
    colours = np.linspace(0.1, 1, plen)
    
    # Create the scatterplot object
    scatterplot = plt.scatter(
        prices[etfs[0]], prices[etfs[1]], 
        s=30, c=colours, cmap=colour_map, 
        edgecolor='k', alpha=0.8
    )
    
    # Add a colour bar for the date colouring and set the 
    # corresponding axis tick labels to equal string-formatted dates
    colourbar = plt.colorbar(scatterplot)
    colourbar.ax.set_yticklabels(
        [str(p.date()) for p in prices[::plen//9].index]
    )
    plt.xlabel(prices.columns[0])
    plt.ylabel(prices.columns[1])
    plt.show()

def calc_slope_intercept_kalman(means_prices, observed_prices):
    x = means_prices.iloc[:,0]
    # print(x)
    y = observed_prices.iloc[:,0]
    
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    print(obs_mat)
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # observed price y(t) is 1-dimensional, (m(t), R)/ (mean, state_covariance) is 2-dimensional
                    initial_state_mean=[0,0],
                    initial_state_covariance=np.ones((2, 2)),
                    transition_matrices=np.eye(2),
                    observation_matrices=obs_mat,
                    observation_covariance=1,
                    transition_covariance=trans_cov)

    state_means, state_covs = kf.filter(y)
    return state_means, state_covs    

def draw_slope_intercept_changes(prices, state_means):
    """
    Plot the slope and intercept changes from the 
    Kalman Filte calculated values.
    """
    pd.DataFrame(
        dict(
            slope=state_means[:, 0], 
            intercept=state_means[:, 1]
        ), index=prices.index
    ).plot(subplots=True)
    plt.show()
    
if __name__ == "__main__":

    df = json_normalize(pd.Series(open('./tradeevents_19_may.txt').readlines()).apply(json.loads))

    # df = pd.read_json('./tradeevents_19_may.csv', lines=True, convert_dates=False)
    # df = (pd.DataFrame(df['original_payload']).values)
    #df1 = df[['exchange', 'original_payload.amount', 'timestamp.$date']]
    # print(df1.head())

    trade_size_df = pd.DataFrame(df, columns=['original_payload.amount'])
    print(trade_size_df.describe())
    means_prices = pd.DataFrame(df, columns=['original_payload.index_price'])
    observed_prices = pd.DataFrame(df, columns=['original_payload.price'])
    
    state_means, state_covs = calc_slope_intercept_kalman(means_prices, observed_prices)

    draw_slope_intercept_changes(observed_prices.iloc[:,0], state_means)

    input("Press any key to close")
