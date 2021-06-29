import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


current_dir = os.getcwd()
current_dir += '/log'
learning_curve_list = glob.glob(current_dir + '/apex*.csv')
result_list = glob.glob(current_dir + '/results*.csv')


def filename_to_index_names(file_name):
    if 'apex' in file_name:
        try:
            num_player, num_informed = list(map(int, file_name.split('_train')[0].split('_')[-2:]))
        except ValueError:
            num_player, num_informed = (2, 1)
    elif 'results_information' in file_name:
        try:
            num_player, num_informed = list(map(int, file_name.split('_train')[0].split('_')[-2:]))
        except ValueError:
            num_player, num_informed = (2, 1)
    else:
        raise ValueError('Not Valid Filename')

    theta, sigma, delay, tick = file_name.replace('.csv', '').split('_')[-4:]
    theta = theta.replace('[', '').replace(']', '').split(', ')
    theta = [np.float(_) for _ in theta]
    sigma = np.float(sigma)
    delay = int(delay)
    tick = int(tick)
    return num_player, num_informed, theta, sigma, delay, tick


def make_figure(df, figure_name):
    # plot
    plt.figure(figsize=(12, 6))
    for i in range(1, len(df.columns)-1):
        plt.plot(df['episodes'], df.iloc[:, i])
    # plt.plot(df['episodes'], df['total'])
    plt.xlabel('Episodes')
    plt.ylabel('Rewards per Arrival')
    plt.legend(list(df.columns[1:-1]))
    plt.savefig(figure_name, dpi=400)
    # plt.show()
    plt.close()


def make_df_from_results(i, file_name):
    df = pd.read_csv(file_name, index_col=0)
    df['transaction_num_informed'] = df['transaction_num1'] - df['zero_order1']
    df['transaction_num_predatory'] = df['transaction_num2'] - df['zero_order2']
    df['transaction_ratio_informed'] = df['transaction_num_informed'] / df['transaction_num1']
    df['transaction_ratio_predatory'] = df['transaction_num_predatory'] / df['transaction_num2']

    df1 = df[['transaction_num_informed', 'transaction_num_predatory',
              'transaction_ratio_informed', 'transaction_ratio_predatory', 'transaction_num1', 'transaction_num2',
              'spread_mean', 'bid_depth_mean', 'ask_depth_mean',
              'cumulative_bid_depth_mean', 'cumulative_ask_depth_mean',
              'large_buy_limit_mean', 'large_sell_limit_mean']].T
    df2 = df[['bid_depth_std', 'ask_depth_std', 'cumulative_bid_depth_std',
              'cumulative_ask_depth_std', 'large_buy_limit_std',
              'large_sell_limit_std']].T
    df3 = pd.DataFrame(data=pd.concat([df1.mean(axis=1), np.sqrt(np.square(df2).mean(axis=1))]))
    df3.columns = [i]
    return df3


def gen_num_transaction(df_result, i):
    return df_result.T['transaction_num1'][i], df_result.T['transaction_num2'][i]


def make_df_from_learning(i, file_name, num_transaction):
    num_player, num_informed, theta, sigma, delay, tick = filename_to_index_names(file_name)
    informed_num, predatory_num = num_transaction

    df = pd.read_csv(file_name, index_col=0)
    df = df.iloc[:, :500].T
    df.columns = ['episodes'] + ['informed trader' + str(i) for i in range(num_informed)] \
                 + ['predatory trader' + str(i) for i in range(num_player - num_informed)] + ['episode_len_mean']
    print('episodes: ', df['episodes'][-1])
    figure_name = os.getcwd() + '/figure/' + str(num_player) + '_' + str(num_informed) + '_' \
                  + file_name.split('\\')[-1].split('with_')[-1].split('.csv')[0] + '.jpg'
    make_figure(df, figure_name)

    # get statistics
    df = df[df['episodes'] > df['episodes'][-1] - 1000]
    print('df len: ', len(df))

    reward_informed = np.average([df['informed trader'+str(_)].mean() for _ in range(num_informed)])
    reward_predatory = np.average([df['predatory trader'+str(_)].mean() for _ in range(num_player - num_informed)])

    reward_std_informed = np.sqrt(np.average([df['informed trader'+str(_)].std() ** 2 for _ in range(num_informed)]))
    reward_sharpe_informed = reward_informed / reward_std_informed

    reward_std_predatory = np.sqrt(np.average([df['predatory trader'+str(_)].std() ** 2 for _ in range(num_informed)]))
    reward_sharpe_predatory = reward_predatory / reward_std_predatory

    reward_total = reward_informed * informed_num + reward_predatory * predatory_num
    temp = np.array([informed_num * df['informed trader'+str(_)].values for _ in range(num_informed)]) \
           + np.array([predatory_num * df['predatory trader'+str(_)].values for _ in range(num_player - num_informed)])
    reward_std_total = np.std(np.array(temp).flatten())
    reward_sharpe_total = reward_total / reward_std_total

    data = np.array([
        sigma, delay, tick, reward_informed, reward_std_informed, reward_sharpe_informed,
        reward_predatory, reward_std_predatory, reward_sharpe_predatory,
        reward_total, reward_std_total, reward_sharpe_total
    ])

    data = np.concatenate((np.array([num_player, num_informed]), np.array(theta), data))
    df_summary = pd.DataFrame(data, index=[
        'num_player', 'num_informed',
        'sigma_1', 'sigma_2', 'sigma_3', 'sigma_12', 'sigma_13', 'sigma_23',
        'sigma', 'delay', 'tick', 'reward_informed', 'reward_std_informed', 'reward_sharpe_informed',
        'reward_predatory', 'reward_std_predatory', 'reward_sharpe_predatory',
        'reward_total', 'reward_std_total', 'reward_sharpe_total'
        ])
    df_summary.columns = [i]
    return df_summary


df_all = pd.DataFrame()
for i, (file_learning, file_result) in enumerate(zip(learning_curve_list, result_list)):
    assert filename_to_index_names(file_learning) == filename_to_index_names(file_result)
    print(filename_to_index_names(file_learning))

    df_result = make_df_from_results(i, file_name=file_result)
    df_learning = make_df_from_learning(i, file_name=file_learning, num_transaction=gen_num_transaction(df_result, i))
    df_all = pd.merge(df_all, pd.concat([df_learning, df_result]), how='outer', left_index=True, right_index=True)
df_all.to_csv(os.getcwd() + '/figure/summary_table2.csv')
