from utils import get_val_from_info
from plob import OrderBookEnv
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

import argparse
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy as DQNPolicyGraph
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG, ApexTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--custom-dir", default="/tmp/pycharm_project_719/log/", type=str)
parser.add_argument("--num-player", default=[(2, 1), (4, 2)], type=int)     # [] , (4, 2)(8, 4)
parser.add_argument("--maturity", default=10000, type=int)
parser.add_argument("--train-num", default=500, type=int) # 500
parser.add_argument("--test-num", default=10, type=int) # 10
parser.add_argument("--convergence-epsilon", default=0.0001, type=float)
parser.add_argument("--thetas", default=[
    # [0.3, 0.3, 0.4, 0.2, 0.2, 0.2],
    # [0.4, 0.4, 0.4, 0.0, 0.2, 0.2],
    # [0.4, 0.0, 0.5, 0.0, 0.2, 0.0],
    # [0.5, 0.5, 0.6, 0.0, 0.0, 0.0],
    # [0.5, 0.4, 0.6, 0.0, 0.0, 0.0],
    # [0.5, 0.3, 0.6, 0.0, 0.0, 0.0],
    # [0.5, 0.2, 0.6, 0.0, 0.0, 0.0],
    # [0.5, 0.1, 0.6, 0.0, 0.0, 0.0],
    # [0.5, 0.0, 0.6, 0.0, 0.0, 0.0]
    [0.5, 0.5, 0.2, 0.0, 0.0, 0.0],
])
parser.add_argument("--sigmas", default=[1])  #  [1, 0.6, 0.3]
parser.add_argument("--delays", default=[10])  # [5, 10, 15]


if __name__ == "__main__":
    args = parser.parse_args()
    custom_dir = args.custom_dir
    num_player_list = args.num_player
    maturity = args.maturity
    train_num = args.train_num
    test_num = args.test_num
    convergence_epsilon = args.convergence_epsilon
    theta_list = args.thetas
    sigma_list = args.sigmas
    delay_list = args.delays


    def train_test(num_player, num_informed, theta, sigma_V, delay, grid):
        ray.shutdown()
        info = ray.init(ignore_reinit_error=True, log_to_driver=False, include_dashboard=False, num_cpus=48, num_gpus=6)
        print(info)
        assert ray.is_initialized() == True

        print('test set:', num_player, num_informed, theta, sigma_V, delay, grid)
        register_env("multi_agent_order_book",
                     lambda _: OrderBookEnv(maturity=maturity, num_informed=num_informed,
                                            num_player=num_player, max_depth=4, theta=theta, sigma_V=sigma_V,
                                            delay=delay, grid=grid))
        game = OrderBookEnv(maturity=maturity, num_informed=num_informed, num_player=num_player,
                            max_depth=4, theta=theta, sigma_V=sigma_V, delay=delay, grid=grid)
        act_space = game.action_space
        obs_space = game.observation_space
        agents = game.agents

        config = APEX_DEFAULT_CONFIG.copy()
        config['num_workers'] = 24
        config["num_envs_per_worker"] = 2
        config['num_cpus_per_worker'] = 2

        config['num_atoms'] = 51
        config['noisy'] = True
        config["n_step"] = 10
        config["v_min"] = -max(game.P)
        config["v_max"] = max(game.P)
        config['model'] = {
            # "fcnet_activation": "relu",
            # "fcnet_hiddens": [32, 32, 32],
            "fcnet_hiddens": [64, 64],
            # "fcnet_hiddens": [256, 256],
        }
        config["lr"] = 0.001
        config["gamma"] = game.gamma
        config["multiagent"] = {
            # "policies": {
            #     "dqn_policy1": (DQNPolicyGraph, obs_space[agents[0]], act_space[agents[0]], {"gamma": game.gamma}),
            #     "dqn_policy2": (DQNPolicyGraph, obs_space[agents[1]], act_space[agents[1]], {"gamma": game.gamma}),
            # },
            # "policies_to_train": ["dqn_policy1", "dqn_policy2"],
            # "policy_mapping_fn":
            #     lambda agent_id:
            #     "dqn_policy1"
            #     if int(agent_id.split('_')[1]) < num_informed
            #     else "dqn_policy2",
            "policies": {
                "dqn_policy"+str(i): (DQNPolicyGraph, obs_space[agent], act_space[agent], {"gamma": game.gamma})
                for i, agent in enumerate(agents)
            },
            "policies_to_train": ["dqn_policy"+str(i) for i, agent in enumerate(agents)],
            "policy_mapping_fn": lambda agent_id: "dqn_policy" + agent_id.split('_')[1]
        }

        trainer = ApexTrainer(env="multi_agent_order_book", config=config)
        # Train
        # if os.path.exists("/tmp/pycharm_project_719/log/checkpoint_000902/"):
        #     trainer.restore("/tmp/pycharm_project_719/log/checkpoint_000902/checkpoint-902")
        #     train_num = 0
        checkpoint = None
        train_memory = np.zeros((num_player+2, train_num))
        check_break = 0
        prior_episode = 0
        for i in tqdm(range(train_num)):
            # Improve the DQN policy

            check_nan = 0
            while check_nan == 0:
                stats = trainer.train()
                check_nan = len(stats["policy_reward_mean"])

            print("== Iteration", i, "== Episode", stats["episodes_total"], "== AvgReward:",
                  stats["policy_reward_mean"],
                  "== AvgLength:", stats["episode_len_mean"])
            train_memory[0][i] = stats["episodes_total"]
            for j in range(num_player):
                train_memory[1+j][i] = stats["policy_reward_mean"]['dqn_policy'+str(j)]
            # train_memory[1][i] = stats["policy_reward_mean"]['dqn_policy1']
            # train_memory[2][i] = stats["policy_reward_mean"]['dqn_policy2']
            train_memory[-1][i] = stats["episode_len_mean"]

            if i % 200 == 0:
                checkpoint = trainer.save(custom_dir + "checkpoint_%s_%s__%s_%s_with_%s_%s_%s_%s.txt"
                                          % (num_player, num_informed, train_num, test_num, theta, sigma_V, delay, grid))

            if (abs(train_memory[1][i - 1] - train_memory[1][i]) < convergence_epsilon) and (
                    abs(train_memory[2][i - 1] - train_memory[2][i]) < convergence_epsilon):
                check_break += 1
            elif abs(prior_episode - stats["episode_reward_mean"]) < convergence_epsilon:
                check_break += 1
            else:
                check_break = 0
            prior_episode = stats["episode_reward_mean"]

            if check_break >= 10:
                checkpoint = trainer.save(custom_dir)
                print("Convergence Criteria Satisfied!!")
                break

        checkpoint = trainer.save(custom_dir + "checkpoint_%s_%s__%s_%s_with_%s_%s_%s_%s.txt"
                                  % (num_player, num_informed, train_num, test_num, theta, sigma_V, delay, grid))
        if not checkpoint:
            with open(custom_dir + "checkpoint_%s_%s__%s_%s_with_%s_%s_%s_%s.txt"
                      % (num_player, num_informed, train_num, test_num, theta, sigma_V, delay, grid), 'w') as f:
                f.write(checkpoint)

        temp = ['episodes_total'] + ['dqn_policy' + str(j) for j in range(num_player)] + ["episode_len_mean"]
        df_policy = pd.DataFrame(train_memory,
                                 index=temp)
        df_policy.to_csv(custom_dir
                         + "apex_policy_result_train_%s_%s__%s_%s_with_%s_%s_%s_%s.csv"
                         % (num_player, num_informed, train_num, test_num, theta, sigma_V, delay, grid))
        # trainer.restore(checkpoint)

        # Test
        print("test starts!!!")
        agent_names = ["agent_%d" % i for i in range(num_player)]
        policy_map = trainer.config["multiagent"]["policy_mapping_fn"]

        episode_info_list = []
        total_info_log = []
        for _ in tqdm(range(test_num)):
            # print("Game ",i)
            obs = game.reset()
            # game.render()
            dones = {}

            R = np.zeros(num_player)
            R_list = []
            t = 0
            action_dict = {}
            info_list = []
            # pbars = tqdm(total=maturity)
            while not dones.get("__all__", False):
                # Get actions from neural network
                action_dict = {x: trainer.compute_action(obs[x], policy_id=policy_map(x)) for x in agent_names}

                # Make move
                obs, rewards, dones, info = game.step(action_dict)

                # Add rewards up
                rew = np.array(list(rewards.values()))
                R += rew
                R_list.append(rew)
                t += 1
                # Check info
                info_list.append(get_val_from_info(info, num_informed))
            #     pbars.update(1)
            # pbars.close()

            # Episode information
            info_list = np.array(info_list, dtype=np.float)
            transactions = np.sum(info_list[:, :2], axis=0)
            zero_orders = np.sum(info_list[:, 2:4], axis=0)
            temp = info_list[:, 4:]
            info_average = np.mean(temp, axis=0)
            info_std = np.std(temp, axis=0)
            R_std = np.std(R_list, axis=0)
            episode_info_list.append(np.concatenate((transactions, zero_orders, info_average, info_std, R, R_std)))
            total_info_log.append(info_list)

        df = pd.DataFrame(episode_info_list, columns=[
            "transaction_num1", "transaction_num2", "zero_order1", "zero_order2",
            "spread_mean", "bid_depth_mean", "ask_depth_mean",
            "cumulative_bid_depth_mean", "cumulative_ask_depth_mean",
            "large_buy_limit_mean", "large_sell_limit_mean",
            "spread_std", "bid_depth_std", "ask_depth_std",
            "cumulative_bid_depth_std", "cumulative_ask_depth_std",
            "large_buy_limit_std", "large_sell_limit_std"]
                                                     + ["reward"+str(i) for i in range(num_player)]
                                                     + ["reward"+str(i)+"_std" for i in range(num_player)]
        )
        df.to_csv(custom_dir +
                  "results_information_%s_%s_train_%s_test_%s_with_%s_%s_%s_%s.csv"
                  % (num_player, num_informed, train_num, test_num, theta, sigma_V, delay, grid))

        ray.shutdown()
        assert ray.is_initialized() == False


    for num_player, num_informed in num_player_list:
        for theta in theta_list:
            for sigma_V in sigma_list:  # [1, 0.6, 0.3]
                for delay in delay_list:    # [5, 10, 15]
                    for grid in [0, 1, 2, 3]:
                        train_test(num_player, num_informed, theta, sigma_V, delay, grid)

    ######################################
    #
    # def custom_log_creator(custom_path, custom_str):
    #
    #     timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    #     logdir_prefix = "{}_{}".format(custom_str, timestr)
    #
    #     def logger_creator(_config):
    #         if not os.path.exists(custom_path):
    #             os.makedirs(custom_path)
    #         logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
    #         return UnifiedLogger(_config, logdir, loggers=None)
    #
    #     return logger_creator
    #
    # trainer = ApexTrainer(env="multi_agent_order_book", config=config)
    # def trainer(config, reporter):
    #     agent = ApexTrainer(
    #         env="multi_agent_order_book", config=config,
    #         # logger_creator=custom_log_creator(os.path.expanduser(custom_dir), 'custom_dir')
    #     )
    #     # agent.restore()  # continue training
    #
    #     i = 0
    #     while True:
    #         result = agent.train()
    #         checkpoint_path = agent.save()
    #         print('checkpoint_path: ', checkpoint_path)
    #         if len(result["policy_reward_mean"]) == 0:
    #             continue
    #         if reporter is None:
    #             continue
    #         else:
    #             reporter(**result)
    #
    #         # if i % 10 == 0:  # save every 10th training iteration
    #         #     checkpoint_path = agent.save()
    #         #     # print(checkpoint_path)
    #         i += 1
    #
    #
    # trainingSteps = 5
    # results = tune.run(
    #     ApexTrainer,
    #     config=config,
    #     # resources_per_trial={
    #     #     "cpu": 48,
    #     #     "gpu": 1,
    #     #     "extra_cpu": 0,
    #     # },
    #     stop={
    #         "training_iteration": trainingSteps,
    #     },
    #     metric="episode_reward_mean",
    #     mode="max",
    #     # log_to_file=True,
    #     local_dir="/tmp/pycharm_project_719/log",
    #     # checkpoint_freq=1,
    #     # keep_checkpoints_num=1,
    #     # checkpoint_at_end=True,
    #     # checkpoint_score_attr="policy_reward_mean",
    #     checkpoint_at_end=True
    # )
    #
    # print(1)
    # agent = ApexTrainer(
    #     env="multi_agent_order_book", config=config,
    #     # logger_creator=custom_log_creator(os.path.expanduser(custom_dir), 'custom_dir')
    # )
    # checkpoint_path = results.get_trial_checkpoints_paths(results.get_best_trial("policy_reward_mean", mode='max'))
    # checkpoint_path2 = results.get_best_trial(metric="policy_reward_mean", mode="max", scope="all")
    # checkpoint_path = results.best_checkpoint
    # results.get_best_checkpoint(trial=results.best_trial, metric="episode_reward_mean", mode="max")
    #
    # agent.restore(checkpoint_path)
    # results.save_checkpoint()
    # ray.tune.analysis.experiment_analysis.ExperimentAnalysis
