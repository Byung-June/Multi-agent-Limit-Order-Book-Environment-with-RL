from plob import OrderBookEnv

import numpy as np
from tqdm import tqdm
import argparse
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.agents.dqn.dqn import DQNTrainer as DQNAgent
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy as DQNPolicyGraph


parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    info = ray.init(ignore_reinit_error=True, log_to_driver=False, include_dashboard=False)
    print(info)

    num_informed = 1
    maturity = 10000

    register_env("multi_agent_order_book",
                 lambda _: OrderBookEnv(maturity=maturity, penalize=False, num_informed=num_informed, max_depth=4))
    game = OrderBookEnv(maturity=maturity, penalize=False, num_informed=num_informed, max_depth=4)
    act_space = game.action_space
    obs_space = game.observation_space
    agents = game.agents

    config = DEFAULT_CONFIG.copy()
    # config['num_workers'] = 6
    # config["num_cpus_per_worker"] = 8

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
    config["exploration_config"]["type"] = "EpsilonGreedy"
    config['num_cpus_per_worker'] = 0
    config["lr"] = 0.001
    config["gamma"] = game.gamma
    config["multiagent"] = {
            "policies": {
                "dqn_policy1": (DQNPolicyGraph, obs_space[agents[0]], act_space[agents[0]], {"gamma": game.gamma}),
                "dqn_policy2": (DQNPolicyGraph, obs_space[agents[1]], act_space[agents[1]], {"gamma": game.gamma}),
            },
            "policies_to_train": ["dqn_policy1", "dqn_policy2"],
            "policy_mapping_fn":
                lambda agent_id:
                "dqn_policy1"
                if int(agent_id.split('_')[1]) < num_informed
                else "dqn_policy2"
        }

    trainer = DQNAgent(env="multi_agent_order_book", config=config)

    # Train
    N = 10
    mean_reward = np.zeros(N)
    mean_length = np.zeros(N)
    for i in tqdm(range(N)):
        # Improve the DQN policy
        stats = trainer.train()
        # if len(stats["policy_reward_mean"]) != 0:
        #     print(1)
        print("== Iteration", i, "==Episode", stats["episodes_total"], "== AvgReward:", stats["policy_reward_mean"],
              "== AvgLength:", stats["episode_len_mean"])
        if i % 100 == 0:
            checkpoint = trainer.save("/tmp/pycharm_project_719/log")
    trainer.restore(checkpoint)

    # Test
    agent_names = ["agent_%d" % (i + 1) for i in range(2)]
    policy_map = trainer.config["multiagent"]["policy_mapping_fn"]

    # N = 10
    # for i in tqdm(range(N)):
    #     # print("Game ",i)
    #     obs = game.reset()
    #     # game.render()
    #     dones = {}
    #
    #     avgR = np.zeros(2)
    #     R = np.zeros(2)
    #     t = 0
    #     action_dict = {}
    #     while not dones.get("__all__", False):
    #         # Get actions from neural network
    #         action_dict = {x: trainer.compute_action(obs[x], policy_id=policy_map(x)) for x in agent_names}
    #
    #         # Make move
    #         obs, rewards, dones, _ = game.step(action_dict)
    #
    #         # Add rewards up
    #         for q in rewards:
    #             index = int(q.split('_')[1]) - 1
    #             R[index] += rewards[q]
    #
    #         # game.render()
    #
    #     avgR = avgR + R
    #
    # print('average reward:', avgR / N)
