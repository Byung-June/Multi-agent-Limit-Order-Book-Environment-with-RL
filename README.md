# Multi-agent-Limit-Order-Book-Environment-with-RL

## Code for the Doctoral Dissertation

This code is for 'Effects of Tick Size Change on PredatoryTraders and Market Quality,' which is the doctoral dissertation of Byung-June Kim at POSTECH.
Byung-June Kim conducts Conceptualization, Methodology, Software, Analysis, and Writing. His CV site is https://sites.google.com/view/kbj219/

My supervisor is professor Bong-Gyu Jang at POSTECH.
This work was supported by the Ministry of Education of the Republic of Korea and the National Research Foundation of Korea. (NRF-2019S1A5A2A03054249)

## Summary of the Paper

I model a dynamic public limit order market in which three types of traders - informed traders, predatory traders, and noise traders - arrive and choose orders. I adopt the reinforcement learning algorithm (Ape-X Rainbow DQN) to scrutinize the formation and the characteristics of the market equilibrium among multiple traders of the model. The ensemble of optimal strategies of multiple traders determines the market equilibrium, given market conditions, such as tick size, value dynamics, or types of traders. This study explores the relationship between these market conditions and the market quality, including liquidity measures and gains of individual traders. There are three principal findings with numerous simulations. First, predatory trading deteriorates the payoff of each trader and the total profit of the market. Second, multiple homogeneous agents in each heterogeneous group would have different simulation results with a representative agent of each heterogeneous trader type; the assumption of the representative agents might overestimate the payoff of the market participants. Third, the market participation of the noise traders has a key role in the effect of predatory traders on liquidity measures for large orders.

## File Structure

1) Main files \
apex.py: the main file to execute experiments with Ape-X Rainbow DQN \
dqn.py: the main file to execute experiments with DQN 

2) Simulation environment \
plob.py: the RL environment based on MultiAgentEnv of ray.rllib 

3) Analysis \
file_name_convert.py -> analysis.py

4) etc \
gen_state.py: generate que of market arrivals and value dynamics \
utils: additional modules for plob.py

