
using Random
using POMDPs
using POMDPTools
using LiSCPOMDP
using MCTS
using DiscreteValueIteration

Random.set_global_seed!(0)


# Initializing the POMDP, Belief Updater, and initial state, as well as the MDP version of the POMDP for MCTS
pomdp = LiPOMDP() #always use continous and use POMCPOW obs widening params to control the discretization
up = LiBeliefUpdater(pomdp)
s0 = pomdp.init_state
#b0 = initialize_belief(up, s0)
mdp = GenerativeBeliefMDP(pomdp, up)

