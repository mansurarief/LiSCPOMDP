using Random
using POMDPs 
using POMDPTools
using LiPOMDPs
using MCTS
using DiscreteValueIteration
using POMDPPolicies


Random.set_global_seed!(0)


# Initializing the POMDP, Belief Updater, and initial state, as well as the MDP version of the POMDP for MCTS
pomdp = LiPOMDP() #always use continous and use POMCPOW obs widening params to control the discretization
up = LiBeliefUpdater(pomdp)


policy = RandomPolicy(pomdp)
s0 = pomdp.init_state
println(typeof(up))
println(typeof(s0))
b0 = initialize_belief(up, s0)
