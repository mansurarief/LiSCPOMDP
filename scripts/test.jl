using Random
using POMDPs 
using POMDPTools
using POMDPPolicies
#using LiPOMDPs
using MCTS
using DiscreteValueIteration
using POMCPOW 
using Distributions
using Parameters
using Plots
using Plots.PlotMeasures

rng = MersenneTwister(1)
pomdp = initialize_lipomdp(obj_weights=[0.25, 0.25, 1.0, 1.0, 0.25]) 
up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)
mdp = GenerativeBeliefMDP(pomdp, up)