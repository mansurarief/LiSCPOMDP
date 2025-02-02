using Random
using POMDPs 
using POMDPTools
using LiPOMDPs
using MCTS
using DiscreteValueIteration
using POMCPOW 
using Distributions

pomdp = initialize_lipomdp(obj_weights=[0.25, 0.25, 1.0, 1.0, 0.25]) 
up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)
mdp = GenerativeBeliefMDP(pomdp, up)
max_steps = 2
random_planner = RandPolicy(pomdp)

for (s, a, o, r) in stepthrough(pomdp, random_planner, "s,a,o,r", max_steps=max_steps)
    println("in state $s")
    println("took action $a")
    println("received observation $o and reward $r")
end