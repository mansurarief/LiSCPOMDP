using Random
using POMDPs 
using POMDPTools
using POMDPPolicies
using LiPOMDPs
using MCTS
using DiscreteValueIteration
using POMCPOW 
using Distributions
using Parameters
using ARDESPOT
# using Iterators

# Random.set_global_seed!(0)

rng = MersenneTwister(1)

# Initializing the POMDP, Belief Updater, and initial state, as well as the MDP version of the POMDP for MCTS
pomdp = initialize_lipomdp() 
up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)
policy = RandomPolicy(pomdp)
a = action(policy, b)
rng = MersenneTwister(0)
s = rand(initialstate(pomdp))

# c = states(pomdp)

println(fieldnames(typeof(initialstate(pomdp).val)))


sp, o, r = gen(pomdp, s, a, rng)

mdp = GenerativeBeliefMDP(pomdp, up)

random_planner = RandPolicy(pomdp)
strong_planner = EfficiencyPolicy(pomdp, [true, true, true, true])
robust_planner = EfficiencyPolicyWithUncertainty(pomdp, 1., [true, true, true, true])
eco_planner = EmissionAwarePolicy(pomdp, [true, true, true, true])

# #MCTS Solver -- uses mdp version of pomdp
mcts_solver = DPWSolver(
     depth=8,
     n_iterations = 100,
     estimate_value=RolloutEstimator(robust_planner, max_depth=100),
     enable_action_pw=false,
     enable_state_pw=true,
     k_state = 4.,
    alpha_state = 0.1,
 )
mcts_planner = solve(mcts_solver, mdp)

despot_solver = DESPOTSolver(bounds=(-1000.0, 1000.0))
despot_planner = solve(despot_solver, pomdp)

# # POMCPOW Solver
solver = POMCPOW.POMCPOWSolver(
     tree_queries=1000, 
     estimate_value = estimate_value, #RolloutEstimator(RandomPolicy(pomdp)), #estimate_value,
     k_observation=4., 
     alpha_observation=0.1, 
     max_depth=15, 
     enable_action_pw=false,
     init_N=10  
 ) # Estimate value should fix the previous problem with action functions
pomcpow_planner = solve(solver, pomdp)

n_reps=20
max_steps=15

planners = [pomcpow_planner, despot_planner, mcts_planner, random_planner, strong_planner, robust_planner, eco_planner]
for planner in planners
    println(" ")
    println("=====Simulating ", typeof(planner), "=====")
    println(" ")
    for (s, a, o, r) in stepthrough(pomdp, planner, "s,a,o,r", max_steps=max_steps)
        println("in state $s")
        println("took action $a")
        println("received observation $o and reward $r")
    end
end