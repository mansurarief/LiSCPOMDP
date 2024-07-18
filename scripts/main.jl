#=
Modeling the US Path to Lithium Self Sufficiency Using POMDPs
Summer 2023
Yasmine Alonso, Mansur Arief, Anthony Corso, Jef Caers, and Mykel Kochenderfer

File: main.jl
----------------
This file contains the code that runs and evaluates all of our policies, printing out necessary information to the console.
=#
using Random
using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools
using MCTS
using Serialization

Random.set_global_seed!(0)

include("LiPOMDP.jl")
include("utils.jl")
include("policies.jl")


# Initializing the POMDP, Belief Updater, and initial state, as well as the MDP version of the POMDP for MCTS
pomdp = LiPOMDP() #always use continous and use POMCPOW obs widening params to control the discretization
up = LiBeliefUpdater(pomdp)
s0 = pomdp.init_state
#b0 = initialize_belief(up, s0)
mdp = GenerativeBeliefMDP(pomdp, up)

# benchmark planners (from policies.jl)
random_planner = RandomPolicy(pomdp)
strong_planner = EfficiencyPolicy(pomdp, [true, true, true, true])
robust_planner = EfficiencyPolicyWithUncertainty(pomdp, 1., [true, true, true, true])
eco_planner = EmissionAwarePolicy(pomdp, [true, true, true, true])


#MCTS Solver -- uses mdp version of pomdp
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


# POMCPOW Solver
println("POMCPOW Solver")
solver = POMCPOWSolver(
    tree_queries=1000, 
    estimate_value = estimate_value,#RolloutEstimator(RandomPolicy(pomdp)), #estimate_value,
    k_observation=4., 
    alpha_observation=0.1, 
    max_depth=15, 
    enable_action_pw=false,
    init_N=10  
) # Estimate value should fix the previous problem with action functions
pomcpow_planner = solve(solver, pomdp)

planners = [random_planner, strong_planner, robust_planner, eco_planner, pomcpow_planner, mcts_planner] #

n_reps=20
max_steps=15

# Compares all the policies and prints out relevant information
# Read in utils.jl
evaluate_policies(pomdp, planners, n_reps, max_steps)