using Random
using POMDPs 
using POMDPTools
# using POMDPPolicies
using LiPOMDPs
using MCTS
# using DiscreteValueIteration
using POMCPOW 
using Distributions
using Parameters
using ARDESPOT
# using Iterators
using ParticleFilters

rng = MersenneTwister(1)

pomdp = initialize_lipomdp() 

# s = pomdp.init_state
# s.m = [true, false, false, false]

# a = rand(rng, actions(pomdp))
# sp, o, r = gen(pomdp, s, a, rng)

up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)
policy = RandomPolicy(pomdp)

# rand(rng, b)
# a = action(policy, b)
# rng = MersenneTwister(0)
# s = rand(initialstate(pomdp))

# c = states(pomdp)
# b
println(fieldnames(typeof(initialstate(pomdp).val)))


# sp, o, r = gen(pomdp, s, a, rng)

# mdp = GenerativeBeliefMDP(pomdp, up)

random_planner = RandPolicy(pomdp)
# # strong_planner = EfficiencyPolicy(pomdp, [true, true, true, true])
robust_planner = EfficiencyPolicyWithUncertainty(pomdp, 1., [true, true, true, true])
# # eco_planner = EmissionAwarePolicy(pomdp, [true, true, true, true])

# # # #MCTS Solver -- uses mdp version of pomdp
# mcts_solver = DPWSolver(
#      depth=8,
#      n_iterations = 100,
#      estimate_value=RolloutEstimator(robust_planner, max_depth=100),
#      enable_action_pw=false,
#      enable_state_pw=true,
#      k_state = 4.,
#     alpha_state = 0.1,
#  )
# mcts_planner = solve(mcts_solver, mdp)

# despot_solver = DESPOTSolver(bounds=(-20000.0, 20000.0))
# despot_planner = solve(despot_solver, pomdp)

# # # POMCPOW Solver
solver = POMCPOW.POMCPOWSolver(
     tree_queries=1000, 
     estimate_value = RolloutEstimator(RandomPolicy(pomdp)), #estimate_value,
     k_observation=4., 
     alpha_observation=0.1, 
     max_depth=30, 
     enable_action_pw=false,
     init_N=10  
 ) # Estimate value should fix the previous problem with action functions
pomcpow_planner = solve(solver, pomdp)

n_reps=3
max_steps=30

# planners = [random_planner, robust_planner, mcts_planner]
# for planner in planners
#     println(" ")
#     println("=====Simulating ", typeof(planner), "=====")
#     println(" ")
#     for (s, a, o, r) in stepthrough(pomdp, planner, up, b, "s,a,o,r", max_steps=max_steps)
#         println("in state $s")
#         println("took action $a")
#         println("received observation $o and reward $r")
#     end
# end

hr = HistoryRecorder(max_steps=max_steps)

@time random_hist = simulate(hr, pomdp, random_planner, up, b);
println("reward $(typeof(random_planner)): $(round(discounted_reward(random_hist), digits=2))")


@time robust_hist = simulate(hr, pomdp, robust_planner, up, b);
println("reward $(typeof(robust_planner)): $(round(discounted_reward(robust_hist), digits=2))")

@time phist = simulate(hr, pomdp, pomcpow_planner, up, b);
println("reward POMCPOW Planner: $(round(discounted_reward(phist), digits=2))")

# @time mhist = simulate(hr, pomdp, mcts_planner, up, b);

# @time dhist = simulate(hr, pomdp, despot_planner, up, b);
# println("reward DESPOT Planner: $(round(discounted_reward(dhist), digits=2))")

# POMDPs.pdf(prod::ProductDistribution) = pdf(prod.dists[1]) * pdf(prod.dists[2])

# sp = rand(rng, initialstate(pomdp))
# a = rand(rng, actions(pomdp))

# prd = observation(pomdp, a, sp);
# println(fieldnames(typeof(prd)))
# typeof(prd)

# pdf(prd.dists[1], 1)


# o = rand(rng, prd)
# pdf(prd, o)

phist