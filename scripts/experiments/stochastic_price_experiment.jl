using Random
using POMDPs 
using POMDPTools
using LiPOMDPs
using MCTS
using DiscreteValueIteration
using POMCPOW 
using Distributions
using Parameters
using ARDESPOT
using Plots

rng = MersenneTwister(1)

"""
EXPERIMENT ONE: COMPARING POLICIES WITH STOCHASTIC PRICING
"""
# Initializing the POMDP, Belief Updater, and initial state, as well as the MDP version of the POMDP for MCTS
pomdp = initialize_lipomdp(stochastic_price=true) 
up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)

# Initialize random policy
policy = RandomPolicy(pomdp)
a = action(policy, b)
rng = MersenneTwister(0)
s = rand(initialstate(pomdp))

sp, o, r = gen(pomdp, s, a, rng)

random_planner = RandPolicy(pomdp)

# POMCPOW Solver
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
max_steps=30
results = Dict()# Dict of named tuples

planners = [(pomcpow_planner, "POMCPOW Planner"),  
            (random_planner, "Random Planner"), 
            ]

for (planner, name) in planners
    reward_tot = 0.0
    reward_disc = 0.0
    emission_tot = 0.0
    emission_disc = 0.0
    vol_tot = 0.0 #mined domestically
    imported_tot = 0.0 #imported/mined internationally
    disc = 1.0

    println(" ")
    println("=====Simulating ", typeof(planner), "=====")
    println(" ")
    for (s, a, o, r) in stepthrough(pomdp, planner, "s,a,o,r", max_steps=max_steps)
        println("in state $s")
        println("took action $a")
        println("received observation $o and reward $r")

        #compute reward and discounted reward
        reward_tot += r
        reward_disc += r * disc

        #compute emissions and discount emeissions
        e = get_action_emission(pomdp, a)
        emission_tot += e
        emission_disc += e * disc

        if a.a == "MINE1" || a.a == "MINE2"
            vol_tot += 1
        elseif a.a == "MINE3" || a.a == "MINE4"
            imported_tot += 1
        end

        disc *= discount(pomdp)
    end
    named_tuple = (reward_tot = reward_tot, 
                   reward_disc = reward_disc,
                   emission_tot = emission_tot, 
                   emission_disc = emission_disc, 
                   vol_tot = vol_tot, 
                   imported_tot = imported_tot)

    results[name] = named_tuple
end

for (key, value) in results
    println("Key: $key, Value: $value")
end


"""
EXPERIMENT TWO: DETERMINISTIC PRICING
"""
# Initializing the POMDP, Belief Updater, and initial state, as well as the MDP version of the POMDP for MCTS
pomdp = initialize_lipomdp() 
up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)

# Initialize random policy
policy = RandomPolicy(pomdp)
a = action(policy, b)
rng = MersenneTwister(0)
s = rand(initialstate(pomdp))

sp, o, r = gen(pomdp, s, a, rng)

random_planner = RandPolicy(pomdp)

# POMCPOW Solver
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
max_steps=30
results = Dict()# Dict of named tuples

planners = [(pomcpow_planner, "POMCPOW Planner"),  
            (random_planner, "Random Planner"), 
            ]

for (planner, name) in planners
    reward_tot = 0.0
    reward_disc = 0.0
    emission_tot = 0.0
    emission_disc = 0.0
    vol_tot = 0.0 #mined domestically
    imported_tot = 0.0 #imported/mined internationally
    disc = 1.0

    println(" ")
    println("=====Simulating ", typeof(planner), "=====")
    println(" ")
    for (s, a, o, r) in stepthrough(pomdp, planner, "s,a,o,r", max_steps=max_steps)
        println("in state $s")
        println("took action $a")
        println("received observation $o and reward $r")

        #compute reward and discounted reward
        reward_tot += r
        reward_disc += r * disc

        #compute emissions and discount emeissions
        e = get_action_emission(pomdp, a)
        emission_tot += e
        emission_disc += e * disc

        if a.a == "MINE1" || a.a == "MINE2"
            vol_tot += 1
        elseif a.a == "MINE3" || a.a == "MINE4"
            imported_tot += 1
        end

        disc *= discount(pomdp)
    end
    named_tuple = (reward_tot = reward_tot, 
                   reward_disc = reward_disc,
                   emission_tot = emission_tot, 
                   emission_disc = emission_disc, 
                   vol_tot = vol_tot, 
                   imported_tot = imported_tot)

    results[name] = named_tuple
end

for (key, value) in results
    println("Key: $key, Value: $value")
end