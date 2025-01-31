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
using Plots

rng = MersenneTwister(1)

# Initializing the POMDP, Belief Updater, and initial state, as well as the MDP version of the POMDP for MCTS
pomdp = initialize_lipomdp() 
up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)
policy = RandomPolicy(pomdp)
a = action(policy, b)
rng = MersenneTwister(0)
s = rand(initialstate(pomdp))

actions(pomdp)

println(fieldnames(typeof(initialstate(pomdp).val)))

println(initialstate(pomdp))
a1 = Action("MINE4")
a2 = Action("EXPLORE3")

get_action_type(a1)
get_action_type(a2)
pomdp

sp, o, r = gen(pomdp, s, a, rng)

mdp = GenerativeBeliefMDP(pomdp, up, terminal_behavior=ContinueTerminalBehavior(pomdp, up))

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

despot_solver = DESPOTSolver(bounds=IndependentBounds(-1000.0, 1000.0, check_terminal = true))
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
max_steps=30
results = Dict()# Dict of named tuples

planners = [(pomcpow_planner, "POMCPOW Planner"), 
            (despot_planner, "DESPOT Planner"), 
            (random_planner, "Random Planner"), 
            (strong_planner, "Strong Planner"), 
            (robust_planner, "Robust Planner"), 
            (eco_planner, "Emission Aware Policy"), 
            (mcts_planner, "MCTs Planner")]

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

#(Negative CO2 Emission, Domestic Volume Total, Imported Volume Total)
Pemission_aware_policy = (-107.0, 2.0, 27.0)
Pdespot_policy = (-157.0, 16.0, 1.0)
Prandom_planner = (-133.0, 6.0, 9.0)
Pmcts_planner = (-147.0, 29.0, 0.0)
Probust_planner = (-257.0, 12.0, 17.0)
Ppomcpow_planner = (-103.0, 9.0, 11.0)
Pstrong_planner = (-257.0, 12.0, 17.0)

p = scatter(Pemission_aware_policy, label="Emission Aware Policy", zlabel = "Total Volume Imported")
scatter!(Pdespot_policy, label="DESPOT Policy")
scatter!(Prandom_planner, label="Random Policy")
scatter!(Pmcts_planner, label="MCTs Planner")
scatter!(Probust_planner, label = "Robust Planner")
scatter!(Ppomcpow_planner, label="POMCPOW Planner")
scatter!(Pstrong_planner, label="Strong Planner")

xlabel!("Negative Total Emissions")
ylabel!("Total Domestic Volume")
zlabel!("Total Volume Imported")

#Could we have a plane to show where we want to be. Like, in 2D graph, you want to be on the lienar line


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

#Change Mine Parameters and rerun simulations

#Example use of mine_params with 5 mines
#mine_params(pomdp, n_deposits, deposits vector, objective weights vector, emissions vector)
println("LiPOMDP before mine_params() update: $pomdp")
mine_params(pomdp, 5, [16.0, 60.0, 60.0, 50.0, 100.0], [.2, .2, .2, .2, .2], [5, 7, 2, 5, 10])
s = rand(initialstate(pomdp)) #Must reinitinalize state to update have_mined vector{Bool}
println(pomdp.init_state)

println("LiPOMDP after mine_params() update - updated to 5 mines, modifed deposit volumes and CO2_emissions: $pomdp")

#Example with 10 mines. 
mine_params(pomdp, 
            10, 
            [16.0, 60.0, 60.0, 50.0, 100.0, 40.0, 21.0, 33.0, 8.0, 90.0], 
            [.2, .2, .2, .2, .2], 
            [5, 7, 2, 5, 10, 9, 2, 3, 4, 5])

pomdp

n_reps=20
max_steps=30
results = Dict() # Dict of named tuples

planners = [(pomcpow_planner, "POMCPOW Planner"), 
            (despot_planner, "DESPOT Planner"), 
            (random_planner, "Random Planner"), 
            (strong_planner, "Strong Planner"), 
            (robust_planner, "Robust Planner"), 
            (eco_planner, "Emission Aware Policy"), 
            (mcts_planner, "MCTs Planner")]
            
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