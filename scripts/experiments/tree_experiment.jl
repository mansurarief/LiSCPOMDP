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
using Statistics
using ProgressBars
using DataStructures
using D3trees

rng = MersenneTwister(1) #control seeding, specific random number generator, sampling initial state and sampling it to random seeding 
#set the random seed generator to start at the same start


function display_results(result_dict)
    for (key, value) in results
        println("Key: $key, Value: $value")
    end
end

function compute_metrics(samples)
    sample_mean = mean(samples)
    n = length(samples)
    sample_se = std(samples) / sqrt(n)
    return (mean = sample_mean, se = sample_se)
end

function experiment(planners, eval_pomdp, n_reps=20, max_steps=30)
    results = OrderedDict() 

    for (planner, planner_name) in planners
        reward_tot_all = []
        reward_disc_all = []
        emission_tot_all = []
        emission_disc_all = []
        domestic_tot_all = []
        imported_tot_all = []
        
        println(" ")
        println("=====Simulating ", typeof(planner), "=====")
        println(" ")
    
        for t = tqdm(1:n_reps)
            reward_tot = 0.0
            reward_disc = 0.0
            emission_tot = 0.0
            emission_disc = 0.0
            vol_tot = 0.0 #mined domestically
            imported_tot = 0.0 #imported/mined internationally
            disc = 1.0 #change to 0.99 
    
            for (s, a, o, r) in stepthrough(eval_pomdp, planner, "s,a,o,r, action_info", max_steps=max_steps)
                tree = info[:tree]
                t = D3Tree(tree)
    
                #compute reward and discounted reward
                reward_tot += r
                reward_disc += r * disc
    
                #compute emissions and discount emeissions
                e = get_action_emission(eval_pomdp, a)
                emission_tot += e
                emission_disc += e * disc
    
                if a.a == "MINE1" || a.a == "MINE2"
                    vol_tot += 1
                elseif a.a == "MINE3" || a.a == "MINE4"
                    imported_tot += 1
                end
    
                disc *= discount(eval_pomdp)
            end 
            push!(reward_tot_all, reward_tot)
            push!(reward_disc_all, reward_disc)
            push!(emission_tot_all, emission_tot)
            push!(emission_disc_all, emission_disc)
            push!(domestic_tot_all, vol_tot)
            push!(imported_tot_all, imported_tot)
        end
    
        results[planner_name] = OrderedDict(
            "Total Reward" => compute_metrics(reward_tot_all),
            "Disc. Reward" => compute_metrics(reward_disc_all),
            "Total Emissions" => compute_metrics(emission_tot_all),
            "Disc. Emissions" => compute_metrics(emission_disc_all),
            "Total Domestic" => compute_metrics(domestic_tot_all),
            "Total Imported" => compute_metrics(imported_tot_all)
        )
    end
    return results  
end

function display_results(results)
    for planner_name in keys(results)
        println("Planner: ", planner_name)

        for metric in keys(results[planner_name])
            println(metric, ": ",results[planner_name][metric])
        end
    end
end

function main()
    """
    EXPERIMENT ONE: COMPARING POLICIES WITH STOCHASTIC PRICING
    """
    # Initialize POMDPs, belief updaters, and initial beliefs
    # Stochastic pricing
    sto_pomdp = initialize_lipomdp(stochastic_price=true) 
    sto_up = LiBeliefUpdater(sto_pomdp)
    sto_b = initialize_belief(sto_up)

    # Deterministic pricing
    det_pomdp = initialize_lipomdp() 
    det_up = LiBeliefUpdater(det_pomdp)
    det_b = initialize_belief(det_up)

    # TESTING stochastic and  deterministic at eval time and plan time
    policy = RandomPolicy(sto_pomdp)
    a = action(policy, sto_b)
    rng = MersenneTwister(0)
    s = rand(initialstate(sto_pomdp))

    sp, o, r = gen(sto_pomdp, s, a, rng)

    random_planner = RandPolicy(sto_pomdp)

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
    pomcpow_planner = solve(solver, sto_pomdp)

    planners = [(pomcpow_planner, "POMCPOW Planner"),  
           (random_planner, "Random Planner"), 
    ]

    # Run the experiment using the deterministic or stochastic POMDP for evaluation
    results = experiment(planners, sto_pomdp) #evaluation 
    display_results(results)
end

main()