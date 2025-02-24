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

rng = MersenneTwister(1)


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
    results = Dict() 

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
            disc = 1.0
    
            for (s, a, o, r) in stepthrough(eval_pomdp, planner, "s,a,o,r", max_steps=max_steps)
    
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
    
        results[planner_name] = Dict(
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

function display_pomcpow_tree(planner, state)
    a, info = action_info(planner, state)
    D3Tree(info[:tree])
end

function plot_pareto(results)
    xs = collect(LinRange(0, 1, 6))
    ys = [results[x]["volume"][1] for x in xs]
    yerror = [results[x]["volume"][2] for x in xs]
    p = plot(xs,ys,grid=false,yerror=yerror, label="y = Total Volume", xlabel="Emissions Reward Coefficient", xticks=0:0.2:1)
    savefig(p, "emissions_volume_tradeoff.png")
end

function compute_tradeoff(alpha=1, stochastic_price=false, train_same=true)
    train_pomdp = initialize_lipomdp(alpha=alpha, stochastic_price=stochastic_price)
    train_up = LiBeliefUpdater(train_pomdp) 
    train_b = initialize_belief(train_up)

    if train_same # Same POMDP at train and eval time
        eval_pomdp = train_pomdp
    else
        eval_pomdp = initialize_lipomdp(alpha=alpha, stochastic_price=!stochastic_price)
    end
    
    # Test Random and 
    policy = RandomPolicy(train_pomdp)
    a = action(policy, train_b)
    rng = MersenneTwister(0)
    s = rand(initialstate(train_pomdp))

    sp, o, r = gen(train_pomdp, s, a, rng)

    random_planner = RandPolicy(train_pomdp)

    # POMCPOW Solver
    solver = POMCPOW.POMCPOWSolver(
        tree_queries=1000, 
        estimate_value = estimate_value, #RolloutEstimator(RandomPolicy(pomdp)), #estimate_value,
        k_observation=4., 
        alpha_observation=0.1, 
        max_depth=15, 
        enable_action_pw=false,
        init_N=10,
        tree_in_info=true  
    ) # Estimate value should fix the previous problem with action functions
    pomcpow_planner = solve(solver, train_pomdp)

    planners = [(pomcpow_planner, "POMCPOW Planner"),  
           (random_planner, "Random Planner"), 
    ]

    results = experiment(planners, eval_pomdp)
    display_results(results)
    return results
end

function main()
    alpha_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results_rand = Dict()
    results_pomcpow = Dict()
    emissions = []
    for alpha in alpha_values
        alpha_results = compute_tradeoff(alpha, false, true)
        results_pomcpow[alpha] = Dict("emissions" => alpha_results["POMCPOW Planner"]["Total Emissions"], 
                              "volume"    => (alpha_results["POMCPOW Planner"]["Total Domestic"][1] + 
                                             alpha_results["POMCPOW Planner"]["Total Imported"][1], 
                                             alpha_results["POMCPOW Planner"]["Total Domestic"][2] + 
                                             alpha_results["POMCPOW Planner"]["Total Imported"][2]))

    end
    print(results_pomcpow)
    plot_pareto(results_pomcpow)
end

main()