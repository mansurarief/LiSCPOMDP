function run_simulation(pomdp, planner, max_steps)
    reward_tot = 0.0
    reward_disc = 0.0
    emission_tot = 0.0
    emission_disc = 0.0
    vol_tot = 0.0 
    imported_tot = 0.0
    disc = 1.0
    final_info = nothing
    
    for (s, a, o, r, info) in stepthrough(pomdp, planner, "s,a,o,r,action_info", max_steps=max_steps)
        # Check if info is not nothing before trying to access :tree
        if info !== nothing && haskey(info, :tree)
            final_info = info
        end
        
        reward_tot += r
        reward_disc += r * disc

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
    
    return reward_tot, reward_disc, emission_tot, emission_disc, vol_tot, imported_tot, final_info
end

function experiment(planners, eval_pomdp, n_reps=3, max_steps=30)
    results = OrderedDict() 

    for (planner, planner_name) in planners
        println("\n=====Simulating ", planner_name, "=====\n")
        
        reward_tot_all = []
        reward_disc_all = []
        emission_tot_all = []
        emission_disc_all = []
        domestic_tot_all = []
        imported_tot_all = []
        
        # Keep track of the final tree info
        final_tree_info = nothing
        
        for t = tqdm(1:n_reps)
            metrics = run_simulation(eval_pomdp, planner, max_steps)
            
            # Store metrics
            push!(reward_tot_all, metrics[1])
            push!(reward_disc_all, metrics[2])
            push!(emission_tot_all, metrics[3])
            push!(emission_disc_all, metrics[4])
            push!(domestic_tot_all, metrics[5])
            push!(imported_tot_all, metrics[6])
            
            # Update final tree info if available
            # Make sure metrics[7] is not nothing and has :tree key
            if metrics[7] !== nothing && haskey(metrics[7], :tree)
                final_tree_info = metrics[7]
            end
        end
        
        # Visualize the final tree after all iterations
        if final_tree_info !== nothing && haskey(final_tree_info, :tree)
            println("\nGenerating final tree visualization for $planner_name...")
            tree = D3Tree(final_tree_info[:tree], 
                         init_expand=2,
                         title="Final Decision Tree - $(planner_name)")
            inchrome(tree)
            println("Tree visualization complete for $planner_name")
        else
            println("\nNo tree visualization available for $planner_name")
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

function main()
    # Set random seed for reproducibility
    rng = MersenneTwister(1)

    # Initialize POMDPs and beliefs
    sto_pomdp = initialize_lipomdp(stochastic_price=true) 
    sto_up = LiBeliefUpdater(sto_pomdp)
    sto_b = initialize_belief(sto_up)

    # Initialize POMCPOW Solver with tree visualization enabled
    solver = POMCPOW.POMCPOWSolver(
        tree_queries=1000, 
        estimate_value = estimate_value,
        k_observation=4., 
        alpha_observation=0.1, 
        max_depth=15, 
        enable_action_pw=false,
        init_N=10,
        tree_in_info=true  # Enable tree visualization
    )
    
    pomcpow_planner = solve(solver, sto_pomdp)

    # Initialize Random Policy
    random_planner = RandomPolicy(sto_pomdp)

    # Define planners for experiment
    planners = [
        (pomcpow_planner, "POMCPOW Planner"),
        (random_planner, "Random Planner")
    ]

    # Run experiment
    println("Starting experiment...")
    results = experiment(planners, sto_pomdp)
    
    # Display results
    println("\nExperiment Results:")
    display_results(results)
end

# Run the main function
main()