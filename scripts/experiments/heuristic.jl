using Random
using POMDPs 
using POMDPTools
using LiPOMDPs
using Distributions
using Parameters
using Plots
using Statistics
using ProgressBars

rng = MersenneTwister(1)

function compute_metrics(samples)
    sample_mean = mean(samples)
    n = length(samples)
    sample_se = std(samples) / sqrt(n)
    return (mean = sample_mean, se = sample_se)
end

function experiment(planner, eval_pomdp, n_reps=100, max_steps=30)
    reward_tot_all = []
    reward_disc_all = []
    emission_tot_all = []
    emission_disc_all = []
    domestic_tot_all = []
    imported_tot_all = []
    
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

            #compute emissions and discount emissions
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

    results = Dict(
        "Total Reward" => compute_metrics(reward_tot_all),
        "Disc. Reward" => compute_metrics(reward_disc_all),
        "Total Emissions" => compute_metrics(emission_tot_all),
        "Disc. Emissions" => compute_metrics(emission_disc_all),
        "Total Domestic" => compute_metrics(domestic_tot_all),
        "Total Imported" => compute_metrics(imported_tot_all)
    )
    
    return results  
end

function display_results(results)
    for metric in keys(results)
        println(metric, ": ", results[metric])
    end
end

function plot_pareto(results)
    alphas = LinRange(0, 1, 50)  # Linearly spaced reward coefficients

    # Extract data
    xs = [results[alpha]["emissions"][1] for alpha in alphas]
    ys = [results[alpha]["volume"][1] for alpha in alphas]

    # Get error bars
    xerror = [results[alpha]["emissions"][2] for alpha in alphas]
    yerror = [results[alpha]["volume"][2] for alpha in alphas]

    # Sort values for clean plotting
    sorted_indices = sortperm(xs)
    xs = xs[sorted_indices]
    ys = ys[sorted_indices]
    xerror = xerror[sorted_indices]
    yerror = yerror[sorted_indices]

    # Create the plot
    p = plot(
        xs,
        ys,
        xerr = xerror,
        yerr = yerror,
        seriestype = :scatter,
        xlabel = "Total Emissions",
        ylabel = "Total Volume",
        title = "Pareto Tradeoff: Import-Only Policy",
        legend = :topright,
        grid = true,
        gridalpha = 0.3,
        linewidth = 2.5,
        size = (850, 500),
        background_color = :white,
        foreground_color = :black,
        label = "Emissions vs Volume ± Error"
    )

    # Connect the points with a line
    plot!(xs, ys,
        seriestype = :path,
        label = "Pareto curve",
        linecolor = :blue,
        alpha = 0.9,
        markershape = :circle,
        markercolor = :red,
        markersize = 7,
        markerstrokewidth = 2
    )

    # Save figure
    savefig(p, "import_only_pareto.png")
    return p
end

function compute_tradeoff(alpha=1, stochastic_price=false, train_same=true, max_steps=30)
    # Initialize POMDP
    pomdp = initialize_lipomdp(alpha=alpha, stochastic_price=stochastic_price, compute_tradeoff=true)
    
    # Create the import-only planner
    import_only_planner = ImportOnlyPolicy(pomdp, max_steps)
    
    # Run experiment directly with the import-only planner
    results = experiment(import_only_planner, pomdp)
    
    return results
end

function main()
    max_steps = 30  # Maximum steps to simulate
    
    # Define alpha values for the Pareto curve
    alpha_values = collect(LinRange(0, 1, 50))
    
    # Store results
    results_import_only = Dict()
    
    println("\nGenerating Pareto curve for Import-Only policy...")
    for alpha in tqdm(alpha_values)
        # Compute results for this alpha value
        results = compute_tradeoff(alpha, false, true, max_steps)
        
        # Store the metrics we need for the Pareto curve
        results_import_only[alpha] = Dict(
            "emissions" => results["Total Emissions"],
            "volume" => (
                results["Total Domestic"].mean + results["Total Imported"].mean,
                sqrt(results["Total Domestic"].se^2 + results["Total Imported"].se^2)
            )
        )
    end
    
    # Create the Pareto plot
    p = plot_pareto(results_import_only)
    
    println("\nPareto curve generated and saved as 'import_only_pareto.png'")
    
    # Display summary statistics at key points
    println("\nSummary at key alpha values:")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]
        idx = findmin(abs.(alpha_values .- alpha))[2]
        actual_alpha = alpha_values[idx]
        println("\nAlpha = $actual_alpha:")
        println("  Emissions: $(results_import_only[actual_alpha]["emissions"].mean) ± $(results_import_only[actual_alpha]["emissions"].se)")
        println("  Volume: $(results_import_only[actual_alpha]["volume"][1]) ± $(results_import_only[actual_alpha]["volume"][2])")
    end
end

main()