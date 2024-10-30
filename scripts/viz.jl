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

up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)
# random_policy = RandomPolicy(pomdp)
random_planner = RandPolicy(pomdp)
robust_planner = EfficiencyPolicyWithUncertainty(pomdp, 1., [true, true, true, true])
solver = POMCPOW.POMCPOWSolver(
     tree_queries=20000, 
    #  estimate_value = RolloutEstimator(random_planner), #estimate_value,
    #  k_observation=4., 
     alpha_observation=0.1,
     max_depth=100, 
     enable_action_pw=false  
 ) 
pomcpow_planner = solve(solver, pomdp)

n_reps=1
max_steps=30

# hr = HistoryRecorder(max_steps=max_steps)

# @time random_hist = simulate(hr, pomdp, random_planner, up, b);
# println("reward $(typeof(random_planner)): $(round(discounted_reward(random_hist), digits=2))")


# @time robust_hist = simulate(hr, pomdp, robust_planner, up, b);
# println("reward $(typeof(robust_planner)): $(round(discounted_reward(robust_hist), digits=2))")

# @time phist = simulate(hr, pomdp, pomcpow_planner, up, b);
# println("reward POMCPOW Planner: $(round(discounted_reward(phist), digits=2))")

# function print_actions(hist)
#     for step in hist
#         println("in state $(step.s)")
#         println("took action $(step.a)")
#         println("received reward $(step.r)")
#     end
# end

# random_hist[1]
# print_actions(random_hist)

function run_sims(policy, pomdp, nreps, up, rng)
    r1s = []
    r2s = []
    r3s = []
    r4s = []
    r5s = []
    rs = []
    explore_actions = []
    explore_times = []
    invest_actions = []
    invest_times = []
    decommision_actions = []
    decommision_times = []
    nothing_actions = []
    nothing_times = []
    operation_states = []
    bs = []
    
    for i in 1:nreps
        r1 = []
        r2 = []
        r3 = []
        r4 = []
        r5 = []
        r = []

        explore_action = []
        explore_time = []
        invest_action = []
        invest_time = []
        decommision_action = []
        decommision_time = []
        nothing_action = []
        nothing_time = []

        operation_state = Array{Bool}(undef, pomdp.n, pomdp.T)
        
        b = initialize_belief(up)        
        s = rand(rng, initialstate(pomdp))
        t = 0

        while !isterminal(pomdp, s)
            println("t: $t, i: $i")
            t += 1
            a = action(policy, b)
            sp, o, r_ = gen(pomdp, s, a, rng)
            r1_, r2_, r3_, r4_, r5_, r_ = reward(pomdp, s, a, o)
            bp = update(up, b, a, o)

            println("r1: $r1_, r2: $r2_, r3: $r3_, r4: $r4_, r5: $r5_, r: $r_")

            push!(r1, r1_)
            push!(r2, r2_)
            push!(r3, r3_)
            push!(r4, r4_)
            push!(r5, r5_)
            push!(r, r_)
            push!(bs, b)          
            # println("Done with reward")  

            if get_action_type(a) == "EXPLORE"
                push!(explore_action, Int(a)-4)
                push!(explore_time, s.t)
            elseif get_action_type(a) == "MINE"
                push!(invest_action, Int(a))
                push!(invest_time, s.t)
            elseif get_action_type(a) == "RESTORE"
                push!(decommision_action, Int(a)-8)
                push!(decommision_time, s.t)
            else
                push!(nothing_action, Int(a))
                push!(nothing_time, s.t)
            end

            operation_state[:, t] = s.m

            s = deepcopy(sp)
            b = deepcopy(bp)
        end

        push!(r1s, r1)
        push!(r2s, r2)
        push!(r3s, r3)
        push!(r4s, r4)
        push!(r5s, r5)
        push!(rs, r)
        push!(operation_states, operation_state)
        push!(explore_actions, explore_action)
        push!(explore_times, explore_time)
        push!(invest_actions, invest_action)
        push!(invest_times, invest_time)
        push!(decommision_actions, decommision_action)
        push!(decommision_times, decommision_time)
        push!(nothing_actions, nothing_action)
        push!(nothing_times, nothing_time)

    end

    return Dict(
        "r1s" => r1s,
        "r2s" => r2s,
        "r3s" => r3s,
        "r4s" => r4s,
        "r5s" => r5s,
        "rs" => rs,
        "explore_actions" => explore_actions,
        "explore_times" => explore_times,
        "invest_actions" => invest_actions,
        "invest_times" => invest_times,
        "decommision_actions" => decommision_actions,
        "decommision_times" => decommision_times,
        "operation_states" => operation_states,
        "nothing_actions" => nothing_actions,
        "nothing_times" => nothing_times,
        "bs" => bs
    )
end

sim_results = run_sims(pomcpow_planner, pomdp, n_reps, up, rng)
df = deepcopy(sim_results)


using Plots
# function plot_results(pomdp::LiPOMDP, df::NamedTuple;ylims=(-200, 200))

T = pomdp.T
    #plot 1: actions vs time
p0 = scatter(df["explore_times"], df["explore_actions"], label="EXPLORE", markersize=10, xticks=1:T);
scatter!(df["invest_times"], df["invest_actions"], label="INVEST", markersize=10);
scatter!(df["decommision_times"], df["decommision_actions"], label="DECOMMISSION/REHAB", markersize=10);
scatter!(df["nothing_times"], df["nothing_actions"], label="NO ACTION", markersize=10)

#     scatter!(df.t_invest, df.a_invest, label="INVEST", markersize=10);
#     scatter!(df.t_decommision, df.a_decommision, label="DECOMMISSION/REHAB", markersize=10);
#     scatter!(
#         df.t_mine, df.a_mine, 
#         label="MINE", markersize=10, 
#         xticks=0:1:T, 
#         yticks=(1:4, ["1 (SilverPeak, USA)", "2 (ThackerPass, USA)", "3 (Greenbushes, AUS)", "4 (Pilgangoora, AUS)"]),
#         ylims=(0.5, 4.5), 
#         ylabel="Deposit Site", xlabel="Time", 
#         title="Actions vs Time", 
#         legend=:outerbottomright);
#     vline!([pomdp.t_goal], label="Time Delay Goal", color=:red, linestyle=:dash);        

#     #set xticks to be integers
#     p1 = bar(df.r1, label="r1", xlabel="Time", ylabel="\$ Value (in Millions)", title="Domestic Mining (Penalty)", legend=false, xticks=0:5:T, ylims=ylims);
#     p5 = bar(df.r5, label="r5", xlabel="Time", ylabel="\$ Value (in Millions)", title="Cash Flow", legend=false, ylims=ylims, xticks=0:5:T);
#     p4 = bar(df.r4, label="r4", xlabel="Time", ylabel="\$ Value (in Millions)", title="Unmet Demand (Penalty)", legend=false, ylims=ylims=ylims, xticks=0:5:T);
#     p2 = bar(df.r2.domestic+df.r2.imported, label="r2", xlabel="Time", ylabel="Thousand Metric Tons", title="LCE Volume Mined", legend=false, xticks=0:5:T);
#     p3 = bar(df.r3, label="r3", xlabel="Time", ylabel="Units", title="CO2 Emission", legend=false, ylims=(-30, 0), xticks=0:5:T);

#     prow1 = plot(p5, p1, p4, layout=(1, 3), margin=3mm);  
#     prow2 = plot(p2, p3, layout=(1, 2));  
#     return (action=p0, econ=prow1, other=prow2)
# end

# policy_path = "pomcpow.jld2"
# save(policy_path, "policy", pomcpow_planner)