using POMDPs, LiPOMDPs, Random, JuMP, GLPK
using Plots.PlotMeasures
using Plots
using POMDPTools

# Usage example
pomdp = LiPOMDP(
    p=10.0, ρ=0.6,
    T=30, w=[9.0, 3.2, 0.1, 1.2],
    a=[-20, -20, -20, -30],
    mine_rate=[300, 300, 100, 100],
    v0 = [4200., 2400., 5500., 2500.]
)

up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)
rng = MersenneTwister(2024)
s = rand(rng, b)
s0 = State(
    v=[mean(bi) for bi in b.v_dists],
    t=1.0,
    m=[0 for i in 1:pomdp.n]
)

# Create and solve the optimization model
T = 1:25
I = 1:4
model = create_lithium_mining_model(pomdp, s0, T, I)
optimize!(model)


# # Print results
print_optimization_results(model, pomdp, T, I)

# # Plot the results




results = format_optimization_results(model, pomdp, T, I)
p = _plot_results(pomdp, results)
pall = plot(p.action, p.econ, p.other, layout=(3, 1), size=(1100, 800), margin=5mm)


policy = create_optimization_policy(model, pomdp, T, I)

# # Use policy in POMDP simulation
up = updater(policy)
b0 = initialize_belief(up)
hr = HistoryRecorder(max_steps=length(T))
hist = simulate(hr, pomdp, policy, up, b0, s)

#get rewards
df = _get_rewards(pomdp, hist)
p0 = _plot_results(pomdp, df)
pall0 = plot(p0.action, p0.econ, p0.other, layout=(3, 1), size=(1100, 800), margin=5mm)

#TODO: fix converter of optimization action to POMDP policy
#TODO: eva