#=
Modeling the US Path to Lithium Self Sufficiency Using POMDPs
Summer 2023
Yasmine Alonso, Mansur Arief, Anthony Corso, Jef Caers, and Mykel Kochenderfer

File: poster_plots
----------------
This file contains the code to generate the plots that I used for my poster
=#

# All of the imports
using POMDPs
using POMDPModelTools
using POMDPPolicies
using QuickPOMDPs
using Parameters
using Random
using DiscreteValueIteration
using Distributions
using Plots
using POMCPOW
using LinearAlgebra
using Statistics
using QMDP
using D3Trees
using Plots
using Measures

#using ConjugatePriors: posterior

include("LiPOMDP.jl")
include("main.jl")
include("utils.jl")



function run_sims(up::LiBeliefUpdater, b0::LiBelief, s0::State, action_sequence::Vector{Action}, rng::AbstractRNG)
    b = b0
    s = s0
    belief_history = []
    state_history = []
    push!(belief_history, b)
    push!(state_history, s)
    
    for a in action_sequence
        sp, o, r = gen(P,s,a, rng)  # from anthony
        o = Float64.(o)
        # Maybe here instead of passing in the vector i need to select the right entry and pass that into update
        new_belief = update(up, b, a, o)
        b = new_belief
        s = sp
        # Changed this from s -> sp
        if isterminal(P, sp)
            break
        end
        push!(belief_history, new_belief)
        push!(state_history, sp)
    end 
    return belief_history, state_history
end

P = LiPOMDP()
init_belief = LiBelief([Normal(9, 0.2), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], 0.0, 0.0, [false, false, false, false])
init_state = P.init_state

# Beliefs at all four sites over time graph
function do_plots(rng::AbstractRNG)
    P = LiPOMDP()
    up = LiBeliefUpdater(P)
    init_belief = LiBelief([Normal(10, 0.2), Normal(5, 1), Normal(3, 0.2), Normal(9, 4)], 0.0, 0.0, [false, false, false, false])
    init_state = State([9.5, 4.5, 3, 9], 0.0, 0.0, [false, false, false, false])


    # Deposit 1 stuff
    dep_1_actions = [EXPLORE1, EXPLORE2, EXPLORE3, EXPLORE4, MINE1, MINE2, MINE3, MINE4]
    action_sequence = [rand(rng, dep_1_actions) for x in 1:20]
    println(action_sequence)
    # Change MersenneTwister(1) to rng
    belief_history, state_history = run_sims(up, init_belief, init_state, action_sequence, MersenneTwister(7))
    times = [b.t for b in belief_history]

    # Deposit 1 stuff
    μ1s = [mean(b.deposit_dists[1]) for b in belief_history]
    σ1s = [std(b.deposit_dists[1]) for b in belief_history]
    true_v1s = [s.deposits[1] for s in state_history] # actual amount of Li in deposit 3

    # Deposit 2 stuff
    μ2s = [mean(b.deposit_dists[2]) for b in belief_history]
    σ2s = [std(b.deposit_dists[2]) for b in belief_history]
    true_v2s = [s.deposits[2] for s in state_history] # actual amount of Li in deposit 2

    # Deposit 3 stuff
    μ3s = [mean(b.deposit_dists[3]) for b in belief_history]
    σ3s = [std(b.deposit_dists[3]) for b in belief_history]
    true_v3s = [s.deposits[3] for s in state_history] # actual amount of Li in deposit 3

    # Deposit 4 stuff
    μ4s = [mean(b.deposit_dists[4]) for b in belief_history]
    σ4s = [std(b.deposit_dists[4]) for b in belief_history]
    true_v4s = [s.deposits[4] for s in state_history] # actual amount of Li in deposit 4

    # Set default size
    default(size=(2600,1800))
    # Plottting everything, start with Deposit 1: Plum
    p = plot(times, μ1s, grid=true, ribbon=σ1s, fillalpha=0.5, title="Deposit Beliefs vs. time", xlabel="Time (t)", ylabel="Volume of Lithium", label="Belief μ1", linecolor="#620059", fillcolor="#734675", linestyle=:dash, linewidth=5, tickfontsize=40, guidefontsize=50, titlefontsize=60, legendfontsize=40, bottommargin=10mm)
    plot!(p, times, true_v1s, label="Actual V₁", color="#350D36", linewidth=5)

    # Deposit 2: Spirited
    plot!(p, times, μ2s, grid=true, ribbon=σ2s, fillalpha=0.5, label="Belief μ2", linecolor="#E04F39", fillcolor="#F4795B", linestyle=:dash, linewidth=5)
    plot!(p, times, true_v2s, label="Actual V₂", color="#C74632", linewidth=5)

    # Deposit 3: Palo Verde
    plot!(p, times, μ3s, grid=true, ribbon=σ3s, fillalpha=0.5, label="Belief μ3", linecolor="#279989", fillcolor="#59B3A9", linestyle=:dash, linewidth=5)
    plot!(p, times, true_v3s, label="Actual V₃", color="#017E7C", linewidth=5)

    # Deposit 4: Poppy
    plot!(p, times, μ4s, grid=true, ribbon=σ4s, fillalpha=0.5, label="Belief μ4", linecolor="#E98300", fillcolor="#F9A44A", linestyle=:dash, linewidth=5)
    plot!(p, times, true_v4s, label="Actual V₄", color="#D1660F", linewidth=5)


    plot!(p, leftmargin=22mm, bottommargin=10mm)
end
do_plots(MersenneTwister(2))
savefig("/Users/yasminealonso/Desktop/poster_images/beliefs_final.png")

belief_history, state_history = run_sims(up, init_belief, init_state, action_sequence, rng)
times = [b.t for b in belief_history]  # Goes up at top like an iteration counter
d1_normals = [b.deposit_dists[1] for b in belief_history]


@gif for i in 1:length(times)
    normal = d1_normals[i]
    if i < 7
        a = action_sequence[i]
    else
        a = "DONE"
    end    
    
    plot(5:0.01:10, (x) -> pdf(normal, x), title = "Iter. $i, a: $a", ylim = (0, 20), xlim = (5, 10), xlabel = "V₁ belief", label= "V₁ belief", legend=:topright, color=:purple)
end fps = 2


# Plotting the normals for center poster image
# Plot 1: Greenbushes Australia mine -- high volume, high certainty
d = Normal(10, 0.3)
x = range(-3, 13, length=1000)
y = pdf.(d, x)

# Plot the distribution
p = plot(x, y, label="", xlabel="Volume of Li", ylabel="Probability Density", linewidth=5, linecolor="#620059", 
    size=(2000, 1400), dpi=300, tickfontsize=24, guidefontsize=30)

# Adjust the margin
plot!(p, leftmargin=22mm, bottommargin=10mm)

# Customize the x-axis labels
xticks!([1, 5, 10], ["Low", "Medium", "High"])
savefig(p, "/Users/yasminealonso/Desktop/poster_images/plot1.png")


# Plot 2: Pilgangoora Australia mine -- medium volume, slightly less certainty than Greenbushes
d = Normal(8, 1.5)
x = range(-3, 13, length=1000)
y = pdf.(d, x)


# Plot the distribution
p = plot(x, y, label="", xlabel="Volume of Li", ylabel="Probability Density", linewidth=5, linecolor="#E04F39", 
    size=(2000, 1400), dpi=300, tickfontsize=24, guidefontsize=30)

# Adjust the margin
plot!(p, leftmargin=22mm, bottommargin=10mm)

# Customize the x-axis labels
xticks!([1, 5, 10], ["Low", "Medium", "High"])
savefig("/Users/yasminealonso/Desktop/poster_images/plot2.png")




# Plot 3: NV mine 1 -- low volume, decently high certainty
d = Normal(2.3, 0.5)
x = range(-3, 13, length=1000)
y = pdf.(d, x)

# Plot the distribution
p = plot(x, y, label="", xlabel="Volume of Li", ylabel="Probability Density", linewidth=5, linecolor="#279989", 
    size=(2000, 1400), dpi=300, tickfontsize=24, guidefontsize=30)

# Adjust the margin
plot!(p, leftmargin=22mm, bottommargin=10mm)


# Customize the x-axis labels
xticks!([1, 5, 10], ["Low", "Medium", "High"])
savefig("/Users/yasminealonso/Desktop/poster_images/plot3.png")




# Plot 4: NV mine 2 -- medium volume, low certainty
d = Normal(5, 3)
x = range(-3, 13, length=1000)
y = pdf.(d, x)

# Plot the distribution
p = plot(x, y, label="", xlabel="Volume of Li", ylabel="Probability Density", linewidth=5, linecolor="#E98300", 
    size=(2000, 1400), dpi=300, tickfontsize=24, guidefontsize=30)

# Adjust the margin
plot!(p, leftmargin=22mm, bottommargin=10mm)

# Customize the x-axis labels
xticks!([1, 5, 10], ["Low", "Medium", "High"])
savefig("/Users/yasminealonso/Desktop/poster_images/plot4.png")


init_belief_dists_center()