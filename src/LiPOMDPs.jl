module LiPOMDPs

using Random
using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools
using POMDPModelTools
using POMDPPolicies
using Serialization
using Distributions
using ParticleFilters
using Parameters
using Plots
using LinearAlgebra
using Statistics
using D3Trees
using MCTS
using Plots.PlotMeasures
using JLD2
using JuMP
using GLPK
 
export 
    LiPOMDP, 
    LiBelief, 
    LiBeliefUpdater,
    State, 
    Action,
    initialize_lipomdp
include("model.jl")

export
    #Functions
    evaluate_policies,
    evaluate_policy,
    print_policy_results,
    replicate_simulation,
    simulate_policy,
    get_action_type,
    str_to_action,
    get_site_number,
    splice,
    save_policy,
    extract_belief
include("utils.jl")

export 
    plot_results,
    _plot_results,
    get_rewards,
    _get_rewards
include("viz.jl")

export
    #types
    RandPolicy,
    EfficiencyPolicy,
    EfficiencyPolicyWithUncertainty,
    EmissionAwarePolicy,
    AusDomPolicy,
    HeuristicPolicy
include("policies.jl")

export 
    #Functions
    create_lithium_mining_model,
    print_optimization_results,
    format_optimization_results,
    solve_lithium_mining_model
include("MILP_no_exploration.jl")

export 
    OptimizationPolicy,
    create_optimization_policy,
    optimization_to_pomdp_action
include("policies_MILP.jl")

export 
    #Functions
    compute_r1, 
    compute_r2,
    compute_r3,
    compute_r4,
    compute_r5
include("pomdp.jl")

end #module