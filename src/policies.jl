#=
Modeling the US Path to Lithium Self Sufficiency Using POMDPs
Summer 2023
Yasmine Alonso, Mansur Arief, Anthony Corso, Jef Caers, and Mykel Kochenderfer

File: policies.jl
----------------
This file contains the multiple baseline policies to test our POMCPOW and MCTS-DPW planners against. 
=#

#RANDOM POLICY -- selects a random action to take (from the available ones)
struct RandPolicy <: Policy
    pomdp::LiPOMDP
end

function POMDPs.action(p::RandPolicy, b::LiBelief)
    potential_actions = actions(p.pomdp, b)
    return rand(potential_actions)
end

function POMDPs.action(p::RandPolicy, x::Deterministic{State})
    potential_actions = actions(p.pomdp, x)
    return rand(potential_actions)
end

function POMDPs.updater(policy::RandPolicy)
    return LiBeliefUpdater(policy.pomdp)
end

#GREEDY EFFICIENCY POLICY -- explore all deposits first, then 
@with_kw mutable struct EfficiencyPolicy <: Policy 
    pomdp::LiPOMDP
    need_explore::Vector{Bool}
end

function POMDPs.action(p::EfficiencyPolicy, b::LiBelief)

    # Explore all that needs exploring first, then mine site with highest amount of Li
    for (index, to_explore) in enumerate(p.need_explore)
        if to_explore
            p.need_explore[index] = false
            return Action("EXPLORE$(index)")
        end
    end
    
    # If we have explored all deposits, greedily decide which one to mine that is allowed by the belief.
    scores = zeros(p.pomdp.n_deposits)
    for i in 1:p.pomdp.n_deposits
        if can_explore_here(Action("MINE$(i)"), b)
            score = mean(b.deposit_dists[i])
        else
            score = -Inf
        end
        scores[i] = score
    end
    _, best_mine = findmax(scores)
    
    return Action("MINE$(best_mine)")
end

function POMDPs.updater(policy::EfficiencyPolicy)
    return LiBeliefUpdater(policy.pomdp)
end


#GREEDY EFFICIENCY POLICY CONSIDERING UNCERTAINTY -- same idea as EfficiencyPolicy, but also considers uncertainty
@with_kw mutable struct EfficiencyPolicyWithUncertainty <: Policy 
    pomdp::LiPOMDP
    lambda::Float64  # Penalty factor for uncertainty
    need_explore::Vector{Bool}
end

function POMDPs.action(p::EfficiencyPolicyWithUncertainty, b::LiBelief)
    
    # Explore all that needs exploring first
    for (index, to_explore) in enumerate(p.need_explore)
        if to_explore
            p.need_explore[index] = false
            return (Action("EXPLORE$(index)"))
        end
    end
    
    # If we have explored all deposits, decide which one to mine that is allowed by the belief.
    # We will consider both the expected Lithium and the uncertainty in our decision.    
    scores = zeros(p.pomdp.n_deposits)
    for i in 1:p.pomdp.n_deposits
        if can_explore_here(Action("MINE$(i)"), b)
            score = mean(b.deposit_dists[i])  - p.lambda * std(b.deposit_dists[i])
        else
            score = -Inf
        end
        scores[i] = score
    end
    _, best_mine = findmax(scores)
    return Action("MINE$(best_mine)")
end


function POMDPs.updater(policy::EfficiencyPolicyWithUncertainty)
    return LiBeliefUpdater(policy.pomdp)
end


#EMISSION AWARE POLICY -- explores first, then mines the deposit with the highest expected Lithium per CO2 emission
@with_kw mutable struct EmissionAwarePolicy <: Policy 
    pomdp::LiPOMDP
    need_explore::Vector{Bool}
end

function POMDPs.action(p::EmissionAwarePolicy, b::LiBelief)
    # Explore all that needs exploring first
    for (index, to_explore) in enumerate(p.need_explore)
        if to_explore
            p.need_explore[index] = false
            return Action("EXPLORE$(index)")
        end
    end
    
    # If we have explored all deposits, decide which one to mine.
    # We will prioritize mining the site with the most expected Lithium,
    # but also factor in emissions.

    scores = zeros(p.pomdp.n_deposits)
    for i in 1:p.pomdp.n_deposits
        if can_explore_here(Action("MINE$(i)"), b)
            score = mean(b.deposit_dists[i])/p.pomdp.CO2_emissions[i]
        else
            score = -Inf
        end
        scores[i] = score
    end
    
    _, best_mine = findmax(scores)
    
    return Action("MINE$(best_mine)")
end

function POMDPs.updater(policy::EmissionAwarePolicy)
    return LiBeliefUpdater(policy.pomdp)
end

function POMDPs.updater(policy::POMCPOWPlanner{LiPOMDP, POMCPOW.POWNodeFilter, MaxUCB, typeof(estimate_value), Int64, Float64, POMCPOWSolver{Random.AbstractRNG, POMCPOW.var"#6#12"}})
    return LiBeliefUpdater(policy.problem)
end

function POMDPs.updater(policy::MCTS.DPWPlanner{GenerativeBeliefMDP{LiPOMDP, LiBeliefUpdater, LiBelief{Normal{Float64}}, Action}, 
                                           LiBelief{Normal{Float64}}, 
                                           Action, 
                                           MCTS.SolvedRolloutEstimator{EfficiencyPolicyWithUncertainty, Random.AbstractRNG}, 
                                           RandomActionGenerator{Random.AbstractRNG}, MCTS.var"#18#22", Random.AbstractRNG})
    return LiBeliefUpdater(policy.solved_estimate.policy.pomdp)
end

function POMDPs.updater(policy::MCTS.DPWPlanner{GenerativeBeliefMDP{LiPOMDP, LiBeliefUpdater, ContinueTerminalBehavior{LiPOMDP, LiBeliefUpdater}, LiBelief{Normal{Float64}}, Action}, LiBelief{Normal{Float64}}, Action, MCTS.SolvedRolloutEstimator{EfficiencyPolicyWithUncertainty, Random.AbstractRNG}, RandomActionGenerator{Random.AbstractRNG}, MCTS.var"#18#22", Random.AbstractRNG})
    return LiBeliefUpdater(policy.solved_estimate.policy.pomdp)
 end 

 
