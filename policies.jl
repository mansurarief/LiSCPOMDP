#=
Modeling the US Path to Lithium Self Sufficiency Using POMDPs
Summer 2023
Yasmine Alonso, Mansur Arief, Anthony Corso, Jef Caers, and Mykel Kochenderfer

File: policies.jl
----------------
This file contains the multiple baseline policies to test our POMCPOW and MCTS-DPW planners against. 
=#

#RANDOM POLICY -- selects a random action to take (from the available ones)
struct RandomPolicy <: Policy
    pomdp::LiPOMDP
end

function POMDPs.action(p::RandomPolicy, b::LiBelief)
    potential_actions = actions(p.pomdp, b)
    #println("random actions list: $potential_actions for $b")
    return rand(potential_actions)
end

function POMDPs.updater(policy::RandomPolicy)
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
            return eval(Meta.parse("EXPLORE$(index)"))
        end
    end
    
    # If we have explored all deposits, greedily decide which one to mine that is allowed by the belief.
    scores = zeros(p.pomdp.n_deposits)
    for i in 1:p.pomdp.n_deposits
        if can_explore_here(eval(Meta.parse("MINE$(i)")), b)
            score = mean(b.deposit_dists[i])
        else
            score = -Inf
        end
        scores[i] = score
    end
    _, best_mine = findmax(scores)
    
    return eval(Meta.parse("MINE$(best_mine)"))
end


#GREEDY EFFICIENCY POLICY CONSIDERING UNCERTAINTY -- same idea as EfficiencyPolicy, but also considers uncertainty
@with_kw mutable struct EfficiencyPolicyWithUncertainty <: Policy 
    pomdp::LiPOMDP
    lambda::Float64  # Penalty factor for uncertainty
    need_explore::Vector{Bool}
end

function POMDPs.action(p::EfficiencyPolicyWithUncertainty, b::LiBelief)

    #println("EfficiencyPolicyWithUncertainty actions list: $(actions(p.pomdp, b)) for $b")

    # Explore all that needs exploring first
    for (index, to_explore) in enumerate(p.need_explore)
        if to_explore
            p.need_explore[index] = false
            return eval(Meta.parse("EXPLORE$(index)"))
        end
    end
    
    # If we have explored all deposits, decide which one to mine that is allowed by the belief.
    # We will consider both the expected Lithium and the uncertainty in our decision.    
    scores = zeros(p.pomdp.n_deposits)
    for i in 1:p.pomdp.n_deposits
        if can_explore_here(eval(Meta.parse("MINE$(i)")), b)
            score = mean(b.deposit_dists[i])  - p.lambda * std(b.deposit_dists[i])
        else
            score = -Inf
        end
        scores[i] = score
    end
    _, best_mine = findmax(scores)
    return eval(Meta.parse("MINE$(best_mine)"))
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
            return eval(Meta.parse("EXPLORE$(index)"))
        end
    end
    
    # If we have explored all deposits, decide which one to mine.
    # We will prioritize mining the site with the most expected Lithium,
    # but also factor in emissions.

    scores = zeros(p.pomdp.n_deposits)
    for i in 1:p.pomdp.n_deposits
        if can_explore_here(eval(Meta.parse("MINE$(i)")), b)
            score = mean(b.deposit_dists[i])/p.pomdp.CO2_emissions[i]
        else
            score = -Inf
        end
        scores[i] = score
    end
    
    _, best_mine = findmax(scores)
    
    return eval(Meta.parse("MINE$(best_mine)"))
end