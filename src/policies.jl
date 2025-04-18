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
    scores = zeros(p.pomdp.n)
    for i in 1:p.pomdp.n
        if can_explore_here(eval(Meta.parse("MINE$(i)")), b)
            score = mean(b.v_dists[i]) - p.lambda * std(b.v_dists[i])
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


@with_kw mutable struct ExploreNStepsPolicy <: Policy
    pomdp::LiPOMDP
    explore_steps::Int64
    curr_steps::Int64
end

function POMDPs.action(p::ExploreNStepsPolicy, b::LiBelief)
    chosen_action = nothing
    # Explore all that needs exploring first
    if p.curr_steps < p.explore_steps
        index = rand(1:4)
        chosen_action = Action("EXPLORE$(index)")
    else
        scores = zeros(p.pomdp.n_deposits)
        for i in 1:p.pomdp.n_deposits
            if can_explore_here(Action("MINE$(i)"), b)
                score = mean(b.deposit_dists[i]) / p.pomdp.CO2_emissions[i]
            else
                score = -Inf
            end
            scores[i] = score
        end

        _, best_mine = findmax(scores)

        chosen_action = Action("MINE$(best_mine)")
    end
    p.curr_steps += 1
    return chosen_action
end

function POMDPs.updater(policy::ExploreNStepsPolicy)
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
            score = mean(b.deposit_dists[i]) / p.pomdp.CO2_emissions[i]
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

function POMDPs.updater(policy::POMCPOWPlanner{LiPOMDP,POMCPOW.POWNodeFilter,MaxUCB,POMCPOW.RandomActionGenerator{Random.AbstractRNG},typeof(estimate_value),Int64,Float64,POMCPOWSolver{Random.AbstractRNG,POMCPOW.var"#6#12"}})
    return LiBeliefUpdater(policy.problem)
end

function POMDPs.updater(policy::MCTS.DPWPlanner{GenerativeBeliefMDP{LiPOMDP,LiBeliefUpdater,LiBelief{Normal{Float64}},Action},
    LiBelief{Normal{Float64}},
    Action,
    MCTS.SolvedRolloutEstimator{EfficiencyPolicyWithUncertainty,Random.AbstractRNG},
    RandomActionGenerator{Random.AbstractRNG},MCTS.var"#18#22",Random.AbstractRNG})
    return LiBeliefUpdater(policy.solved_estimate.policy.pomdp)
end

function POMDPs.updater(policy::MCTS.DPWPlanner{GenerativeBeliefMDP{LiPOMDP,LiBeliefUpdater,ContinueTerminalBehavior{LiPOMDP,LiBeliefUpdater},LiBelief{Normal{Float64}},Action},LiBelief{Normal{Float64}},Action,MCTS.SolvedRolloutEstimator{EfficiencyPolicyWithUncertainty,Random.Random.AbstractRNG},RandomActionGenerator{Random.Random.AbstractRNG},MCTS.var"#18#22",Random.Random.AbstractRNG})
    return LiBeliefUpdater(policy.solved_estimate.policy.pomdp)
end


struct AusDomPolicy <: Policy
    pomdp::LiPOMDP
    t::Vector{Int}
end

function POMDPs.action(p::AusDomPolicy, b::LiBelief)
    if b.t == p.t[1]
        return MINE3
    elseif b.t == p.t[2]
        return MINE4
    elseif b.t == p.t[3]
        return MINE1
    elseif b.t == p.t[4]
        return MINE2
    else
        return DONOTHING
    end
end

function POMDPs.action(p::AusDomPolicy, s::State)  # Changed from Deterministic{State}
    if s.t == p.t[1]
        return MINE3
    elseif s.t == p.t[2]
        return MINE4
    elseif s.t == p.t[3]
        return MINE1
    elseif s.t == p.t[3]
        return MINE2
    else
        return DONOTHING
    end
end

function POMDPs.updater(policy::AusDomPolicy)
    return LiBeliefUpdater(policy.pomdp)
end





struct HeuristicPolicy <: Policy
    pomdp::LiPOMDP
    t_mine::Vector{Int}
    t_restore::Vector{Int}
    t_explore::Vector{Vector{Int}}
end

function POMDPs.action(p::HeuristicPolicy, b::LiBelief)
    if b.t == p.t_mine[1]
        return MINE1
    elseif b.t == p.t_mine[2]
        return MINE2
    elseif b.t == p.t_mine[3]
        return MINE3
    elseif b.t == p.t_mine[4]
        return MINE4
    elseif b.t == p.t_restore[1]
        return RESTORE1
    elseif b.t == p.t_restore[2]
        return RESTORE2
    elseif b.t == p.t_restore[3]
        return RESTORE3
    elseif b.t == p.t_restore[4]
        return RESTORE4
    elseif b.t in p.t_explore[1]
        return EXPLORE1
    elseif b.t in p.t_explore[2]
        return EXPLORE2
    elseif b.t in p.t_explore[3]
        return EXPLORE3
    elseif b.t in p.t_explore[4]
        return EXPLORE4
    else
        return DONOTHING
    end
end

function POMDPs.updater(policy::HeuristicPolicy)
    return LiBeliefUpdater(policy.pomdp)
end


#IMPORT ONLY POLICY -- never mines from domestic deposits, prioritizes foreign deposits with largest volume

@with_kw mutable struct ImportOnlyPolicy <: Policy
    pomdp::LiPOMDP
    explore_steps::Int64        # Max number of exploration steps
    curr_steps::Int64 = 1       # Current step counter
    explored_sites::Vector{Bool} = fill(false, pomdp.n_deposits) # Tracks which sites have been explored
end

# Alternative constructor
function ImportOnlyPolicy(pomdp::LiPOMDP, explore_steps::Int64, curr_steps::Int64=1)
    n_deposits = pomdp.n_deposits
    explored_sites = fill(false, n_deposits)  # Initially no sites are explored
    return ImportOnlyPolicy(pomdp=pomdp, explore_steps=explore_steps, 
                          curr_steps=curr_steps, explored_sites=explored_sites)
end

function POMDPs.action(p::ImportOnlyPolicy, b::LiBelief)
    chosen_action = nothing
    
    # Phase 1: Exploration phase
    if p.curr_steps <= p.explore_steps
        # Find the next unexplored site
        for i in 1:p.pomdp.n_deposits
            if !p.explored_sites[i]
                p.explored_sites[i] = true
                chosen_action = Action("EXPLORE$(i)")
                break
            end
        end
        
        # If all sites have been explored but we're still in exploration phase,
        # pick a random site to explore again
        if isnothing(chosen_action)
            index = rand(1:p.pomdp.n_deposits)
            chosen_action = Action("EXPLORE$(index)")
        end
    else
        # Phase 2: Mining phase - Only consider foreign deposits (3 and 4)
        foreign_indices = [3, 4]  # Indices for foreign deposits
        scores = fill(-Inf, p.pomdp.n_deposits)
        
        for i in foreign_indices
            if can_explore_here(Action("MINE$(i)"), b)
                # Prioritize by estimated volume
                scores[i] = mean(b.deposit_dists[i])
            end
        end
        
        # Find the foreign deposit with highest estimated volume
        max_score, best_mine = findmax(scores)
        
        # If a valid foreign site was found
        if max_score > -Inf
            chosen_action = Action("MINE$(best_mine)")
        else
            # If no valid site, do nothing
            chosen_action = Action("DONOTHING")
        end
    end
    
    p.curr_steps += 1
    return chosen_action
end

function POMDPs.updater(policy::ImportOnlyPolicy)
    return LiBeliefUpdater(policy.pomdp)
end

#NO EXPLORATION POLICY --  assumes correct initial belief and start mining from the sites with largest estimated reserve
@with_kw mutable struct NoExplorationPolicy <: Policy
    pomdp::LiPOMDP
end

function POMDPs.action(p::NoExplorationPolicy, b::LiBelief)
    # Calculate the expected amount of lithium at each deposit based on current belief
    scores = zeros(p.pomdp.n_deposits)
    
    for i in 1:p.pomdp.n_deposits
        if can_explore_here(Action("MINE$(i)"), b)
            # Use mean of the belief distribution for the deposit
            score = mean(b.deposit_dists[i])
        else
            score = -Inf
        end
        scores[i] = score
    end
    
    # Mine the deposit with the highest expected amount
    _, best_mine = findmax(scores)
    
    return Action("MINE$(best_mine)")
end

function POMDPs.updater(policy::NoExplorationPolicy)
    return LiBeliefUpdater(policy.pomdp)
end