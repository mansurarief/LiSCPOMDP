struct OptimizationPolicy <: Policy
    pomdp::LiPOMDP
    actions::Dict{Int, Dict{Int, Symbol}}  # Time -> Site -> Action type
    valid_window::UnitRange{Int}        # Time window where the optimization is valid
end

"""
Convert optimization model solution to a policy
"""
function create_optimization_policy(model, pomdp, T, I)
    # Helper to convert binary decision to action
    function get_action_at_time(t, i, x_val, y_val, z_val)
        if y_val[t,i] > 0.5  # Construction/Mining decision
            return eval(Meta.parse("MINE$i"))
        elseif z_val[t,i] > 0.5  # Restoration decision
            return eval(Meta.parse("RESTORE$i"))
        elseif x_val[t,i] > 0.5  # Operation continues
            return eval(Meta.parse("EXPLORE$i"))
        else
            return DONOTHING
        end
    end
    
    # Extract values from model
    x_val = JuMP.value.(model[:x])  # Exploration decisions
    y_val = JuMP.value.(model[:y])  # Construction decisions
    z_val = JuMP.value.(model[:z])  # Restoration decisions
    
    # Create actions dictionary
    actions = Dict{Int, Action}()
    for t in T
        for i in I
            actions[t]= get_action_at_time(t, i, x_val, y_val, z_val)
        end
    end
    
    return actions
end

"""
Convert optimization action to POMDP action
"""

function optimization_to_pomdp_action(action_type::Symbol, site::Int)
    if action_type == :MINE
        return eval(Meta.parse("MINE$site"))
    elseif action_type == :RESTORE
        return eval(Meta.parse("RESTORE$site"))
    else
        return DONOTHING
    end
end

"""
Get action for current state/belief
"""
function POMDPs.action(policy::OptimizationPolicy, b::Union{State,LiBelief})
    t = Int(b.t)
    
    # Check if we're within the optimization window
    if !(t in policy.valid_window)
        return DONOTHING
    end
    
    # Find most important action for this timestep
    for i in 1:policy.pomdp.n
        action_type = get(get(policy.actions, t, Dict()), i, :NONE)
        if action_type ∈ [:MINE, :RESTORE]
            return optimization_to_pomdp_action(action_type, i)
        end
    end
    
    return DONOTHING
end

function POMDPs.updater(policy::OptimizationPolicy)
    return LiBeliefUpdater(policy.pomdp)
end