using Distributions
using ParticleFilters


function simulate_policy(pomdp, policy, up, b0, s0; max_steps=10, rng=MersenneTwister(1))
    r_total = 0.
    r_disc = 0.
    e_total = 0.
    e_disc = 0.
    t = 0
    d = 1.
    b = deepcopy(b0)
    s = deepcopy(s0)
    while (!isterminal(pomdp, s) && t < max_steps)
        t += 1
        a = action(policy, b)
        (s, o, r) = gen(pomdp, s, a, rng)
        b = update(up, b, a, o)
        e = get_action_emission(pomdp, a)
        r_total += r
        r_disc += r*d        
        e_total += e
        e_disc += e*d
        d *= discount(pomdp)
        #@show(t=t, s=s, a=a, r=r, o=o)
    end
    return (rdisc=r_disc, edisc=e_disc, rtot=r_total, etot=e_total, vt=s.Vₜ)
end

function simulate_mcts(pomdp, policy, s0; max_steps=10, rng=MersenneTwister(1))
    r_total = 0.
    r_disc = 0.
    e_total = 0.
    e_disc = 0.
    t = 0
    d = 1.
    s = deepcopy(s0)
    while (!isterminal(pomdp, s) && t < max_steps)
        t += 1
        a = action(policy, s)
        (s, o, r) = gen(pomdp, s, a, rng)
        e = get_action_emission(pomdp, a)
        r_total += r
        r_disc += r*d        
        e_total += e
        e_disc += e*d
        d *= discount(pomdp)
        #@show(t=t, s=s, a=a, r=r, o=o)
    end
    return (rdisc=r_disc, edisc=e_disc, rtot=r_total, etot=e_total, vt=s.Vₜ)
end


function compute_chunk_boundaries(quantile_vols::Vector{Float64})
    n = length(quantile_vols)
    @assert n > 0 "quantile_vols must not be empty"

    chunk_boundaries = Vector{Float64}(undef, n-1)
    for i = 2:n
        chunk_boundaries[i-1] = (quantile_vols[i] + quantile_vols[i-1]) / 2
    end
    return chunk_boundaries
end

function compute_chunk_probs(chunk_boundaries::Vector{Float64}, site_dist::Normal)
    n = length(chunk_boundaries)
    @assert n > 0 "chunk_boundaries must not be empty"

    chunk_probs = Vector{Float64}(undef, n+1)

    chunk_probs[1] = cdf(site_dist, chunk_boundaries[1])
    for i = 2:n
        chunk_probs[i] = cdf(site_dist, chunk_boundaries[i]) - cdf(site_dist, chunk_boundaries[i-1])
    end
    chunk_probs[n+1] = 1 - cdf(site_dist, chunk_boundaries[n])

    return chunk_probs
end

function get_action_emission(P, a)
    action_type = get_action_type(a)
    action_number = get_site_number(a)
    
    # Subtract carbon emissions (if relevant)
    r3 = (action_type == "MINE") ? P.CO2_emissions[action_number] * -1 : 0
    return r3
end

function get_site_number(a::Action)
    action_str = string(a)
    len = length(action_str)
    deposit_number = Int(action_str[len]) - 48  # -48 because Int() gives me the ascii code
    return deposit_number
end

# I'm sure there's some builtin for this but I couldn't find it lol
function splice(begin_index, end_index, str)
    result = ""
    for i = begin_index:end_index
        result = result * str[i]
    end
    return result
end

function get_action_type(a::Action)
    action_str = string(a)
    len = length(action_str)
    action_type = splice(1, len - 1, action_str)
    return action_type
end

function str_to_action(s::String)
    if s == "MINE1"
        return MINE1
    elseif s == "MINE2"
        return MINE2
    elseif s == "MINE3"
        return MINE3
    elseif s == "MINE4"
        return MINE4
    elseif s == "EXPLORE1"
        return EXPLORE1
    elseif s == "EXPLORE2"
        return EXPLORE2
    elseif s == "EXPLORE3"
        return EXPLORE3
    else
        return EXPLORE4
    end
end

function can_explore_here(a::Action, b::Any)
    action_type = get_action_type(a)
    site_number = get_site_number(a)

    if action_type == "MINE" || isa(b, ParticleFilters.ParticleCollection{State})
        return true
    end

    if isa(b, POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}})
        b = sample_state_belief(b)
    end
    
    return !b.have_mined[site_number]        
end


function sample_state_belief(b::POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}}, bin_beliefstate=true)
    
    b_dist = b.sr_belief.dist

    out =  rand(b_dist)[1]
    return out
end

function get_all_states(b::POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}})
    dist = b.sr_belief.dist
    states = [x[1] for x in dist.items]
    states = [x.deposits for x in states]
    return states
end

function get_all_states(s::State)
    return [s.deposits]
end


function compute_portion_below_threshold(P, b, idx::Int64)
    if isa(b, LiBelief)
        dist = b.deposit_dists[idx]
        portion_below_threshold = cdf(dist, P.min_n_units)
    elseif isa(b, ParticleCollection{State}) || isa(b, State)
        portion_below_threshold = 0.
    else
        sampled_belief = get_all_states(b)
        n_rows = length(sampled_belief)

        #! 
        num_below_threshold = sum(row[idx] < P.min_n_units for row in sampled_belief)
        portion_below_threshold = num_below_threshold / n_rows
    end
    return portion_below_threshold
end