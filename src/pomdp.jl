#=
Original Authors: Yasmine Alonso, Mansur Arief, Anthony Corso, Jef Caers, and Mykel Kochenderfer
Extended by: CJ Oshiro, Mansur Arief, Mykel Kochenderfer
----------------
=#


POMDPs.support(b::LiBelief) = rand(b)  #TODO: change to be the support of deposit distributions


function POMDPs.initialstate(P::LiPOMDP)
    return Deterministic(P.init_state)
end

function POMDPs.states(P::LiPOMDP)

    deposits = collect(P.V_deposit_min:P.Δdeposit:P.V_deposit_max)
    V_tot_bounds = collect(P.V_deposit_min * P.n_deposits:P.ΔV:P.V_deposit_max * P.n_deposits )
    booleans = [true, false]
    time_bounds = collect(1:P.time_horizon)


    # deposits_combinations = Iterators.product(d_ for d_ in fill(deposits, P.n_deposits)) #TODO: make this more flexible
    all_combinations = Iterators.product(deposits, deposits, deposits, deposits, V_tot_bounds, time_bounds, booleans, booleans, booleans, booleans)
    𝒮 = [State([s[1], s[2], s[3], s[4]],  s[5], s[6], [s[7], s[8], s[9], s[10]]) for s in all_combinations] 
    push!(𝒮, P.null_state)

    return 𝒮

end

# Removed the dependency on the belief
function POMDPs.actions(P::LiPOMDP)
    potential_actions = [MINE1, MINE2, MINE3, MINE4, EXPLORE1, EXPLORE2, EXPLORE3, EXPLORE4]
    return potential_actions
end


function POMDPs.reward(P::LiPOMDP, s::State, a::Action)

    domestic_mining_actions = [MINE1, MINE2]

    if isterminal(P, s)
        return 0
    end

    #TODO: move this to the problem struct
    demand_unfulfilled_penalty = -100
    domestic_mining_penalty = -1000

    
    # Obj #1: delay mining domesticall P.t_goal years, and if we do that before P.t_goal, we are penalized
    r1 = s.t <= P.t_goal && a in domestic_mining_actions ? domestic_mining_penalty : 0 

    # Obj #2: maximize the amount of Li mined
    r2 = s.Vₜ

    # Obj #3: minimize CO2 emissions
    r3 = get_action_emission(P, a)

    # Obj #4: satisfy the demand at everytimestep
    production = sum(s.have_mined)*P.mine_output
    r4 = P.demands[Int(s.t)] > production ? demand_unfulfilled_penalty : 0

    #TODO: add r5: NPV
    #whenever you mine, you incur capex: -100
    #at everytime step, incur opex: -10
    #at everytime step, generate revenue: price*production

    reward = dot([r1, r2, r3, r4], P.obj_weights) #TODO:modify this to include r5, num_objectives

    return reward
end


function POMDPs.transition(P::LiPOMDP, s::State, a::Action)

    if s.t >= P.time_horizon || isterminal(P, s)  
        sp = deepcopy(P.null_state)
        return Deterministic(sp)
    else
        sp = deepcopy(s) # Make a copy!!! need to be wary of this in Julia deepcopy might be slow
        sp.t = s.t + 1  # Increase time by 1 in all cases
        action_type = get_action_type(a)
        site_number = get_site_number(a)

        if action_type == "MINE" 
            if s.deposits[site_number] >= P.mine_output
                mine_output = P.mine_output
            else
                mine_output = s.deposits[site_number]
            end
            sp.deposits[site_number] = s.deposits[site_number] - mine_output
            sp.Vₜ = s.Vₜ + mine_output
        end

        # If we're mining, update state to reflect that we now have mined and can no longer explore
        if action_type == "MINE"
            sp.have_mined[site_number] = true
        end

        return Deterministic(sp)
    end
   
end

function POMDPs.gen(P::LiPOMDP, s::State, a::Action, rng::AbstractRNG)

    sp =rand(rng, transition(P, s, a))  
    o = rand(rng, observation(P, a, sp))  
    r = reward(P, s, a)
    out = (sp=sp, o=o, r=r)  

    return out
end


function POMDPs.observation(P::LiPOMDP, a::Action, sp::State)

    site_number = get_site_number(a)  #1, 2, 3, or 4, basically last character of    
    action_type = get_action_type(a)  # "EXPLORE" or "MINE"

    sentinel_dist = DiscreteNonParametric([-1.], [1.])
    temp::Vector{UnivariateDistribution} = fill(sentinel_dist, 4)

    # handle degenerate case where we have no more Li at this site
    if sp.deposits[site_number] <= 0
        site_dist = sentinel_dist
        return product_distribution(temp)        
    end

    if action_type == "MINE"
        return product_distribution(temp) 
    else    
        site_dist = Normal(sp.deposits[site_number], P.σ_obs)
        quantile_vols = collect(0.:1.:60.) #TODO: put in the problem struct

        # Now get the chunk boundaries (Dashed lines in my drawings)
        chunk_boundaries = compute_chunk_boundaries(quantile_vols)

        # Now compute the probabilities of each chunk
        probs = compute_chunk_probs(chunk_boundaries, site_dist)

        temp[site_number] = DiscreteNonParametric(quantile_vols, probs)

        return product_distribution(temp)

    end
end


POMDPs.discount(P::LiPOMDP) = P.γ

POMDPs.isterminal(P::LiPOMDP, s::State) = s == P.null_state || s.t >= P.time_horizon

function POMDPs.initialize_belief(up::LiBeliefUpdater)

    deposit_dists = [
        Normal(up.P.init_state.deposits[1]),
        Normal(up.P.init_state.deposits[2]),
        Normal(up.P.init_state.deposits[3]),
        Normal(up.P.init_state.deposits[4])
    ]

    t = 1.0
    V_tot = 0.0
    have_mined = [false, false, false, false]
    
    return LiBelief(deposit_dists, t, V_tot, have_mined)
end

POMDPs.initialize_belief(up::Updater, dist) = POMDPs.initialize_belief(up)


function POMDPs.update(up::Updater, b::LiBelief, a::Action, o::Vector{Float64})
    
    action_type = get_action_type(a)
    site_number = get_site_number(a)
    P = up.P

    if action_type == "EXPLORE"
        bi = b.deposit_dists[site_number]  # This is a normal distribution
        μi = mean(bi)
        σi = std(bi)
        zi = o[site_number]
        μ_prime, σ_prime = kalman_step(P, μi, σi, zi)
        bi_prime = Normal(μ_prime, σ_prime)
        
        # Default, not including updated belief
        belief = LiBelief([b.deposit_dists[1], b.deposit_dists[2], b.deposit_dists[3], b.deposit_dists[4]], b.t + 1, b.V_tot, b.have_mined)
        
        # Now, at the proper site number, update to contain the updated belief
        belief.deposit_dists[site_number] = bi_prime
        
        
        return belief

    # MINE actions: Shifts our mean of the distribution corresponding to the proper deposit down by 1 (since we
    # have just mined one unit deterministically). Does not affect certainty at all. 
    else # a must be a MINE action
        bi = b.deposit_dists[site_number]
        μi = mean(bi)
        σi = std(bi)
        
        if μi >= 1
            μi_prime = μi - 1
            n_units_mined = 1  # we were able to mine a unit
        else 
            μi_prime = μi
            n_units_mined = 0  # we did NOT mine a unit
        end
        
        # Default, not including updated belief
        belief = LiBelief([b.deposit_dists[1], b.deposit_dists[2], b.deposit_dists[3], b.deposit_dists[4]], b.t + 1, b.V_tot + n_units_mined, [b.have_mined[1], b.have_mined[2], b.have_mined[3], b.have_mined[4]])
        # Now, at the proper site number, update to contain the updated belief
        belief.deposit_dists[site_number] = Normal(μi_prime, σi)
        belief.have_mined[site_number] = true

        return belief
    end 
end
