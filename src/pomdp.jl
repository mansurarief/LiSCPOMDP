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
    V_tot_bounds = collect(P.V_deposit_min * P.n:P.ΔV:P.V_deposit_max * P.n )
    I_tot_bounds = collect(P.V_deposit_min * P.n:P.ΔV:P.V_deposit_max * P.n )
    booleans = [true, false]
    time_bounds = collect(1:P.T)
    all_combinations = Iterators.product(deposits, deposits, deposits, deposits, V_tot_bounds, I_tot_bounds, time_bounds, booleans, booleans, booleans, booleans)
    𝒮 = [State([s[1], s[2], s[3], s[4]],  s[5], s[6], s[7], [s[8], s[9], s[10], s[11]]) for s in all_combinations] 
    println("length of state space: ", length(𝒮))
    println(𝒮[1])
    println(P.null_state)
    push!(𝒮, P.null_state)
    return 𝒮 #TODO: fix error
end

# Removed the dependency on the belief
function POMDPs.actions(P::LiPOMDP)
    potential_actions = [DONOTHING MINE1 MINE2 MINE3 MINE4 EXPLORE1 EXPLORE2 EXPLORE3 EXPLORE4 RESTORE1 RESTORE2 RESTORE3 RESTORE4]
    return potential_actions
end

function compute_r1(P::LiPOMDP, s::State, a::Action)
    action_type = get_action_type(a)
    site_num = get_site_number(a)

    if action_type == "MINE" && !s.m[site_num] && s.t < P.td && site_num in P.Jd
        penalty = P.pd
    else
        penalty = 0
    end
        
    return penalty
end

function compute_r2(P::LiPOMDP, s::State, a::Action)
    return (domestic=s.Vₜ, imported=s.Iₜ)
end

function compute_r3(P::LiPOMDP, s::State, a::Action)
    action_type = get_action_type(a)
    site_num = get_site_number(a)

    if action_type == "MINE" && !s.m[site_num]
        new_emission = get_action_emission(P, a)
    else
        new_emission = 0
    end
    
    existing_emissions = 0
    for i in 1:P.n
        if s.m[i]
            existing_emissions -= P.e[i]
        end
    end

    return new_emission + existing_emissions

end

function compute_r4(P::LiPOMDP, s::State, a::Action, production::Float64)
    return max(0, P.d[Int(s.t)] - production)
end

function compute_r5(P::LiPOMDP, s::State, a::Action, E::Vector{Float64}, Z::Vector{Float64})

    production = P.ρ * sum(E) + sum(Z)

    reward = 0

    action_type = get_action_type(a)
    site_num = get_site_number(a)

    if action_type == "EXPLORE" && !s.m[site_num]
        reward -= P.ce
    end

    if action_type == "MINE" && !s.m[site_num]
        reward -= P.cb
    end



    reward -= P.co*sum(s.m) # mining operating cost


    reward -= sum(P.ct[j] * E[j] for j in 1:P.n) # transportation cost #TODO: add into paper
    reward -= sum(P.cp * Z[j] for j in 1:P.n) # processing cost #TODO: add into paper

    if action_type == "RESTORE" && s.m[site_num]
        reward -= P.cr
    end

    reward += P.p*production

    return reward
end

function POMDPs.reward(P::LiPOMDP, s::State, a::Action, E::Vector{Float64}, Z::Vector{Float64})

    if isterminal(P, s)
        return 0.0
    end

    production = sum(Z) * P.ρ
    r1 = compute_r1(P, s, a)
    r2 = sum(compute_r2(P, s, a)) #r2 output is a tuple(domestic, imported)
    r3 = compute_r3(P, s, a)
    r4 = compute_r4(P, s, a, production)
    r5 = compute_r5(P, s, a, E, Z)
    r = dot([r1, r2, r3, r4, r5], P.obj_weights)

    return r
end


function POMDPs.transition(P::LiPOMDP, s::State, a::Action, rng)
    
    if s.t >= P.T || isterminal(P, s)  
        sp = deepcopy(P.null_state)    
        E = [0. for _ in 1:P.n] 
        L = [0. for _ in 1:P.n]  
    else
        # pre-generate the LCE extracted at each site
        # state transition randomization is mainly due to these two lines
        E = [rand(rng, ϕ) for ϕ in P.ϕ] 
        L = [rand(rng, ψ) for ψ in P.ψ] 

        sp = deepcopy(s) 
        sp.t = s.t + 1  

        ΔV = 0.0
        ΔI = 0.0
        new_mine_output = 0.0

        #process new mine
        action_type = get_action_type(a)
        site_number = get_site_number(a)     

        #process existing mines
        for j in 1:P.n
            if s.m[j]
                mine_output = min(s.v[j], E[j])                
                sp.v[j] = s.v[j] - mine_output
                if j in P.Jd                   
                    ΔV += mine_output
                else
                    ΔI += mine_output
                end
                E[j] = mine_output                
            else
                E[j] = 0.0
            end
        end

        #process new mine        
        if action_type == "MINE"             
            new_mine_output = min(s.v[site_number], E[site_number])
            E[site_number] = new_mine_output
            sp.v[site_number] = s.v[site_number] - E[site_number]            
            sp.m[site_number] = true

            if get_site_number(a) in P.Jd
                ΔV += E[site_number]
            else
                ΔI += E[site_number]
            end
        end

        sp.Vₜ = s.Vₜ + ΔV
        sp.Iₜ = s.Iₜ + ΔI
        L = [ j in P.Ji && sp.m[j] ? L[j] : 0.0 for j in 1:P.n]
    end

    Z = E .- L

    return (sp=sp, E=E, L=L, Z=Z)
end

function POMDPs.gen(P::LiPOMDP, s::State, a::Action, rng::AbstractRNG)

    sp, E, L, Z = transition(P, s, a, rng)

    #compute reward
    r = reward(P, s, a, E, Z)

    #observation
    o = rand(rng, observation(P, a, sp))  

    return (sp=sp, o=o, r=r)  
end


function POMDPs.observation(P::LiPOMDP, a::Action, sp::State)

    site_number = get_site_number(a)  #1, 2, 3, or 4, basically last character of    
    action_type = get_action_type(a)  # "EXPLORE" or "MINE"

    sentinel_dist = DiscreteNonParametric([-1.], [1.])
    temp::Vector{UnivariateDistribution} = fill(sentinel_dist, 4)

    # handle do nothing action
    if site_number <= 0
        site_dist = sentinel_dist
        return product_distribution(temp)        
    end

    # handle degenerate case where we have no more Li at this site    
    if sp.v[site_number] <= 0
        site_dist = sentinel_dist
        return product_distribution(temp)        
    end
    

    if action_type == "EXPLORE" 
        site_dist = Normal(sp.v[site_number], P.σo)
        quantile_vols = collect(0.:1000.:10000.) #TODO: put in the problem struct
        chunk_boundaries = compute_chunk_boundaries(quantile_vols)
        probs = compute_chunk_probs(chunk_boundaries, site_dist)
        temp[site_number] = DiscreteNonParametric(quantile_vols, probs)
        # return product_distribution(temp)
    end

    #process the opened mine
    for j in 1:P.n
        if sp.m[j] && j != site_number
            avg_output = mean(P.ϕ[site_number])
            std_output = std(P.ϕ[site_number])
            site_dist =  Normal(sp.v[site_number]-avg_output, std_output)
            quantile_vols = collect(0.:1000.:10000.) #TODO: put in the problem struct
            chunk_boundaries = compute_chunk_boundaries(quantile_vols)
            probs = compute_chunk_probs(chunk_boundaries, site_dist)
            temp[site_number] = DiscreteNonParametric(quantile_vols, probs)
        end
    end
    
    return product_distribution(temp) 
end


POMDPs.discount(P::LiPOMDP) = P.γ

POMDPs.isterminal(P::LiPOMDP, s::State) = s == P.null_state || s.t >= P.T

function POMDPs.initialize_belief(up::LiBeliefUpdater)

    deposit_dists = [
        Normal(up.P.init_state.v[1]),
        Normal(up.P.init_state.v[2]),
        Normal(up.P.init_state.v[3]),
        Normal(up.P.init_state.v[4])
    ]
    
    return LiBelief(deposit_dists)
end

POMDPs.initialize_belief(up::Updater, dist) = POMDPs.initialize_belief(up)


function POMDPs.update(up::Updater, b::LiBelief, a::Action, o::Vector{Float64})
    
    action_type = get_action_type(a)
    site_number = get_site_number(a)
    P = up.P

    bp = LiBelief([b.v_dists[1], b.v_dists[2], b.v_dists[3], b.v_dists[4]])

    for j in 1:P.n
        if o[j] != -1
            bi = b.v_dists[site_number]  # This is a normal distribution
            μi = mean(bi)
            σi = std(bi)
            zi = o[site_number]
            if j == site_number && action_type == "EXPLORE" 
                σo = P.σo   #use observation noise   
            else
                σo = std(P.ϕ[j]) #use extraction noise
            end
            μ_prime, σ_prime = kalman_step(σo, μi, σi, zi)
            bi_prime = Normal(μ_prime, σ_prime)
            bp.v_dists[site_number] = bi_prime
        end
    end

    return bp
end
