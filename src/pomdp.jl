function POMDPs.initialstate(P::LiPOMDP)
    return Deterministic(P.init_state)
end


# function POMDPs.actions(P::LiPOMDP)
#     potential_actions = [DONOTHING MINE1 MINE2 MINE3 MINE4 EXPLORE1 EXPLORE2 EXPLORE3 EXPLORE4 RESTORE1 RESTORE2 RESTORE3 RESTORE4]
#     return potential_actions
# end

#TODO: ensure we use this dispatch. it is stil using the old way without state
function POMDPs.actions(P::LiPOMDP, pc::ParticleCollection{State})
    actions = [DONOTHING]

    #if no particles, return the default action
    if isempty(pc.particles)
        return actions
    end
    s = pc.particles[1]
    for i in 1:P.n
        if !s.m[i]            
            if i == 1
                push!(actions, MINE1)
                push!(actions, EXPLORE1)
            elseif i == 2
                push!(actions, MINE2)
                push!(actions, EXPLORE2)
            elseif i == 3
                push!(actions, MINE3)
                push!(actions, EXPLORE3)
            elseif i == 4
                push!(actions, MINE4)
                push!(actions, EXPLORE4)
            end            
        else
            if s.v[i] < P.ΔV
                if i == 1
                    push!(actions, RESTORE1)
                elseif i == 2
                    push!(actions, RESTORE2)
                elseif i == 3
                    push!(actions, RESTORE3)
                elseif i == 4
                    push!(actions, RESTORE4)
                end
            end
        end
    end
end


function POMDPs.actions(P::LiPOMDP, b::LiBelief)
    actions = [DONOTHING]

    for i in 1:P.n
        if !b.m[i]            
            if i == 1
                push!(actions, MINE1)
                push!(actions, EXPLORE1)
            elseif i == 2
                push!(actions, MINE2)
                push!(actions, EXPLORE2)
            elseif i == 3
                push!(actions, MINE3)
                push!(actions, EXPLORE3)
            elseif i == 4
                push!(actions, MINE4)
                push!(actions, EXPLORE4)
            end            
        else
            if mean(b.v_dists[i]) < P.ΔV
                if i == 1
                    push!(actions, RESTORE1)
                elseif i == 2
                    push!(actions, RESTORE2)
                elseif i == 3
                    push!(actions, RESTORE3)
                elseif i == 4
                    push!(actions, RESTORE4)
                end
            end
        end
    end
    return actions
end


function compute_r1(P::LiPOMDP, s::State, a::Action)
    action_type = get_action_type(a)
    site_num = get_site_number(a)

    if action_type == "MINE" && s.t < P.td && site_num in P.Jd && !s.m[site_num] 
        penalty = P.pd
    else
        penalty = 0
    end
        
    return -penalty
end

function compute_ore_arrived_at_plants(P::LiPOMDP, o::Observation)
    domestic = sum(o.E[j] for j in P.Jd)
    imported = sum(o.E[j] for j in P.Jf)
    return (domestic=domestic, imported=imported)
end

function compute_r2(P::LiPOMDP, s::State, a::Action)
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

function compute_r3(P::LiPOMDP, s::State, a::Action, o::Observation)
    production = P.ρ * sum(o.E) + sum(o.Z)
    return -max(0, P.d[Int(s.t)] - production)
end

function compute_r4(P::LiPOMDP, s::State, a::Action, o::Observation)

    action_type = get_action_type(a)
    r = 0.0
    r += P.p * P.ρ * sum(compute_ore_arrived_at_plants(P, o)) #revenue
    r -= (action_type == "EXPLORE" ? P.ce : 0) # exploration cost
    r -= (action_type == "MINE" ? P.cb : 0) # building mining cost
    r -= (action_type == "RESTORE" ? P.cr : 0) # restoration cost
    r -= P.co*sum(s.m) # mining operating cost
    r -= sum(P.ct[j] * (o.E[j] + o.Z[j]) for j in 1:P.n) # transportation cost #TODO: update the paper to reflect this
    r -= P.cp*sum(compute_ore_arrived_at_plants(P, o)) # processing cost #TODO: update the paper to reflect this    

    return r
end

function POMDPs.reward(P::LiPOMDP, s::State, a::Action, o::Observation)

    if isterminal(P, s)
        return 0.0
    end
    
    r1 = compute_r1(P, s, a)
    r2 = compute_r2(P, s, a)
    r3 = compute_r3(P, s, a, o)
    r4 = compute_r4(P, s, a, o)
    r = dot([r1, r2, r3, r4], P.w)

    return (r1=r1, r2=r2, r3=r3, r4=r4, r=r)
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
        L = [ j in P.Jf && sp.m[j] ? L[j] : 0.0 for j in 1:P.n]
    end

    Z = E .- L

    return (sp=sp, E=E, L=L, Z=Z)
end

function POMDPs.gen(P::LiPOMDP, s::State, a::Action, rng::AbstractRNG)

    sp, E, L, Z = transition(P, s, a, rng)

    #observation
    v_pred = rand(rng, observation(P, a, sp))
    o = Observation(v=v_pred, E=E, Z=Z)

    #compute reward
    r1, r2, r3, r4, r = reward(P, s, a, o)

    return (sp=sp, o=o, r=r)  
end


function POMDPs.observation(P::LiPOMDP, a::Action, sp::State)

    site_number = get_site_number(a)  
    action_type = get_action_type(a)  

    sentinel_dist = DiscreteNonParametric([-1.], [1.])
    temp::Vector{UnivariateDistribution} = fill(sentinel_dist, P.n)

    # handle do nothing action
    if site_number <= 0
        return product_distribution(temp)        
    end

    # handle degenerate case where we have no more Li at this site    
    
    if sp.v[site_number] <= 0
        site_dist = sentinel_dist
        return product_distribution(temp)        
    end
    

    if action_type == "EXPLORE" 
        site_dist = Normal(sp.v[site_number], P.σo)
        quantile_vols = P.disc_points 
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
            quantile_vols = P.disc_points 
            chunk_boundaries = compute_chunk_boundaries(quantile_vols)
            probs = compute_chunk_probs(chunk_boundaries, site_dist)
            temp[site_number] = DiscreteNonParametric(quantile_vols, probs)
        end
    end
    
    return product_distribution(temp) 
end


POMDPs.discount(P::LiPOMDP) = P.γ

POMDPs.isterminal(P::LiPOMDP, s::State) = s == P.null_state || s.t > P.T

function POMDPs.initialize_belief(up::LiBeliefUpdater)

    deposit_dists = [
        Normal(up.P.init_state.v[1]),
        Normal(up.P.init_state.v[2]),
        Normal(up.P.init_state.v[3]),
        Normal(up.P.init_state.v[4])
    ]

    
    return LiBelief(deposit_dists, 1, 0.0, 0.0, [false, false, false, false])
end



function POMDPs.update(up::Updater, b::LiBelief, a::Action, o::Observation)
    
    # println("Observation: $o, Belief: $b, Action: $a")
    
    action_type = get_action_type(a)
    site_number = get_site_number(a)
    P = up.P
    ΔV = 0.0
    ΔI = 0.0

    v_dists= [b.v_dists[1], b.v_dists[2], b.v_dists[3], b.v_dists[4]]

    for j in 1:P.n
        if o.v[j] != -1
            bi = b.v_dists[site_number]  # This is a normal distribution
            μi = mean(bi)
            σi = std(bi)
            zi = o.v[site_number]
            if j == site_number && action_type == "EXPLORE" 
                σo = P.σo   #use observation noise   
            else
                σo = std(P.ϕ[j]) #use extraction noise
            end
            μ_prime, σ_prime = kalman_step(σo, μi, σi, zi)
            bi_prime = Normal(μ_prime, σ_prime)
            v_dists[site_number] = bi_prime
        end

        if j in P.Jd
            ΔV += o.E[j]
        else
            ΔI += o.E[j]
        end
    end

    
    if site_number > 0
        if site_number in P.Jd
            ΔV += o.E[site_number]
        else
            ΔI += o.E[site_number]
        end
    end

    
    if action_type == "MINE"
        m = [false, false, false, false]
        m[site_number] = true
    else
        m = b.m
    end

    bp = LiBelief(
        v_dists,
        b.t + 1,
        b.Vₜ + ΔV,
        b.Iₜ + ΔI,
        m
    )

    return bp
end
