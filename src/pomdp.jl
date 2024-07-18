#=
Modeling the US Path to Lithium Self Sufficiency Using POMDPs
Summer 2023
Yasmine Alonso, Mansur Arief, Anthony Corso, Jef Caers, and Mykel Kochenderfer

File: LiPOMDP.jl
----------------
This file contains the implementation of the LiPOMDP. 
=#

# All of the imports


@with_kw mutable struct State
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
    t::Float64 = 0  # current time
    Vₜ::Float64 = 0  # current amt of Li mined up until now
    have_mined::Vector{Bool} = [false, false, false, false]  # Boolean value to represent whether or not we have taken a mine action
end

# All potential actions
@enum Action MINE1 MINE2 MINE3 MINE4 EXPLORE1 EXPLORE2 EXPLORE3 EXPLORE4
rng = MersenneTwister(1)

@with_kw mutable struct LiPOMDP <: POMDP{State, Action, Any} 
    t_goal = 10 #time goal, want to wait 10 years
    σ_obs = 0.1
    Vₜ_goal = 8 #Volume goal
    γ = 0.98 #discounted reward
    n_deposits = 4 
    bin_edges = [0.25, 0.5, 0.75]  # Used to discretize observations
    obs_type = "continuous"
    cdf_threshold = 0.1  # threshold allowing us to mine or not
    min_n_units = 3  # minimum number of units required to mine. So long as cdf_threshold portion of the probability
    obj_weights = [0.33, 0.33, 0.33]  # how we want to weight each component of the reward 
    CO2_emissions::Vector{Float64} = [5, 7, 2, 5]  #[C₁, C₂, C₃, C₄] amount of CO2 each site emits
    null_state::State = State([-1, -1, -1, -1], -1, -1, [true, true, true, true])
    init_state::State = State([8.9, 7, 1.8, 5], 0, 0, [false, false, false, false])  # For convenience for me rn when I want to do a quick test and pass in some state
end

# Belief struct
struct LiBelief{T<:UnivariateDistribution}
    deposit_dists::Vector{T}
    t::Float64
    V_tot::Float64
    have_mined::Vector{Bool} 
end

POMDPs.support(b::LiBelief) = rand(b)

# Unsure if the right way to do this is to have a product distribution over the 4 deposits? 
# Pass in an RNG?
function POMDPs.initialstate(P::LiPOMDP, )
    init_state = State([8.9, 7, 1.8, 5], 0, 0, [false, false, false, false])
    return Deterministic(init_state)
end

# Continuous state space
function POMDPs.states(P::LiPOMDP)
    # Min and max amount per singular deposit
    V_deposit_min = 0
    V_deposit_max = 10

    # Min and max amount total mined, can be at smallest the deposit_min * 4, and at largest, the deposit_max * 4
    V_tot_min = V_deposit_min * P.n_deposits  # 0
    V_tot_max = V_deposit_max * P.n_deposits  # 40

    deposit_vec_bounds = [(V_deposit_min, V_deposit_max) for x in 1:P.n_deposits]  # Make a length-4 vector, one for each deposit
    V_tot_bounds = Interval(V_tot_min, V_tot_max)
    time_bounds = 0:10  # This can be discrete since we're only going a year at a time

    𝒮 = product_state_space(deposit_vec_bounds, V_tot_bounds, time_bounds)  # Cartesian product 
    # QUESTION: how could I add the null state into the space?
    return 𝒮

end

# Action function: now dependent on belief state
function POMDPs.actions(P::LiPOMDP, b)
    potential_actions = [MINE1, MINE2, MINE3, MINE4, EXPLORE1, EXPLORE2, EXPLORE3, EXPLORE4]#actions(P)

    # Checks to ensure that we aren't trying to explore at a site we have already mined at
    potential_actions = filter(a -> can_explore_here(a, b), potential_actions)

    # Ensures that there is <= 10% (or P.cdf_threshold) of the belief distribution below the P.min_n_units
    for i = 1:4
        if isa(b, POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}})
            belief = convert_particle_collection_to_libelief(b) #! made change here too 
            if belief.have_mined[i]  # handle POMCPOW
                continue
            end
        else
            if b.have_mined[i]  #handle LiBelief
                continue
            end
        end

        # dist = b.deposit_dists[i]
        # portion_below_threshold = cdf(dist, P.min_n_units)
        portion_below_threshold = compute_portion_below_threshold(P, b, i)
        if portion_below_threshold > P.cdf_threshold  # BAD!

            bad_action_str = "MINE" * string(i)
            bad_action = str_to_action(bad_action_str)
            # Ensure that this bad_action is not in potential_actions
            potential_actions = filter(a -> a != bad_action, potential_actions)
        end   
    end 
    return potential_actions
end

# Reward function: returns the reward for being in state s and taking action a
# Reward is comprised of three parts:
#       1. Whether or not we have reached our time delay + volume goal (1 if yes, 0 if no)
#       2. The amount of volume we have mined
#       3. The amount of CO2 emissions we have produced if taking a mine action (negative)
# The three parts of the reward are then weighted by the obj_weights vector and returned.

function POMDPs.reward(P::LiPOMDP, s::State, a::Action)

    if isterminal(P, s)
        return 0
    end

    # See if we achieve both time delay goal and volume amount goal
    r1 = (s.t >= P.t_goal && s.Vₜ >= P.Vₜ_goal) ? 100 : 0

    r2 = s.Vₜ

    # Calculates how much CO2 taking this action will emit
    r3 = get_action_emission(P, a)

    reward = dot([r1, r2, r3], P.obj_weights)

    return reward
end

# Gen function: basically the trnasition function, but also samples an observation and reward, returning as a tuple
function POMDPs.gen(P::LiPOMDP, s::State, a::Action, rng::AbstractRNG)
    next_state::State = deepcopy(s) # Make a copy!!! need to be wary of this in Julia deepcopy might be slow
    next_state.t = s.t + 1  # Increase time by 1 in all cases

    if s.t >= P.t_goal && s.Vₜ >= P.Vₜ_goal  # If we've reached all our goals, we can terminate
        next_state = deepcopy(P.null_state)

    else
        action_type = get_action_type(a)
        site_number = get_site_number(a)

        # If we choose to MINE, so long as there is Li available to us, decrease amount in deposit by one unit
        # and increase total amount mined Vₜ by 1 unit. We do not have any transitions for EXPLORE actions because 
        # exploring does not affect state
        if action_type == "MINE" && s.deposits[site_number] >= 1
            next_state.deposits[site_number] = s.deposits[site_number] - 1
            next_state.Vₜ = s.Vₜ + 1
        end

        # If we're mining, update state to reflect that we now have mined and can no longer explore
        if action_type == "MINE"
            next_state.have_mined[site_number] = true
        end
    end
    # Now sample an observation and get the reward as well

    # o is continuous
    o = rand(rng, observation(P, a, next_state))  # Vector of floats
    r = reward(P, s, a)

    out = (sp=next_state, o=o, r=r)  
    return out
end


# Observation function
function POMDPs.observation(P::LiPOMDP, a::Action, sp::State)
    # When we take an action to EXPLORE one of the four sites, we only really gain an observation on said
    # state. So, the other remaining three states have this kinda sentinel distribution thing of -1 to represent
    # that it's not really important/relevant
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
    end

    #handles EXPLORE
    if P.obs_type == "continuous"        
        temp[site_number] = Normal(sp.deposits[site_number], P.σ_obs)
        #println("returning cts obs type")
        return product_distribution(temp)
    else    
        site_dist = Normal(sp.deposits[site_number], P.σ_obs)
        # sample_point = rand(site_dist)  # Step 1: get us a random sample on that distribution

        quantile_vols = collect(0.:1.:10.)#quantile(site_dist, P.bin_edges)  # Step 2: get the Li Volume amounts that correspond to each quantile
        #quantile_vols = [x for x in quantile_vols]  # Round to 1 decimal place

        # Now get the chunk boundaries (Dashed lines in my drawings)
        chunk_boundaries = compute_chunk_boundaries(quantile_vols)

        # Now compute the probabilities of each chunk
        probs = compute_chunk_probs(chunk_boundaries, site_dist)
        #println("sp: ", sp, "q :", quantile_vols)
        
        # I believe the idea was that with other solvers, we need an observation fn that returns an explicit
        # distribution, not just a sample. So, I decided to use a sparsecat here, but I'm unsure, since all of this doesn't
        # really seem to be working properly :(
        temp[site_number] = DiscreteNonParametric(quantile_vols, probs)
        return product_distribution(temp)
    end
end

POMDPs.discount(P::LiPOMDP) = P.γ

POMDPs.isterminal(P::LiPOMDP, s::State) = s == P.null_state

struct LiBeliefUpdater <: Updater
    P::LiPOMDP
end

function POMDPs.initialize_belief(up::Updater, s)
    # Initialize belief to be a vector of 4 normal distributions, one for each deposit
    # Each normal distribution has mean equal to the amount of Li in that deposit, and
    # standard deviation equal to P.σ_obs
    s = rand(s)
    deposit_dists = [Normal(s.deposits[1], 0.2), Normal(s.deposits[2], 0.2), Normal(s.deposits[3], 0.2), Normal(s.deposits[4], 2.0)]
    t = s.t
    V_tot = s.Vₜ
    have_mined = s.have_mined
    return LiBelief(deposit_dists, t, V_tot, have_mined)
end

# takes in a belief, action, and observation and uses it to update the belief
function POMDPs.update(up::Updater, b::LiBelief, a::Action, o::Vector{Float64})
    # EXPLORE actions: Adjust mean of the distribution corresponding to the proper deposit, using the Kalman
    # predict/update step (see kalman_step function above). Time increases by 1 in the belief.
    # Return new belief, with everything else untouched (EXPLORE only allows us to gain info about one site) 
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
