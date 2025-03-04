#=
Modeling the US Path to Lithium Self Sufficiency Using POMDPs
Summer 2023
Yasmine Alonso, Mansur Arief, Anthony Corso, Jef Caers, and Mykel Kochenderfer

File: utils.jl
----------------
This file contains multiple utility functions used throughout the project.
=#

# Used for the POMCPOW solver. Simply sees if we have reached volume goal or not.
function estimate_value(P::LiPOMDP, s, h, steps)
    return s.Vₜ < P.Vₜ_goal ? -100.0 : 0.0
end

#Used for the POMCPOW solver. Same method, defined for specific types to mitigate errors.
function POMCPOW.estimate_value(
    P::LiPOMDP, s::State, h::POWTreeObsNode{POWNodeBelief{State,Action,Any,LiPOMDP},Action,Any,ParticleFilters.ParticleCollection{State}}, steps::Int64)
    return s.Vₜ < P.Vₜ_goal ? -100.0 : 0.0
end

# Runs all simulations for every inputted policy
function evaluate_policies(pomdp::LiPOMDP, policies::Vector, k::Int, max_steps::Int)

    up = LiBeliefUpdater(pomdp)

    for policy in policies
        println("==========start $k simulations for ", typeof(policy), "==========")

        sim_results = replicate_simulation(pomdp, policy, up, k=k, max_steps=max_steps)

        # Print results
        print_policy_results(string(typeof(policy)), sim_results)
    end
end

# Version for when each policy needs a separate pomdp, evaluates a single policy instead of mulitple 
function evaluate_policy(pomdp::LiPOMDP, policy, k::Int, max_steps::Int)

    up = LiBeliefUpdater(pomdp)

    #for policy in policies
    println("==========start $k simulations for ", typeof(policy), "==========")

    sim_results = replicate_simulation(pomdp, policy, up, k=k, max_steps=max_steps)

    # Print results
    print_policy_results(string(typeof(policy)), sim_results)
    #end
end

# Displays results from simulation
function print_policy_results(policy_name, simulation_results)
    println("\n$(policy_name) Results:")
    println("rdisc mean: ", simulation_results[:rdisc_mean], ", stdev: ", simulation_results[:rdisc_std])
    println("edisc mean: ", simulation_results[:edisc_mean], ", stdev: ", simulation_results[:edisc_std])
    println("rtot mean: ", simulation_results[:rtot_mean], ", stdev: ", simulation_results[:rtot_std])
    println("**etot mean: ", simulation_results[:etot_mean], ", stdev: ", simulation_results[:etot_std])
    println("vt mean: ", simulation_results[:vt_mean], ", stdev: ", simulation_results[:vt_std])
    println("**vol_tot mean:", simulation_results[:vol_tot_mean], ", stdev: ", simulation_results[:vol_tot_std])
end


function replicate_simulation(pomdp, policy, up; k=100, max_steps=10, rng=MersenneTwister(1), randomized=true)
    rdisc_values = Float64[]
    edisc_values = Float64[]
    rtot_values = Float64[]
    etot_values = Float64[]
    vt_values = Float64[]
    vol_tot_values = Float64[]

    for i in 1:k
        if policy isa MCTSPlanner
            println(" replication: $i")
        end

        if randomized
            s0 = random_initial_state(pomdp)
            b0 = random_initial_belief(s0)
        else
            s0 = pomdp.init_state
            b0 = initialize_belief(up, s0)
        end

        #println("====new rep====")
        result = simulate_policy(pomdp, policy, up, b0, s0, max_steps=max_steps, rng=rng)
        push!(rdisc_values, result.rdisc)
        push!(edisc_values, result.edisc)
        push!(rtot_values, result.rtot)
        push!(etot_values, result.etot)
        push!(vt_values, result.vt)
        push!(vol_tot_values, result.vol_total)
    end

    return Dict(
        :rdisc_mean => mean(rdisc_values),
        :rdisc_std => std(rdisc_values),
        :edisc_mean => mean(edisc_values),
        :edisc_std => std(edisc_values),
        :rtot_mean => mean(rtot_values),
        :rtot_std => std(rtot_values),
        :etot_mean => mean(etot_values),
        :etot_std => std(etot_values),
        :vt_mean => mean(vt_values),
        :vt_std => std(vt_values),
        :vol_tot_mean => mean(vol_tot_values),
        :vol_tot_std => std(vol_tot_values)
    )
end


function simulate_policy(pomdp, policy, up, b0, s0; max_steps=10, rng=MersenneTwister(1))
    r_total = 0.0
    r_disc = 0.0
    e_total = 0.0
    e_disc = 0.0
    vol_total = 0.0
    t = 0
    d = 1.0

    b = deepcopy(b0)
    s = deepcopy(s0)
    while (!isterminal(pomdp, s) && t < max_steps)
        t += 1
        a = action(policy, b)
        (s, o, r) = gen(pomdp, s, a, rng)
        b = update(up, b, a, o)
        e = get_action_emission(pomdp, a)
        #println("action: $a, type: $(typeof(a)), emission: $e")
        r_total += r
        r_disc += r * d
        e_total += e
        e_disc += e * d
        if get_action_type(a) == "MINE"
            vol_total += 1
        end
        d *= discount(pomdp)
        #@show(t=t, s=s, a=a, r=r, o=o)
    end
    return (rdisc=r_disc, edisc=e_disc, rtot=r_total, etot=e_total, vt=s.Vₜ, vol_total=vol_total)
end

# Used to discretize observations -- not needed in current version
function compute_chunk_boundaries(quantile_vols::Vector{Float64})
    n = length(quantile_vols)
    @assert n > 0 "quantile_vols must not be empty"

    chunk_boundaries = Vector{Float64}(undef, n - 1)
    for i = 2:n
        chunk_boundaries[i-1] = (quantile_vols[i] + quantile_vols[i-1]) / 2
    end
    return chunk_boundaries
end

# Used to discretize observations -- not needed in current version
function compute_chunk_probs(chunk_boundaries::Vector{Float64}, site_dist::Normal)
    n = length(chunk_boundaries)
    @assert n > 0 "chunk_boundaries must not be empty"

    chunk_probs = Vector{Float64}(undef, n + 1)

    chunk_probs[1] = cdf(site_dist, chunk_boundaries[1])
    for i = 2:n
        chunk_probs[i] = cdf(site_dist, chunk_boundaries[i]) - cdf(site_dist, chunk_boundaries[i-1])
    end
    chunk_probs[n+1] = 1 - cdf(site_dist, chunk_boundaries[n])

    return chunk_probs
end

# Inputs an action and the pomdp, and outputs how much carbon will be emitted if that action is taken
function get_action_emission(P, a)
    action_type = get_action_type(a)
    action_number = get_site_number(a)

    # Subtract carbon emissions (if relevant)
    r3 = (action_type == "MINE") ? P.CO2_emissions[action_number] * -1 : 0
    return r3
end

#=
# Inputs an action, outputs the site number of that action
function get_site_number(a::Action)
    a = a.a
    len = length(a)
    if (a[1:4] == "MINE")
        return parse(Int64, a[5:len])
    else
        return parse(Int64, a[8:len])
    end
end
=#

# Inputs an action, outputs either MINE or EXPLORE as a string
function get_action_type(a::Action)
    a = a.a
    len = length(a)
    if (a[1:4] == "MINE")
        return "MINE"
    else
        return "EXPLORE"
    end
end

# I'm sure there's some builtin for this but I couldn't find it lol. Splices a string
function splice(begin_index, end_index, str)
    result = ""
    for i = begin_index:end_index
        result = result * str[i]
    end
    return result
end

# Inputs an action, outputs the site number of that action
#edited by Anna
function get_site_number(a::Action)
    a = a.a

    # Handle DONOTHING action
    if a == "DONOTHING"
        return 0  # Return 0 for DONOTHING
    end

    len = length(a)
    if (a[1:4] == "MINE")
        return parse(Int64, a[5:len])
    elseif (a[1:7] == "EXPLORE")
        return parse(Int64, a[8:len])
    else  # Assuming this is RESTORE
        return parse(Int64, a[8:len])
    end
end

# Inputs an action and belief, outputs whether or not we can explore at that site
function can_explore_here(a::Action, b::Any)
    action_type = get_action_type(a)
    site_number = get_site_number(a)

    if action_type == "MINE" || isa(b, ParticleFilters.ParticleCollection{State})
        return true
    end

    if isa(b, POMCPOW.StateBelief{POWNodeBelief{State,Action,Any,LiPOMDP}})
        b = convert_particle_collection_to_libelief(b)
    end

    return !b.have_mined[site_number]
end


function get_all_states(b::POMCPOW.StateBelief{POWNodeBelief{State,Action,Any,LiPOMDP}})
    dist = b.sr_belief.dist
    states = [x[1] for x in dist.items]
    states = [x.deposits for x in states]
    return states
end

function get_all_states(s::State)
    return [s.deposits]
end

# Used to see how much probability mass is below our min_n_units threshold (used to see if we can MINE at a desired site)
function compute_portion_below_threshold(P, b, idx::Int64)
    if isa(b, LiBelief)
        dist = b.deposit_dists[idx]
        portion_below_threshold = cdf(dist, P.min_n_units)
    elseif isa(b, ParticleCollection{State}) || isa(b, State)
        portion_below_threshold = 0.0
    else
        sampled_belief = get_all_states(b)
        n_rows = length(sampled_belief)

        num_below_threshold = sum(row[idx] < P.min_n_units for row in sampled_belief)
        portion_below_threshold = num_below_threshold / n_rows
    end
    return portion_below_threshold
end

# In: a particle collection Out: an LiBelief representation of that collection
function convert_particle_collection_to_libelief(part_collection::POMCPOW.StateBelief{POWNodeBelief{State,Action,Any,LiPOMDP}})
    items_vec = part_collection.sr_belief.dist.items # this is a vector of tuples (state, V). ignore the V value for now
    states_vec = [tup[1] for tup in items_vec] # vector of states
    deposits_vec = [state.deposits for state in states_vec] # vector of deposit vectors


    # Put all of the v1s, v2s, etc. into their own vectors so we can compute mean and std
    v1s = [deposits[1] for deposits in deposits_vec]
    v2s = [deposits[2] for deposits in deposits_vec]
    v3s = [deposits[3] for deposits in deposits_vec]
    v4s = [deposits[4] for deposits in deposits_vec]


    # Compute the estimated means from all the sample points
    μ1 = mean(v1s)
    μ2 = mean(v2s)
    μ3 = mean(v3s)
    μ4 = mean(v4s)


    # Compute the estimated standard deviations from all the sample points
    σ1 = std(v1s, corrected=true)
    σ2 = std(v2s, corrected=true)
    σ3 = std(v3s, corrected=true)
    σ4 = std(v4s, corrected=true)


    # Assert that all times are the same    
    # for state in states_vec
    #     if state.t != states_vec[1].t
    #         println("state.t : $(state.t), state_vec[1].t : $(states_vec[1].t)")
    #         println("state.Vₜ : $(state.Vₜ), state_vec[1].Vₜ : $(states_vec[1].Vₜ)")
    #     #println("state.t : $(state.t), state_vec[1].t : $(states_vec[1].t)")
    #     end
    # #    @assert(state.t == states_vec[1].t)
    # end
    #@assert(all([state.t == states_vec[1].t for state in states_vec]))


    # Create a new LiBelief with the mean, std of each Deposit, take the other fields (t, V_tot, have_mined) from the first particle
    return LiBelief([Normal(μ1, σ1), Normal(μ2, σ2), Normal(μ3, σ3), Normal(μ4, σ4)], states_vec[1].t, states_vec[1].Vₜ, states_vec[1].Iₜ, [mine for mine in states_vec[1].have_mined])
end

#Base functions ########################

# To make the struct iterable (potentially for value iteration?) Was experiencing errors
function Base.iterate(state::State, index=1)
    if index <= 5  # I should get rid of magic numbers later

        # If on a valid field index, get the field name and then the thing at that field
        field = fieldnames(State)[index]
        value = getfield(state, field)
        # Return value and the next index for iteration
        return (value, index + 1)
    else
        # If we've gone through all fields, return nothing to signify that we're done
        return nothing
    end
end

# Make a copy of the state
function Base.deepcopy(s::State)
    return State(deepcopy(s.deposits), s.t, s.Vₜ, s.Iₜ, deepcopy(s.have_mined))  # don't have to copy t and Vₜ cuz theyre immutable i think
end

# Input a belief and randomly produce a state from it 
function Base.rand(rng::AbstractRNG, b::LiBelief)
    deposit_samples = rand.(rng, b.deposit_dists)
    t = b.t
    V_tot = b.V_tot
    I_tot = b.I_tot
    have_mined = b.have_mined
    return State(deposit_samples, t, V_tot, I_tot, have_mined)
end

# Define == operator to use in the termination thing, just compares two states
Base.:(==)(s1::State, s2::State) = (s1.deposits == s2.deposits) && (s1.t == s2.t) && (s1.Vₜ == s2.Vₜ) && (s1.Iₜ == s2.Iₜ) && (s1.have_mined == s2.have_mined)

##Helper Functions Moved from LiPOMDP.jl
function random_initial_state(P::LiPOMDP, rng::AbstractRNG=Random.default_rng())
    # Randomize resources in each deposit site (assuming resources range between 0 to 10 for example)
    resources = [rand(rng, 2.0:1.0:6.0) for _ in 1:P.n_deposits]
    t = 0
    v = 0
    i = 0
    mined = fill(false, P.n_deposits)

    return State(resources, t, v, i, mined)
end

function random_initial_belief(s::State, rng::AbstractRNG=Random.default_rng())
    # Initialize belief to be a vector of 4 normal distributions, one for each deposit
    # Each normal distribution has mean equal to the amount of Li in that deposit, and
    # standard deviation equal to P.σ_obs
    std_range = collect(1.0:0.5:5.0)
    deposit_dists = [Normal(d, rand(rng, std_range)) for d in s.deposits]
    t = s.t
    V_tot = s.Vₜ
    I_tot = s.Iₜ
    have_mined = s.have_mined
    return LiBelief(deposit_dists, t, V_tot, I_tot, have_mined)
end

# kalman_step is used in the belief updater update function
function kalman_step(P::LiPOMDP, μ::Float64, σ::Float64, z::Float64)
    k = σ / (σ + P.σ_obs)  # Kalman gain
    μ_prime = μ + k * (z - μ)  # Estimate new mean
    σ_prime = (1 - k) * σ   # Estimate new uncertainty
    return μ_prime, σ_prime
end


function get_rewards(pomdp, hist)

    explore_actions = []
    explore_times = []
    invest_actions = []
    invest_times = []
    mine_actions = []
    mine_times = []
    decommision_actions = []
    decommision_times = []

    r1 = []
    r2_domestic = []
    r2_imported = []
    r3 = []
    r4 = []
    r5 = []

    for (_, step) in enumerate(hist)
        a = step.a
        s = step.s
        a_type = get_action_type(a)
        dnum = get_site_number(a)

        if a_type == "EXPLORE"
            if !s.have_mined[dnum]
                push!(explore_actions, dnum)
                push!(explore_times, step.t)
            end
        else
            if !s.have_mined[dnum]
                push!(invest_actions, dnum)
                push!(invest_times, step.t)
            end
        end

        for i in 1:4
            if s.have_mined[i]
                if s.deposits[i] > 0
                    push!(mine_actions, i)
                    push!(mine_times, step.t)
                else
                    push!(decommision_actions, i)
                    push!(decommision_times, step.t)
                end
            end
        end

        r2_ = compute_r2(pomdp, s, a)
        push!(r1, compute_r1(pomdp, s, a))
        push!(r2_domestic, r2_.domestic)
        push!(r2_imported, r2_.imported)
        push!(r3, compute_r3(pomdp, s, a))
        push!(r4, compute_r4(pomdp, s, a))
        push!(r5, compute_r5(pomdp, s, a) * 0.25)
    end

    return (
        a_explore=explore_actions, t_explore=explore_times,
        a_mine=mine_actions, t_mine=mine_times,
        a_invest=invest_actions, t_invest=invest_times,
        a_decommision=decommision_actions, t_decommision=decommision_times,
        r1=r1, r2=(domestic=r2_domestic, imported=r2_imported),
        r3=r3, r4=r4, r5=r5)
end

function plot_results(pomdp::LiPOMDP, df::NamedTuple; ylims=(-200, 200))
    T = pomdp.time_horizon
    #plot 1: actions vs time
    p0 = scatter(df.t_explore, df.a_explore, label="EXPLORE", markersize=10, xticks=1:T)
    scatter!(df.t_invest, df.a_invest, label="INVEST", markersize=10)
    scatter!(df.t_decommision, df.a_decommision, label="DECOMMISSION/REHAB", markersize=10)
    scatter!(
        df.t_mine, df.a_mine,
        label="MINE", markersize=10,
        xticks=0:1:T,
        yticks=(1:4, ["1 (SilverPeak, USA)", "2 (ThackerPass, USA)", "3 (Greenbushes, AUS)", "4 (Pilgangoora, AUS)"]),
        ylims=(0.5, 4.5),
        ylabel="Deposit Site", xlabel="Time",
        title="Actions vs Time",
        legend=:outerbottomright)
    vline!([pomdp.t_goal], label="Time Delay Goal", color=:red, linestyle=:dash)

    #set xticks to be integers
    p1 = bar(df.r1, label="r1", xlabel="Time", ylabel="\$ Value (in Millions)", title="Domestic Mining (Penalty)", legend=false, xticks=0:5:T, ylims=ylims)
    p5 = bar(df.r5, label="r5", xlabel="Time", ylabel="\$ Value (in Millions)", title="Cash Flow", legend=false, ylims=ylims, xticks=0:5:T)
    p4 = bar(df.r4, label="r4", xlabel="Time", ylabel="\$ Value (in Millions)", title="Unmet Demand (Penalty)", legend=false, ylims=ylims = ylims, xticks=0:5:T)
    p2 = bar(df.r2.domestic + df.r2.imported, label="r2", xlabel="Time", ylabel="Thousand Metric Tons", title="LCE Volume Mined", legend=false, xticks=0:5:T)
    p3 = bar(df.r3, label="r3", xlabel="Time", ylabel="Units", title="CO2 Emission", legend=false, ylims=(-30, 0), xticks=0:5:T)

    prow1 = plot(p5, p1, p4, layout=(1, 3), margin=3mm)
    prow2 = plot(p2, p3, layout=(1, 2))
    return (action=p0, econ=prow1, other=prow2)
end