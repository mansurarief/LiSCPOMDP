#=
Original Authors: Yasmine Alonso, Mansur Arief, Anthony Corso, Jef Caers, and Mykel Kochenderfer
Extended by: CJ Oshiro, Mansur Arief, Mykel Kochenderfer
----------------
=#

@with_kw mutable struct State
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
    t::Float64 = 1  # current time
    Vₜ::Float64 = 0  # current amt of Li mined domestically up to time t
    Iₜ::Float64 = 0. # current amt of Li imported up to time t
    have_mined::Vector{Bool}  # Boolean value to represent whether or not we have taken a mine action
end

@with_kw mutable struct Action
    a::String
end


@with_kw mutable struct Observation #TODO: implement this as part of your pomdp
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
end

@with_kw mutable struct LiPOMDP <: POMDP{State, Action, Any} 
    t_goal::Int64  #time goal, want to wait 10 years before mining domestically
    σ_obs::Float64 # Standard deviation of the observation noise
    Vₜ_goal::Float64  #Volume goal
    γ::Float64 #discounted reward
    time_horizon::Int64 #time horizon
    demands::Vector{Float64}
    n_deposits::Int64
    mine_output::Float64
    bin_edges::Vector{Float64}  # Used to discretize observations
    cdf_threshold::Float64  # threshold allowing us to mine or not
    min_n_units::Int64  # minimum number of units required to mine. So long as cdf_threshold portion of the probability
    num_objectives::Int64
    ΔV::Float64  # increment of volume mined in the state space
    Δdeposit::Float64  # increment of deposit in the state space
    V_deposit_min::Float64 #min and max amount per singular deposit
    V_deposit_max::Float64
    obj_weights::Vector{Float64}  # how we want to weight each component of the reward
    CO2_emissions::Vector{Float64}  #[C₁, C₂, C₃, C₄] amount of CO2 each site emits
    null_state::State
    init_state::State
    site_to_dist::Dict
    stochastic_price::Bool
    alpha::Float64 # Parameter to control tradeoff between emissions and volume
    compute_tradeoff::Bool
end

function initialize_lipomdp(;
    t_goal=10, 
    σ_obs=0.1,
    Vₜ_goal=8, 
    γ=0.98,
    time_horizon=30,
    demands=[2.0, 4.0, 3.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0],
    n_deposits=4,
    bin_edges=[0.25, 0.5, 0.75],
    cdf_threshold=0.1,
    min_n_units=3,
    mine_output=2.0,
    num_objectives=5,
    ΔV=1.0,
    Δdeposit=1.0,
    V_deposit_min=0.0,
    V_deposit_max=10.0,
    obj_weights=[0.25, 0.25, 0.25, 0.25, 0.25],
    CO2_emissions=[5, 7, 2, 5],
    null_state=State([-1, -1, -1, -1], -1, -1, -1, [true, true, true, true]),
    init_state=State([16.0, 60.0, 60.0, 50.0], 1, 0.0, 0.0, [false, false, false, false]), # SilverPeak and ThackerPass are domestic, Greenbushes and Pilgangoora are foreign #TODO; find some reference
    site_to_dist=Dict(1=>Normal(50,3), 2=>Normal(70,10), 3=>Normal(52,5), 4=>Normal(60,4)),
    stochastic_price=false,
    alpha=1, # Default reward
    compute_tradeoff=false
    )
    return LiPOMDP(
        t_goal=t_goal, 
        σ_obs=σ_obs, 
        Vₜ_goal=Vₜ_goal,
        γ=γ,
        time_horizon=time_horizon,
        demands=demands,
        n_deposits=n_deposits,
        bin_edges=bin_edges,
        cdf_threshold=cdf_threshold,
        min_n_units=min_n_units,
        num_objectives=num_objectives,
        mine_output=mine_output,
        ΔV=ΔV,
        Δdeposit=Δdeposit,
        V_deposit_min=V_deposit_min,
        V_deposit_max=V_deposit_max,
        obj_weights=obj_weights,
        CO2_emissions=CO2_emissions,
        null_state=null_state,
        init_state=init_state,
        site_to_dist=site_to_dist,
        stochastic_price=stochastic_price,
        alpha=alpha,
        compute_tradeoff=compute_tradeoff
    )
end


struct LiBelief{T<:UnivariateDistribution} 
    deposit_dists::Vector{T}
    t::Float64
    V_tot::Float64
    I_tot::Float64
    have_mined::Vector{Bool} 
end

struct LiBeliefUpdater <: Updater
    P::LiPOMDP
end

function POMDPTools.ModelTools.pdf(d::LiBelief{Normal{Float64}}, s)
    1.0/length(d.deposit_dists)
 end

# Allows user to change mine parameters of pomdp. User must specify init_state.
#mine_params(pomdp, n_deposits, deposits vector, objective weights vector, emissions vector)
function mine_params(
    P::LiPOMDP,
    n_deposits::Int64,
    init_mine_vols::Vector{Float64},
    obj_weights::Vector{Float64} = zeros(n_deposits),
    CO2_emissions::Vector{Int64} = zeros(Int64, n_deposits)
 )
    P.n_deposits = n_deposits
 
    if sum(obj_weights) == 0 
        P.obj_weights = fill((1/n_deposits, n_deposits))
    else
        P.obj_weights = obj_weights
    end

    if sum(CO2_emissions) == 0
        CO2_emissions = [rand(2:9) for _ in 1:length(CO2_emissions)]
    else   
        P.CO2_emissions = CO2_emissions
    end
 
    P.null_state = State(fill(-1, n_deposits), -1, -1, -1, fill(false, n_deposits))
    init_state = State(init_mine_vols, 1, 0.0, 0.0, fill(0, n_deposits))
    P.init_state = init_state

end