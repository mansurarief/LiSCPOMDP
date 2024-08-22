#=
Original Authors: Yasmine Alonso, Mansur Arief, Anthony Corso, Jef Caers, and Mykel Kochenderfer
Extended by: CJ Oshiro, Mansur Arief, Mykel Kochenderfer
----------------
=#

@with_kw mutable struct State
    v::Vector{Float64} # [v₁, v₂, v₃, v₄]
    t::Float64 = 1  # current time
    Vₜ::Float64 = 0  # current amt of Li mined domestically up to time t
    Iₜ::Float64 = 0. # current amt of Li imported up to time t
    m::Vector{Bool}  # Boolean value to represent whether or not we have taken a mine action
end


# All potential actions
@enum Action DONOTHING MINE1 MINE2 MINE3 MINE4 EXPLORE1 EXPLORE2 EXPLORE3 EXPLORE4 RESTORE1 RESTORE2 RESTORE3 RESTORE4


@with_kw mutable struct Observation
    v::Vector{Float64} # [v₁, v₂, v₃, v₄]
    E::Vector{Float64} # [E₁, E₂, E₃, E₄]
end

@with_kw mutable struct LiPOMDP <: POMDP{State, Action, Observation} 
    td::Int64                       # time goal, want to wait 10 years before mining domestically
    σo::Float64                     # Standard deviation of the observation noise
    γ::Float64                      # discounted reward
    T::Int64                        # time horizon
    d::Vector{Float64}              # demand for each time step
    n::Int64                        # number of deposits
    ϕ::Vector{Distribution}         # distribution of LCE extracted from each deposit
    ψ::Vector{Distribution}         # distribution of LCE lost during transport due to disruptions from each foreign deposit
    bin_edges::Vector{Float64}      # Used to discretize observations
    cdf_threshold::Float64          # threshold allowing us to mine or not
    min_n_units::Int64              # minimum number of units required to mine. So long as cdf_threshold portion of the probability
    num_objectives::Int64           # number of objectives
    ΔV::Float64                     # increment of volume mined in the state space
    Δdeposit::Float64               # increment of deposit in the state space
    V_deposit_min::Float64          # min and max amount per singular deposit
    V_deposit_max::Float64          # min and max amount per singular deposit
    obj_weights::Vector{Float64}    # how we want to weight each component of the reward
    e::Vector{Float64}              #[C₁, C₂, C₃, C₄] amount of CO2 each site emits
    v0::Vector{Float64}             # initial volume of lithium in each mine
    null_state::State               # null state
    init_state::State               # initial state
    Jd::Vector{Int64}               # mines that are domestic
    Ji::Vector{Int64}               # mines that are foreign
    ce::Float64                     # exploration cost
    cb::Float64                     # building mining cost
    cr::Float64                     # restoration cost
    ct::Vector{Float64}             # transportation cost from each mine to the processing plant
    cp::Float64                     # processing cost
    co::Float64                     # fixed cost of operating a mine
    p::Float64                      # price of lithium
    ρ::Float64                      # extraction factor of lithium (% of lithium in the ore)
    pd::Float64                     # domestic mining penalty, if done before t_goal    
    rng::AbstractRNG                # random number generator
end

function initialize_lipomdp(;
    td=10,  
    σo=0.2,
    γ=0.95,
    T=30,
    d=[60.0, 70.0, 80.0, 90.0, 100.0, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600],
    n=4,
    ϕ=[Normal(100, 10), Normal(100, 10), Normal(100, 10), Normal(100, 10)],
    ψ=[Normal(0, 0.1), Normal(0, 0.1), Normal(10, 5), Normal(10, 5)],    
    bin_edges=[0.25, 0.5, 0.75],
    cdf_threshold=0.1,
    min_n_units=3,
    num_objectives=5,
    ΔV=10.0,
    Δdeposit=10.0,
    V_deposit_min=0.0,
    V_deposit_max=10.0,
    obj_weights=[0.25, 0.25, 0.25, 0.25, 0.25],
    e=[3, 4, 6, 7],
    v0=[4100., 1800., 5500., 2500.],
    null_state=State([-1, -1, -1, -1], -1, -1, -1, [true, true, true, true]),
    init_state=State(v0, 1, 0.0, 0.0, [false, false, false, false]), 
    Jd=[1, 2],
    Ji=[3, 4],
    ce=50.0, #in millions USD
    cb=500.0, #in millions USD
    cr=100.0, #in millions USD
    ct=[0.0000001, 0.0000001, 0.005, 0.005], #in millions USD, i.e. 2k, 2k, 5k, 5k per ton
    cp=0.003, #3k per ton
    co=5.0, #5M per mine
    p=0.03, #20k per ton
    ρ=0.7, #70% of lithium in the 0.1 ore
    pd=100.0, #USD 100M domestic mining penalty 
    rng=MersenneTwister(1)
    )
    return LiPOMDP(
        td=td,
        σo=σo,
        γ=γ,
        T=T,
        d=d,
        n=n,
        ϕ=ϕ,
        ψ=ψ,
        bin_edges=bin_edges,
        cdf_threshold=cdf_threshold,
        min_n_units=min_n_units,
        num_objectives=num_objectives,
        ΔV=ΔV,
        Δdeposit=Δdeposit,
        V_deposit_min=V_deposit_min,
        V_deposit_max=V_deposit_max,
        obj_weights=obj_weights,
        e=e,
        v0=v0,
        null_state=null_state,
        init_state=init_state,
        Jd=Jd,
        Ji=Ji,
        ce=ce,
        cb=cb,
        cr=cr,
        ct=ct,
        cp=cp,
        co=co,
        p=p,
        ρ=ρ,
        pd=pd,
        rng=rng
    )
end


struct LiBelief{T<:UnivariateDistribution} 
    v_dists::Vector{T}
    t::Int64  # current time
    Vₜ::Float64  # current amt of Li mined domestically up to time t
    Iₜ::Float64 # current amt of Li imported up to time t
    m::Vector{Bool}  # Boolean value to represent whether or not we have taken a mine action    
end

struct LiBeliefUpdater <: Updater
    P::LiPOMDP
end

POMDPs.support(b::LiBelief) = [rand(b) for _ in 1:100]
POMDPs.pdf(ps::Distributions.ProductDistribution, v::Vector{Float64}) = prod(pdf(ps.dists[j], v[j]) for j in 1:length(ps))
POMDPs.pdf(ps::Distributions.ProductDistribution, o::Observation) = prod(pdf(ps.dists[j], o.v[j]) for j in 1:length(ps))
