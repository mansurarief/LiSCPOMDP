@with_kw mutable struct State
    v::Vector{Float64} # [v₁, v₂, v₃, v₄]
    t::Float64 = 1  # current time
    Vₜ::Float64 = 0  # current amt of Li mined domestically up to time t
    Iₜ::Float64 = 0. # current amt of Li imported up to time t
    m::Vector{Bool}  # Boolean value to represent whether or not we have taken a mine action
end

@enum Action DONOTHING MINE1 MINE2 MINE3 MINE4 EXPLORE1 EXPLORE2 EXPLORE3 EXPLORE4 RESTORE1 RESTORE2 RESTORE3 RESTORE4

@with_kw mutable struct Observation
    v::Vector{Float64} # [v₁, v₂, v₃, v₄]
    E::Vector{Float64} # [E₁, E₂, E₃, E₄] #E is the amount of lithium extracted from foreign mines
    Z::Vector{Float64} # [Z₁, Z₂, Z₃, Z₄] #Z is the amount of lithium arrived at the processing plant (E - L(oss))
end

#TODO: check if a;l parameters are correct with the paper
@with_kw mutable struct LiPOMDP <: POMDP{State, Action, Observation} 
    td::Int64=10                       # time goal, want to wait 10 years before mining domestically
    σo::Float64=0.2                    # Standard deviation of the observation noise
    γ::Float64=0.95                    # discounted reward
    T::Int64=30                        # time horizon
    d::Vector{Float64}=[               # demand for each time step
        60.0, 70.0, 80.0, 90.0, 100.0, 
        120.0, 140.0, 160.0, 180.0, 200.0, 
        220.0, 240.0, 260.0, 280.0, 300.0, 
        320.0, 340.0, 360.0, 380.0, 400.0, 
        420.0, 440.0, 460.0, 480.0, 500.0, 
        520.0, 540.0, 560.0, 580.0, 600.0]              
    n::Int64=4                          # number of deposits
    ϕ::Vector{Distributions.Normal{Float64}}=[           # distribution of LCE in each deposit
        Normal(150, 10), Normal(150, 10), 
        Normal(150, 10), Normal(150, 10)]         
    ψ::Vector{Distributions.Normal{Float64}}=[           # distribution of LCE lost during transport due to disruptions from each foreign deposit
        Normal(0, 0.001), Normal(0, 0.001), 
        Normal(10, 5), Normal(10, 5)]         
    bin_edges::Vector{Float64}=[0.25, 0.5, 0.75] # Used to discretize observations
    cdf_threshold::Float64=0.1          # threshold allowing us to mine or not
    min_n_units::Int64=3                # minimum number of units required to mine. So long as cdf_threshold portion of the probability
    num_objectives::Int64=4             # number of objectives
    ΔV::Float64=10.0                    # increment of volume mined in the state space
    Δdeposit::Float64=10.0              # increment of deposit in the state space
    V_deposit_min::Float64=0.0          # min and max amount per singular deposit
    V_deposit_max::Float64=10.0         # min and max amount per singular deposit
    w::Vector{Float64}=[0.1, 1.0, 0.1, 0.05] # how we want to weight each component of the reward 
    e::Vector{Float64}=[3, 4, 6, 7]     #[C₁, C₂, C₃, C₄] amount of CO2 each site emits
    v0::Vector{Float64}=[41000., 18000., 55000., 25000.] # initial volume of lithium in each mine
    null_state::State=State([-1, -1, -1, -1], -1, -1, -1, [true, true, true, true])
    init_state::State=State(v0, 1, 0.0, 0.0, [false, false, false, false])
    Jd::Vector{Int64}=[1, 2]            # mines that are domestic
    Jf::Vector{Int64}=[3, 4]            # mines that are foreign
    ce::Float64=50.0                    # exploration cost
    cb::Float64=500.0                   # building mining cost
    cr::Float64=100.0                   # restoration cost
    ct::Vector{Float64}=[0.002, 0.002, 0.005, 0.005] # transportation cost from each mine to the processing plant
    cp::Float64=0.003                   # processing cost
    co::Float64=5.0                     # fixed cost of operating a mine #TODO: add co to paper
    p::Float64=0.03                     # price of lithium #TODO: add p to paper
    ρ::Float64=0.08                      # extraction factor of lithium (% of lithium in the ore)
    pd::Float64=100.0                   # domestic mining penalty, if done before t_goal
    rng::AbstractRNG=MersenneTwister(1) # random number generator
    disc_points::Vector{Float64}=collect(0.:1000.:10000.) # discretization points for observations
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

POMDPs.support(b::LiBelief) = [rand(b) for _ in 1:100] #TODO: fix this support function
POMDPs.pdf(ps::Distributions.ProductDistribution, v::Vector{Float64}) = prod(pdf(ps.dists[j], v[j]) for j in 1:length(ps))
POMDPs.pdf(ps::Distributions.ProductDistribution, o::Observation) = prod(pdf(ps.dists[j], o.v[j]) for j in 1:length(ps))
