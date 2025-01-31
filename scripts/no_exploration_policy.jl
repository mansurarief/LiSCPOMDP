using POMDPs
using LiPOMDPs
using Random
using JuMP
using GLPK

pomdp = LiPOMDP(
    p=1.0, ρ=0.6, 
    T=31, w=[0.1, 3.2, 0.1, 0.2], 
    a=[-20, -20, -20, -30], 
    mine_rate=[300, 300, 100, 100])
    
up = LiBeliefUpdater(pomdp)
b = initialize_belief(up)
rng = MersenneTwister(2024)
s = rand(rng, b)
s0 = State(v=[mean(bi) for bi in b.v_dists], t=1.0, m=[0 for i in 1:pomdp.n])


# Define the optimization model to o`ptimize when to explore, mine, and restore the deposits
# as a MILP problem, assuming s is known (though might not be the same as the true state s0)

## sets
T = 1:pomdp.T-1
Td = 1:pomdp.td
I = 1:pomdp.n
JD = pomdp.Jd # domestic deposits
JF = pomdp.Jf # foreign deposits`
bigM = 999999

m = Model(GLPK.Optimizer)

## decision variables
@variable(m, x[T, I], Bin) # explore or not at time t in deposit i
@variable(m, y[T, I], Bin) # build a mine or not at time t in deposit i
@variable(m, z[T, I], Bin) # restore or not at time t in deposit i
@variable(m, u[T], Bin) # demand unmet at time t
@variable(m, u_qty[T] >=0) # qty of unmet demand
@variable(m, v[T, I], Bin) # domestic mining penalty at time t in deposit i
@variable(m, w[T, I], Bin) # operate mine at time t in deposit i
@variable(m, w_qty[T], Bin) # qty of lithium sold at time t

## objective function: -domestic mining penalty - co2 emissions penalty - unmet demand penalty + cashflow

## domestic mining: if mine is built before pomdp.td, then domestic mining can only be one
γt = [pomdp.γ^(t-1) for t in T]
domestic_mining = sum(pomdp.pd * γt[t] * v[t, i] for t in T for i in I)

## co2 emissions: if mine i is operated at time t, then add pomdp.e[i] each year t
co2_emissions = sum(pomdp.e[i] * w[t, i] for t in T for i in I)

## unmet demand: first compute the expected processed lithium. If demand is not met, then add the unmet demand penalty
unmet_demand = sum(u_qty[t] for t in T)

## cashflow: capex + opex - revenue
capex = sum(pomdp.cb * γt[t] * y[t, i] for t in T for i in I);
opex = sum(pomdp.co * γt[t] * w[t, i] for t in T for i in I);
revenue = sum(pomdp.p * w_qty[t] for t in T)
profit = revenue - capex - opex

preprocessed_amount = [pomdp.mine_rate[i] - mean(pomdp.ψ[i]) for i in I]
extracted_li = [sum(preprocessed_amount[i] * w[t, i] * pomdp.ρ for i in I) for t in T ]

objectives = [-domestic_mining, -co2_emissions, -unmet_demand, profit]

objective_fn = sum(objectives[i] * pomdp.w[i] for i in 1:pomdp.num_objectives)

##objective function
@objective(m, Max, objective_fn)

## constraints


for t in T
    for i in I
        # Exploration can only happen before mine construction
        @constraint(m, x[t,i] + sum(y[τ,i] for τ in 1:t) <= 1)

        # domestic mining can only be one if domestic mine is built before pomdp.td
        if i in pomdp.Jd
            if t <= pomdp.td
                @constraint(m, v[t, i] == y[t, i])
            else
                @constraint(m, v[t, i] == 0)
            end
        end
    end

    #if extracted lithium is less than demand at year t, then u[t] must be one
    @constraint(m, pomdp.d[t] - extracted_li[t] <= u[t] * bigM)

    #if u[t] is one, then u_qty[t] must be the difference between demand and extracted lithium
    #otherwise, u_qty[t] must be zero
    @constraint(m, u_qty[t] >= pomdp.d[t] - extracted_li[t] - (1 - u[t]) * bigM)
    @constraint(m, u_qty[t] <= pomdp.d[t] - extracted_li[t] + (1 - u[t]) * bigM)

    # the amount of lithium sold at time t must be less than the amount of lithium extracted and demand 
    @constraint(m, w_qty[t] <= extracted_li[t])
    @constraint(m, w_qty[t] <= pomdp.d[t])

end

JuMP.optimize!(m)

