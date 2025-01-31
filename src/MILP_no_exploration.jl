function create_lithium_mining_model(pomdp::LiPOMDP, s::State, T::UnitRange{Int}, I::UnitRange{Int}, bigM::Int=999999)
    m = Model(GLPK.Optimizer)
    
    # === VARIABLES === #
    # Binary decision variables
    @variable(m, x[T, I], Bin)      # Exploration decision
    @variable(m, y[T, I], Bin)      # Mine construction
    @variable(m, z[T, I], Bin)      # Mine restoration
    @variable(m, w[T, I], Bin)      # Mine operation
    @variable(m, v[T, I], Bin)      # Domestic mining indicator
    @variable(m, u[T], Bin)         # Unmet demand indicator
    
    # Continuous variables
    @variable(m, u_qty[T] >= 0)     # Unmet demand quantity
    @variable(m, w_sold_qty[T] >= 0)     # Lithium sold quantity
    @variable(m, w_processed_qty[T] >= 0)     # Lithium processed quantity
    @variable(m, inventory[T] >= 0)  # Lithium inventory
    @variable(m, li_remaining[T, I] >= 0)  # Lithium remaining in each deposit
    @variable(m, mine_vol[T,I] >= 0)  # Lithium mined in each deposit
    @variable(m, processed_vol[T,I] >= 0)  # Volume after waste removal

    # === OBJECTIVE FUNCTION === #
    γt = [pomdp.γ^(t-1) for t in T]  # Discount factors
    
    # Penalty components
    domestic_mining = sum(pomdp.pd * γt[t] * v[t, i] for t in T for i in I)
    co2_emissions = sum(pomdp.e[i] * w[t, i] for t in T for i in I)
    unmet_demand = sum(u_qty[t] for t in T)
    
    # Financial components
    capex = sum(pomdp.cb * γt[t] * y[t, i] for t in T for i in I)
    opex = sum(pomdp.co * γt[t] * w[t, i] for t in T for i in I)
    revenue = sum(pomdp.p * w_sold_qty[t] for t in T)
    profit = revenue - capex - opex

    objectives = [-domestic_mining, -co2_emissions, -unmet_demand, profit]
    @objective(m, Max, sum(objectives[i] * pomdp.w[i] for i in 1:pomdp.num_objectives))

    # === CONSTRAINTS === #
    
    # Initial conditions
    for i in I
        @constraint(m, li_remaining[1, i] == pomdp.v0[i])
    end

    # Mining Operation Constraints
    for t in T
        for i in I
            # Mining volume and processing
            @constraint(m, mine_vol[t,i] == (w[t,i] + z[t, i]) * pomdp.mine_rate[i])
            @constraint(m, processed_vol[t,i] == mine_vol[t,i] - mean(pomdp.ψ[i]))
            
            # Resource depletion tracking
            if t > 1
                @constraint(m, mine_vol[t,i] <= li_remaining[t-1,i])
                @constraint(m, li_remaining[t, i] == li_remaining[t-1, i] - mine_vol[t,i])
                @constraint(m, w[t,i] <= li_remaining[t-1,i]/pomdp.mine_rate[i])
            end
            
            # Operation logic
            @constraint(m, w[t,i] <= sum(y[τ,i] for τ in 1:t))  # Require construction
            @constraint(m, w[t,i] >= sum(y[τ,i] for τ in 1:t) - sum(z[τ,i] for τ in 1:t))  # Must operate after construction until restoration
            @constraint(m, w[t,i] + sum(z[τ,i] for τ in 1:t) <= 1)  # Stop after restoration
            @constraint(m, z[t,i] <= sum(y[τ,i] for τ in 1:t))
            @constraint(m, w[t,i] <= li_remaining[t,i])  # Can't operate without lithium
            
            # Domestic mining constraints
            if i in pomdp.Jd && t <= pomdp.td
                @constraint(m, v[t,i] == y[t,i])
            else
                @constraint(m, v[t,i] == 0)
            end
        end

        # Processing and sales constraints
        @constraint(m, w_processed_qty[t] == sum(processed_vol[t,i] * pomdp.ρ for i in I))
        
        # Inventory balance
        if t == 1
            @constraint(m, inventory[t] == w_processed_qty[t] - w_sold_qty[t])
            @constraint(m, w_sold_qty[t] <= w_processed_qty[t])
        else
            @constraint(m, inventory[t] == inventory[t-1] + w_processed_qty[t] - w_sold_qty[t])
            @constraint(m, w_sold_qty[t] <= w_processed_qty[t] + inventory[t-1])
        end

        # Demand constraints
        @constraint(m, pomdp.d[t] - w_sold_qty[t] <= u[t] * bigM)
        @constraint(m, u_qty[t] >= pomdp.d[t] - w_sold_qty[t] - (1 - u[t]) * bigM)
        @constraint(m, u_qty[t] <= pomdp.d[t] - w_sold_qty[t] + (1 - u[t]) * bigM)
        @constraint(m, w_sold_qty[t] <= pomdp.d[t])
    end

    return m
end

function print_optimization_results(model, pomdp, T, I)
    # Helper function to get and round values
    function get_rounded_values(model, var_name)
        return round.(JuMP.value.(model[var_name]), digits=2)
    end

    # Helper function to print section header
    function print_section_header(title)
        println("\n=== $title ===")
    end

    # Helper function to print variable values
    function print_variable(title, values)
        println("\n$title:")
        display(values)
    end

    # Calculate financial metrics
    function calculate_financials(model, pomdp, T, I)
        total_capex = sum(pomdp.cb * JuMP.value.(model[:y][t,i]) for t in T for i in I)
        total_opex = sum(pomdp.co * JuMP.value.(model[:w][t,i]) for t in T for i in I)
        total_revenue = sum(pomdp.p * JuMP.value.(model[:w_sold_qty][t]) for t in T)
        return total_revenue, total_capex, total_opex
    end

    # Calculate penalty metrics
    function calculate_penalties(model, pomdp, T, I)
        return (
            domestic = sum(pomdp.pd * JuMP.value.(model[:v][t,i]) for t in T for i in I),
            emissions = sum(pomdp.e[i] * JuMP.value.(model[:w][t,i]) for t in T for i in I),
            unmet_demand = sum(JuMP.value.(model[:u_qty][t]) for t in T)
        )
    end

    # 1. Mining Decisions
    print_section_header("Mining Decisions by Time Period and Deposit")
    for (var, title) in [(:x, "Exploration Decisions"), 
                        (:y, "Mine Construction"), 
                        (:w, "Mine Operation"), 
                        (:z, "Mine Restoration")]
        print_variable(title, get_rounded_values(model, var))
    end

    # 2. Production and Sales
    print_section_header("Production and Sales")
    for (var, title) in [(:w_sold_qty, "Lithium Sold by Period"),
                        (:w_processed_qty, "Lithium Extracted by Period"),
                        (:inventory, "Inventory Levels")]
        print_variable(title, get_rounded_values(model, var))
    end

    # 3. Demand Satisfaction
    print_section_header("Demand Satisfaction")
    for (var, title) in [(:u, "Unmet Demand Periods"),
                        (:u_qty, "Unmet Demand Quantities")]
        print_variable(title, get_rounded_values(model, var))
    end

    # 4. Lithium Reserves
    print_section_header("Li Reserves")
    for (var, title) in [(:li_remaining, "Remaining Reserves"),
                        (:mine_vol, "Mining Volume")]
        print_variable(title, get_rounded_values(model, var))
    end

    # 5. Financial and Penalty Summary
    print_section_header("Summary")
    println("\nOptimization Status: ", termination_status(model))
    println("Objective Value: ", round(objective_value(model), digits=2))
    
    # Calculate and display metrics
    revenue, capex, opex = calculate_financials(model, pomdp, T, I)
    penalties = calculate_penalties(model, pomdp, T, I)
    
    print_section_header("Financial Summary")
    println("Total Revenue: ", round(revenue, digits=2))
    println("Total CAPEX: ", round(capex, digits=2))
    println("Total OPEX: ", round(opex, digits=2))
    println("Net Profit: ", round(revenue - capex - opex, digits=2))

    print_section_header("Penalty Summary")
    println("Domestic Mining Penalty: ", round(penalties.domestic, digits=2))
    println("CO2 Emissions Penalty: ", round(penalties.emissions, digits=2))
    println("Unmet Demand Penalty: ", round(penalties.unmet_demand, digits=2))

    println("\nTime Horizon: T = ", T)
end

function solve_lithium_mining_model(pomdp::LiPOMDP, s::State)
    T = 1:pomdp.T
    I = 1:pomdp.n
    model = create_lithium_mining_model(pomdp, s, T, I)
    optimize!(model)
    print_optimization_results(model, pomdp, T, I)
    return model
end


function format_optimization_results(model, pomdp, T, I)
    # Helper function to get values
    function get_values(model, var_name)
        return JuMP.value.(model[var_name])
    end
    
    # Get all decision variables
    y_val = get_values(model, :y)  # invest/construct
    w_val = get_values(model, :w)  # mine operation
    z_val = get_values(model, :z)  # restore
    x_val = get_values(model, :x)  # explore
    u_qty_val = get_values(model, :u_qty)
    w_sold_qty_val = get_values(model, :w_sold_qty)

    # Initialize arrays for actions and times
    explore_actions = Int[]
    explore_times = Float64[]
    invest_actions = Int[]
    invest_times = Float64[]
    restore_actions = Int[]
    restore_times = Float64[]
    mine_on = Int[]
    mine_on_times = Float64[]
    mine_off = Int[]
    mine_off_times = Float64[]

    # Collect actions and times
    for t in T
        for i in I
            # Exploration
            if x_val[t,i] > 0.5  # Using 0.5 threshold for binary variables
                push!(explore_actions, i)
                push!(explore_times, float(t))
            end
            
            # Investment/Construction
            if y_val[t,i] > 0.5
                push!(invest_actions, i)
                push!(invest_times, float(t))
            end
            
            # Restoration
            if z_val[t,i] > 0.5
                push!(restore_actions, i)
                push!(restore_times, float(t))
            end
            
            # Mine operation status
            if w_val[t,i] > 0.5
                push!(mine_on, i)
                push!(mine_on_times, float(t))
            else
                push!(mine_off, i)
                push!(mine_off_times, float(t))
            end
        end
    end

    # Calculate rewards
    r1 = Float64[] # Domestic mining penalty
    r2 = Float64[] # CO2 emissions
    r3 = Float64[] # Unmet demand penalty
    r4 = Float64[] # Cash flow

    for t in T
        # Domestic mining penalty
        push!(r1, -sum(pomdp.pd * get_values(model, :v)[t,i] for i in I))
        
        # CO2 emissions
        push!(r2, -sum(pomdp.e[i] * w_val[t,i] for i in I))
        
        # Unmet demand penalty
        push!(r3, -u_qty_val[t])
        
        # Cash flow: revenue - capex - opex
        revenue = pomdp.p * w_sold_qty_val[t]
        capex = sum(pomdp.cb * y_val[t,i] for i in I)
        opex = sum(pomdp.co * w_val[t,i] for i in I)
        push!(r4, revenue - capex - opex)
    end

    return (
        a_explore=explore_actions, t_explore=explore_times, 
        a_invest=invest_actions, t_invest=invest_times,
        a_restore=restore_actions, t_restore=restore_times,
        mine_on=mine_on, t_mine_on=mine_on_times,
        mine_off=mine_off, t_mine_off=mine_off_times,
        r1=r1, r2=r2, r3=r3, r4=r4)
end