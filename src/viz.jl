
function get_rewards(pomdp, hist)
    explore_actions = []
    explore_times = []
    invest_actions = []
    invest_times = []
    restore_actions = []
    restore_times = []
    mine_on = []
    mine_on_times = []

    r1 = []
    r2 = []
    r3 = []
    r4 = []
    
    for (_, step) in enumerate(hist)
        a = step.a
        s = step.s
        o = step.o
        a_type = get_action_type(a)
        dnum = get_site_number(a)

        if a_type != "DONOTHING" 
            if !s.m[dnum]                     
                if a_type == "EXPLORE"            
                    push!(explore_actions, dnum)
                    push!(explore_times, step.t)
                elseif a_type == "MINE"
                    push!(invest_actions, dnum)
                    push!(invest_times, step.t)
                else 
                    push!(restore_actions, dnum)
                    push!(restore_times, step.t)
                end
            end            
        end

        push!(r1, compute_r1(pomdp, s, a))        
        push!(r2, compute_r2(pomdp, s, a))
        push!(r3, compute_r3(pomdp, s, a, o))
        push!(r4, compute_r4(pomdp, s, a, o))
    end

    return (
        a_explore=explore_actions, t_explore=explore_times, 
        a_mine=mine_actions, t_mine=mine_times, 
        a_invest=invest_actions, t_invest=invest_times,
        a_restore=restore_actions, t_restore=restore_times,
        r1=r1, r2=r2, r3=r3, r4=r4)
end


function plot_results(pomdp::LiPOMDP, df::NamedTuple;ylims=(-500, 500))
    T = pomdp.T
    #plot 1: actions vs time
    p0 = scatter(df.t_explore, df.a_explore, label="EXPLORE", markersize=10, xticks=1:T);
    scatter!(df.t_invest, df.a_invest, label="INVEST", markersize=10);
    scatter!(df.t_restore, df.a_restore, label="RESTORE", markersize=10);
    scatter!(
        df.t_mine, df.a_mine, 
        label="MINE", markersize=10, 
        xticks=0:1:T, 
        yticks=(1:4, ["1 (SilverPeak, USA)", "2 (ThackerPass, USA)", "3 (Greenbushes, AUS)", "4 (Pilgangoora, AUS)"]),
        ylims=(0.5, 4.5), 
        ylabel="Deposit Site", xlabel="Time", 
        title="Actions vs Time", 
        legend=:outerbottomright);
    vline!([pomdp.td], label="Time Delay Goal", color=:red, linestyle=:dash);        

    #set xticks to be integers
    p1 = bar(df.r1, label="r1", xlabel="Time", ylabel="\$ Value (in Millions)", title="Domestic Mining (Penalty)", legend=false, xticks=0:5:T, ylims=ylims);
    p3 = bar(df.r3, label="r3", xlabel="Time", ylabel="\$ Value (in Millions)", title="Unmet Demand (Penalty)", legend=false, ylims=ylims, xticks=0:5:T);
    p4 = bar(df.r4, label="r4", xlabel="Time", ylabel="\$ Value (in Millions)", title="Cash Flow", legend=false, ylims=ylims=ylims, xticks=0:5:T);
    p2 = bar(df.r2, label="r2", xlabel="Time", ylabel="Units", title="CO2 Emission", legend=false, ylims=(-30, 0), xticks=0:5:T);

    prow1 = plot(p1, p4, layout=(1, 2));  
    prow2 = plot(p2, p3, layout=(1, 2));  
    return (action=p0, econ=prow1, other=prow2)
end


function _get_rewards(pomdp, hist)
    explore_actions = []
    explore_times = []
    invest_actions = []
    invest_times = []
    restore_actions = []
    restore_times = []
    mine_on = []
    mine_on_times = []
    mine_off = []
    mine_off_times = []

    r1 = []
    r2 = []
    r3 = []
    r4 = []
    
    for (_, step) in enumerate(hist)
        a = step.a
        s = step.s
        o = step.o
        a_type = get_action_type(a)
        dnum = get_site_number(a)

        if a_type != "DONOTHING" 
            if !s.m[dnum]                     
                if a_type == "EXPLORE"            
                    push!(explore_actions, dnum)
                    push!(explore_times, step.t)
                elseif a_type == "MINE"
                    push!(invest_actions, dnum)
                    push!(invest_times, step.t)
                end
            else
                if a_type == "RESTORE"
                    push!(restore_actions, dnum)
                    push!(restore_times, step.t)
                end
            end            
        end

        for i in 1:pomdp.n
            if s.m[i] > 0
                push!(mine_on, i)
                push!(mine_on_times, step.t)
            else
                push!(mine_off, i)
                push!(mine_off_times, step.t)
            end
        end

        push!(r1, compute_r1(pomdp, s, a))        
        push!(r2, compute_r2(pomdp, s, a))
        push!(r3, compute_r3(pomdp, s, a, o))
        push!(r4, compute_r4(pomdp, s, a, o))
    end

    return (
        a_explore=explore_actions, t_explore=explore_times, 
        a_invest=invest_actions, t_invest=invest_times,
        a_restore=restore_actions, t_restore=restore_times,
        mine_on=mine_on, t_mine_on=mine_on_times,
        mine_off=mine_off, t_mine_off=mine_off_times,
        r1=r1, r2=r2, r3=r3, r4=r4)
end

function _plot_results(pomdp::LiPOMDP, df::NamedTuple;ylims=(-500, 500))
    cols = theme_palette(:auto)
    T = pomdp.T
    # Mine active indicators - solid black outline circles
    p0 = scatter(
        df.t_mine_on, df.mine_on,
        label="MINE ACTIVE", 
        markersize=10,
        markerstrokewidth=1,        # Thick outline
        markerstrokecolor=:black,   # Black outline
        markercolor=cols[4],         # White fill (transparent)
        markershape=:circle,        # Circle shape
        xticks=0:1:T, 
        yticks=(1:4, ["1 (SilverPeak, USA)", "2 (ThackerPass, USA)", "3 (Greenbushes, AUS)", "4 (Pilgangoora, AUS)"]),
        ylims=(0.5, 4.5), 
        ylabel="Deposit Site", 
        xlabel="Time", 
        title="Actions vs Time", 
        legend=:outerbottomright
    );
    
    # Mine inactive indicators - dashed gray outline circles
    scatter!(
        df.t_mine_off, df.mine_off,
        label="MINE INACTIVE", 
        markersize=10,
        markerstrokewidth=1,        # Normal outline width
        markerstrokecolor=:lightgrey,    # Gray outline
        markercolor=RGBA(1,1,1,0),         # White fill (transparent)
        markershape=:circle,        # Circle shape
        linestyle=:dash             # Dashed outline
    );
    scatter!(df.t_explore, df.a_explore, 
        label="EXPLORE", 
        markersize=10, 
        markerstrokewidth=1,  
        markercolor=cols[1],
        xticks=1:T
    );
    scatter!(df.t_invest, df.a_invest, 
        label="INVEST", 
        markersize=10,
        markercolor=cols[2],        
        markerstrokewidth=1   
    );
    scatter!(df.t_restore, df.a_restore, 
        label="RESTORE", 
        markersize=10,
        markercolor=cols[3],        
        markerstrokewidth=1  
    );
    
    
    
    vline!([pomdp.td], label="Time Delay Goal", color=:red, linestyle=:dash);        

    # Remaining plots
    p1 = bar(df.r1, label="r1", xlabel="Time", ylabel="\$ Value (in Millions)", title="Domestic Mining (Penalty)", legend=false, xticks=0:5:T, ylims=ylims);
    p3 = bar(df.r3, label="r3", xlabel="Time", ylabel="\$ Value (in Millions)", title="Unmet Demand (Penalty)", legend=false, ylims=ylims, xticks=0:5:T);
    p4 = bar(df.r4, label="r4", xlabel="Time", ylabel="\$ Value (in Millions)", title="Cash Flow", legend=false, ylims=ylims=ylims, xticks=0:5:T);
    p2 = bar(df.r2, label="r2", xlabel="Time", ylabel="Units", title="CO2 Emission", legend=false, ylims=(-30, 30), xticks=0:5:T);

    prow1 = plot(p1, p4, layout=(1, 2));  
    prow2 = plot(p2, p3, layout=(1, 2));  
    return (action=p0, econ=prow1, other=prow2)
end