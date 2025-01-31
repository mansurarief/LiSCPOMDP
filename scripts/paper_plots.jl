
cols = theme_palette(:auto)
T = pomdp.T

function _get_baseplot(pomdp::LiPOMDP, df::NamedTuple;ylims=(-500, 500))
        # Mine active indicators - solid black outline circles
    node_offsets = 0.2
    p0 = scatter(
        df.t_mine_on, df.mine_on.-node_offsets,
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
    )

    # Mine inactive indicators - dashed gray outline circles
    scatter!(
        df.t_mine_off, df.mine_off.-node_offsets,
        label="MINE INACTIVE", 
        markersize=10,
        markerstrokewidth=1,        # Normal outline width
        markerstrokecolor=:lightgrey,    # Gray outline
        markercolor=RGBA(1,1,1,0),         # White fill (transparent)
        markershape=:circle,        # Circle shape
        linestyle=:dash             # Dashed outline
    )
    scatter!(df.t_explore, df.a_explore.-node_offsets, 
        label="EXPLORE", 
        markersize=10, 
        markerstrokewidth=1,  
        markercolor=cols[1],
        xticks=1:T
    )

    scatter!(df.t_invest, df.a_invest.-node_offsets, 
        label="INVEST", 
        markersize=10,
        markercolor=cols[2],        
        markerstrokewidth=1   
    )
    scatter!(df.t_restore, df.a_restore.-node_offsets, 
        label="RESTORE", 
        markersize=10,
        markercolor=cols[3],        
        markerstrokewidth=1  
    )



    vline!([pomdp.td], label="Time Delay Goal", color=:black, linestyle=:dash)

    
    xstarts = [0, 0, 0, 0]
    xends = [T, T, T, T]

    y_bot = [0.9, 1.9, 2.9, 3.9]
    y_top = [1.1, 2.1, 3.1, 4.1]

    for i in 1:4
        x1 = xstarts[i]
        x2 = xends[i]
        plot!([x1; x2], [y_bot[i]-node_offsets; y_bot[i]-node_offsets], lw=2, lc=:black, label="")
        plot!([x1; x2], [y_top[i]-node_offsets; y_top[i]-node_offsets], lw=2, lc=:black, label="")
    end


    out_plot = plot(p0, layout=(1, 1), size=(1100, 500), margin=5mm)
    return out_plot

end



function plot_obs_belief(p, obs, bs, left_shift=3, init_shift=3)

    p0 = plot(p)

    scatter!([-left_shift], [obs[1]],
        label="",
        markersize=4,
        markerstrokewidth=0,        # Thick outline
        markerstrokecolor=:black,   # Black outline       # White fill (transparent)
        markershape=:circle, 
        markercolor=cols[1],      
    )

    scatter!([-left_shift], [obs[2]],
        label="",
        markersize=4,
        markerstrokewidth=0,        # Thick outline
        markerstrokecolor=:black,   # Black outline       # White fill (transparent)
        markershape=:circle, 
        markercolor=cols[2],      
    )

    scatter!([-left_shift], [obs[3]],
        label="",
        markersize=4,
        markerstrokewidth=0,        # Thick outline
        markerstrokecolor=:black,   # Black outline       # White fill (transparent)
        markershape=:circle, 
        markercolor=cols[3],      
    )

    scatter!([-left_shift], [obs[4]],
        label="",
        markersize=4,
        markerstrokewidth=0,        # Thick outline
        markerstrokecolor=:black,   # Black outline       # White fill (transparent)
        markershape=:circle, 
        markercolor=cols[4],      
    )


    xlims!(-init_shift-0.2, 31)

    ys_pdf = 0:0.01:4.5
    scale = abs(left_shift)^(-0.85)
    
    pdf_scale = [0.5*scale for _ in 1:4]
    pdfs = [pdf.(bs[i], ys_pdf).* pdf_scale[i] for i in 1:4]
    #only add val into pdfs if its is greater than 0.0001
    pdfs = [[val > 0.0001 ? val : NaN for val in pdfs[i]] for i in 1:4]

    
    plot!(pdfs[1].-left_shift.-0.25, ys_pdf, color=cols[1], lw=2, label="")
    plot!(pdfs[2].-left_shift.-0.25, ys_pdf, color=cols[2], lw=2, label="")
    plot!(pdfs[3].-left_shift.-0.25, ys_pdf, color=cols[3], lw=2, label="")
    plot!(pdfs[4].-left_shift.-0.25, ys_pdf, color=cols[4], lw=2, label="")

    #add short line from x=-3 to x=0.5 for each color
    plot!([-left_shift, 0.5], [1, 1], color=cols[1], lw=1, ls=:dash, label="")
    plot!([-left_shift, 0.5], [2, 2], color=cols[2], lw=1, ls=:dash, label="")
    plot!([-left_shift, 0.5], [3, 3], color=cols[3], lw=1, ls=:dash, label="")
    plot!([-left_shift, 0.5], [4, 4], color=cols[4], lw=1, ls=:dash, label="")
    return p0
end




function update_belief(b, o, σo)
    ms = [kalman_step(σo, mean(b[i]), std(b[i]), o[i]) for i in 1:4]    
    return [Normal(ms[i][1], ms[i][2]) for i in 1:4]
end

function kalman_step(σo, μ::Float64, σ::Float64, z::Float64)
    k = σ / (σ + σo)  # Kalman gain
    μ_prime = μ + k * (z - μ)  # Estimate new mean
    σ_prime = (1 - k) * σ   # Estimate new uncertainty
    return μ_prime, σ_prime
end





σo = 0.2
true_s = [0.89, 2.34, 3.31, 4.12]
dist_s = [Normal(v, σo) for v in true_s]

rng_v = MersenneTwister(123)

o0 = [rand(rng_v, dist_s[i]) for i in 1:4]

# obs = [1.05, 2.4, 3.3, 4.16]
b0s = [
    Normal(1.23, 0.15), 
    Normal(2.26, 0.18),
    Normal(3.15, 0.15),
    Normal(4.12, 0.1)]

n_iter = 15
p = _get_baseplot(pomdp, df)
p0 = plot_obs_belief(p, o0, b0s)
savefig(p0, "data/belief0.png")


o1 = [rand(rng_v, v_dist) for v_dist in dist_s]
b1 = update_belief(b0s, o1, σo)
p1 = plot_obs_belief(p0, o1, b1, -1)
savefig(p1, "data/belief15.png")

for i in 1:n_iter
    o1 = [rand(rng_v, v_dist) for v_dist in dist_s]
    b1 = update_belief(b1, o1, σo)
    p1 = plot_obs_belief(p1, o1, b1, -1-i)
end

p1
