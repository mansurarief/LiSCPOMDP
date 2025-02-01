include("express.jl")

using Printf
using Unitful
using Revise

# Define metric tonne unit
@unit MT "MT" MetricTonne 1000u"kg" false

"""
    plot_localized_timeline(hist::SimHistory, volume_ranges::Vector{Tuple{<:Quantity{<:Real,Unitful.𝐌},<:Quantity{<:Real,Unitful.𝐌}}}, T::Int=10; 
                          spacing::Float64=3.0, max_width::Float64=0.4)

Create scatter plots showing the true lithium reserves and belief means for each deposit over time,
using localized scales for better visualization.

Arguments:
- hist: Simulation history (values are in metric tonnes)
- volume_ranges: Vector of (min, max) tuples for each deposit's volume range in metric tonnes
- T: Maximum time steps to plot (default: 10)
- spacing: Vertical spacing between deposits (default: 3.0)
- max_width: Maximum width of the belief distribution plots (default: 0.4)
"""
function plot_localized_timeline(hist::Union{SimHistory, Vector}, volume_ranges::Vector, T::Int=10; 
                               spacing::Float64=3.0, max_width::Float64=0.8, ms::Float64=2.0, 
                               title::String="Li Reserves Uncertainty vs. True Reserves Volume",
                               show_legend::Bool=true,
                               show_xlabel::Bool=true, 
                               show_ylabel::Bool=true)
    @assert length(volume_ranges) == 4 "Must provide volume ranges for all 4 deposits"
    
    # Get theme colors
    cols = theme_palette(:auto)
    
    # Main plot
    p_main = plot(
        size=(800, 800),
        xlabel=show_xlabel ? "Time Period" : "",
        title=title,
        grid=true,
        gridstyle=:dash,
        gridalpha=0.3,
        legend=show_legend ? :outertop : false,  
        legend_columns=3,  # Arrange legend in 3 columns
        right_margin=20mm,
        left_margin=5mm,  
        top_margin=5mm,   
        bottom_margin=0mm,  
        guidefontsize=12,  
        tickfontsize=11,   
        titlefontsize=14,  
        legendfontsize=10, 
        background_color_legend=nothing  
    )

    if show_legend
        scatter!([], [], label="EXPLORE", markercolor=cols[1], markerstrokecolor=:black, markershape=:circle, markersize=6)
        scatter!([], [], label="Mine Operating", markercolor=cols[4], markerstrokecolor=:black, markershape=:circle, markersize=6)
        scatter!([], [], label="MINE", markercolor=cols[2], markerstrokecolor=:black, markershape=:circle, markersize=6)
        scatter!([], [], label="Mine Not Operating", markercolor=RGBA(1,1,1,0), markerstrokecolor=:lightgrey, markershape=:circle, markersize=6, linestyle=:dash)
        scatter!([], [], label="RESTORE", markercolor=cols[3], markerstrokecolor=:black, markershape=:circle, markersize=6)
        scatter!([], [], label="True Reserves Volume", color=:black, markershape=:circle, markersize=ms)
    end

    # Colors for different deposits
    deposit_colors = [:blue, :red, :green, :purple]
    
    # For each deposit
    for i in 1:4
        # Define local range for this deposit
        center = i * spacing
        band_height = spacing * 0.25  # Scale band height with spacing
        local_min = center - band_height
        local_max = center + band_height
        
        # Get true values for this deposit up to time T (already in tonnes)
        true_values = [step.s.v[i] * 1.0MT for step in hist][1:min(T, length(hist))]
        times = 1:length(true_values)
        
        # Get volume range for this deposit
        value_min, value_max = volume_ranges[i]
        value_range = ustrip(value_max - value_min)
        
        # Scale true values to local range
        scaled_true = local_min .+ (ustrip.(true_values .- value_min)) ./ value_range * (2 * band_height)
        
        # Plot true values
        scatter!(p_main, times, scaled_true,
                color=:black,
                markershape=:circle,
                markersize=ms,
                label=nothing,
                alpha=0.8)
        
        # For each timestep, plot the rotated PDF
        for (t, step) in enumerate(hist[1:min(T, length(hist))])
            dist = step.b.v_dists[i]
            
            # Scale the distribution (values already in tonnes)
            scaled_dist = Normal(dist.μ, dist.σ)
            
            # Generate points for PDF
            y_points = range(ustrip(value_min), ustrip(value_max), length=100)
            pdf_values = pdf.(scaled_dist, y_points)
            
            # Normalize PDF values to fit between time steps
            pdf_values = pdf_values ./ maximum(pdf_values) .* max_width
            
            # Scale y values to local range
            y_scaled = local_min .+ (y_points .- ustrip(value_min)) ./ value_range * (2 * band_height)
            
            # Plot the PDF (only right side)
            x_values = fill(t, length(y_points)) .+ pdf_values
            
            # Plot filled area
            plot!(p_main, [fill(t, length(y_points)), x_values], [y_scaled, y_scaled],
                  fill=true,
                  fillalpha=0.2,
                  fillcolor=deposit_colors[i],
                  label=nothing,
                  linewidth=0)
            
            # Plot outline
            plot!(p_main, x_values, y_scaled,
                  color=deposit_colors[i],
                  linewidth=1,
                  alpha=0.4,
                  label=nothing)
        end
        
        # Add horizontal lines to separate deposits
        hline!(p_main, [local_min, local_max], color=:gray, alpha=0.2, label=nothing)
        
        # Format numbers with commas for thousands
        function format_with_commas(x::Real)
            return replace(string(round(Int, x)), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")
        end
        
        min_text = string(format_with_commas(ustrip(value_min)), " MT")
        max_text = string(format_with_commas(ustrip(value_max)), " MT")
        
        # Right side annotations only (with units and commas)
        annotate!(p_main, T + 0.7, local_min, text(min_text, :left, 9, :gray))  # Increased font size
        annotate!(p_main, T + 0.7, local_max, text(max_text, :left, 9, :gray))  # Increased font size
        
        # Plot operating status and action circles
        circle_y = local_max + 0.08  # Position near the top of the band
        for (t, step) in enumerate(hist[1:min(T, length(hist))])
            # Get action for this step
            action = step.a
            action_type = get_action_type(action)
            site_number = get_site_number(action)
            
            if site_number == i  # Action targets this deposit
                if action_type == "EXPLORE"
                    # Explore action - blue fill
                    scatter!(p_main, [t], [circle_y],
                        markersize=6,
                        markerstrokewidth=1,
                        markerstrokecolor=:black,
                        markercolor=cols[1],  # Blue from theme
                        markershape=:circle,
                        label=nothing)
                elseif action_type == "MINE"
                    # Mine action - orange fill
                    scatter!(p_main, [t], [circle_y],
                        markersize=6,
                        markerstrokewidth=1,
                        markerstrokecolor=:black,
                        markercolor=cols[2],  # Orange from theme
                        markershape=:circle,
                        label=nothing)
                elseif action_type == "RESTORE"
                    # Restore action - purple fill
                    scatter!(p_main, [t], [circle_y],
                        markersize=6,
                        markerstrokewidth=1,
                        markerstrokecolor=:black,
                        markercolor=cols[3],  # Purple from theme
                        markershape=:circle,
                        label=nothing)
                else
                    # No action - show operating status
                    if step.s.m[i]  # Using state's mining status
                        # Active mine - solid black outline, colored fill
                        scatter!(p_main, [t], [circle_y],
                            markersize=6,
                            markerstrokewidth=1,
                            markerstrokecolor=:black,
                            markercolor=cols[4],  # Green from theme
                            markershape=:circle,
                            label=nothing)
                    else
                        # Inactive mine - dashed gray outline, transparent fill
                        scatter!(p_main, [t], [circle_y],
                            markersize=6,
                            markerstrokewidth=1,
                            markerstrokecolor=:lightgrey,
                            markercolor=RGBA(1,1,1,0),
                            markershape=:circle,
                            linestyle=:dash,
                            label=nothing)
                    end
                end
            else
                # No action for this deposit - show operating status
                if step.s.m[i]  # Using state's mining status
                    # Active mine - solid black outline, colored fill
                    scatter!(p_main, [t], [circle_y],
                        markersize=6,
                        markerstrokewidth=1,
                        markerstrokecolor=:black,
                        markercolor=cols[4],  # Green from theme
                        markershape=:circle,
                        label=nothing)
                else
                    # Inactive mine - dashed gray outline, transparent fill
                    scatter!(p_main, [t], [circle_y],
                        markersize=6,
                        markerstrokewidth=1,
                        markerstrokecolor=:lightgrey,
                        markercolor=RGBA(1,1,1,0),
                        markershape=:circle,
                        linestyle=:dash,
                        label=nothing)
                end
            end
        end
    end
    
    # Adjust plot limits
    ylims!(p_main, 0.5, 4.5 * spacing)
    xlims!(p_main, 0.2, T + 1.2)
    
    # Add y-axis ticks at deposit centers
    if show_ylabel
        yticks!(p_main, [i * spacing for i in 1:4], ["Deposit $i" for i in 1:4])
    else
        yticks!(p_main, [0 for i in 1:4], [" " for i in 1:4])
    end
    
    return p_main
end



hist_non_pomdp = hhist
hist_wo_bp = []
for i in 1:length(hist_non_pomdp)
    push!(hist_wo_bp, (
        s = hist_non_pomdp[i].s,    # state
        a = hist_non_pomdp[i].a,    # action
        o = hist_non_pomdp[i].o,    # observation
        r = hist_non_pomdp[i].r,    # reward
        b = hist_non_pomdp[1].b,    # use initial belief
        bp = hist_non_pomdp[1].b,   # use initial belief for next belief too
        t = i              # timestep
    ))
end

hist_pomdp = h3hist;



df = _get_rewards(pomdp, hist_pomdp);
p = _plot_results(pomdp, df);
pall = plot(p.action, p.econ, p.other, layout=(3, 1), size=(1100, 800), margin=5mm)


#incorrect-case plot limits
incorrect_limits = [
    (30_000.0MT, 80_000.0MT),
    (10_000.0MT, 60_000.0MT),
    (50_000.0MT, 100_000.0MT),
    (20_000.0MT, 80_000.0MT)
]

correct_limits = [
    (20_000.0MT, 60_000.0MT),
    (0.0MT, 40_000.0MT),
    (30_000.0MT, 80_000.0MT),
    (10_000.0MT, 40_000.0MT)
]

p1 = plot_localized_timeline(
    hist_pomdp, correct_limits, 30, spacing=1.0, show_legend=false,
    show_xlabel=true, 
    show_ylabel=true,
    title="POMDP Policies (e.g. POMCPOW)")

p2 = plot_localized_timeline(
    hist_wo_bp, correct_limits, 30, spacing=1.0, show_legend=false,
    show_xlabel=true, 
    show_ylabel=false,
    title="Non-POMDP Policies (e.g. Greedy)")

# Create a legend-only plot


function plot_legend()
    # Get theme colors
    cols = theme_palette(:auto)
        plegend = plot(
        grid=false,
        showaxis=false,
        ticks=false,
        xticks=[],
        yticks=[],
        legend=:top,
        legend_columns=3,
        background_color=:white,
        background_color_legend=:white,  # White background for legend
        foreground_color_legend=:gray,  # Black box around legend
        legendfontsize=10,  # Match other font sizes
        size=(1400, 50),  # Small height since it's just for legend
        framestyle=:none,
        top_margin=3mm,
        bottom_margin=10mm
    )
    # Add the legend entries    
    scatter!(plegend, [], [], label="EXPLORE", markercolor=cols[1], markerstrokecolor=:black, markershape=:circle, markersize=6)
    scatter!(plegend, [], [], label="MINE", markercolor=cols[2], markerstrokecolor=:black, markershape=:circle, markersize=6)
    scatter!(plegend, [], [], label="RESTORE", markercolor=cols[3], markerstrokecolor=:black, markershape=:circle, markersize=6)
    scatter!(plegend, [], [], label="Mine Operating", markercolor=cols[4], markerstrokecolor=:black, markershape=:circle, markersize=6)
    scatter!(plegend, [], [], label="Mine Not Operating", markercolor=RGBA(1,1,1,0), markerstrokecolor=:lightgrey, markershape=:circle, markersize=6, linestyle=:dash)
    scatter!(plegend, [], [], label="True Reserves Volume", color=:black, markershape=:circle, markersize=0.5)
    return plegend
end

plegend = plot_legend()

# Create the final combined plot with legend and main plots
pall = plot(p1, p2, plegend,
    layout=@layout([
         # Main plots row 95% height
         grid(1,2){0.95h}
         a{0.05h}  
    ]), 
    size=(1200, 700),   # Total plot size
    left_margin=2mm,
    right_margin=20mm,  # Increased right margin
    # top_margin=5mm,
    # bottom_margin=10mm
)

savefig(pall, "results.pdf")