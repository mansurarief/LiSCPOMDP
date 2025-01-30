include("express.jl")

using Printf
using Unitful
using Revise

# Define metric tonne unit
@unit MT "MT" MetricTonne 1000u"kg" false

df = _get_rewards(pomdp, hhist);
p = _plot_results(pomdp, df);
pall = plot(p.action, p.econ, p.other, layout=(3, 1), size=(1100, 800), margin=5mm)
# savefig(pall, "results.pdf")


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
function plot_localized_timeline(hist::SimHistory, volume_ranges::Vector, T::Int=10; 
                               spacing::Float64=3.0, max_width::Float64=0.6, ms::Float64=2.0)
    @assert length(volume_ranges) == 4 "Must provide volume ranges for all 4 deposits"
    
    # Main plot
    p_main = plot(
        size=(1000, 800),
        xlabel="Time Period",
        ylabel="Deposit",
        title="Lithium Reserves Over Time (Localized Scales)",
        grid=true,
        gridstyle=:dash,
        gridalpha=0.3,
        legend=false,  # No built-in legend
        right_margin=40mm
    )

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
        annotate!(p_main, T + 0.7, local_min, text(min_text, :left, 7, :gray))
        annotate!(p_main, T + 0.7, local_max, text(max_text, :left, 7, :gray))
    end
    
    # Adjust plot limits
    ylims!(p_main, 0.5, 4.5 * spacing)
    xlims!(p_main, 0.2, T + 1.2)
    
    # Add y-axis ticks at deposit centers
    yticks!(p_main, [i * spacing for i in 1:4], ["Deposit $i" for i in 1:4])
    
    return p_main
end



# Example usage with metric tonnes
p = plot_localized_timeline(hhist, [
    (0.0MT, 50_000.0MT),
    (0.0MT, 25_000.0MT),
    (44_000.0MT, 62_000.0MT),
    (5_000.0MT, 32_000.0MT)
], 30, spacing=1.5)