include("express.jl")
df = _get_rewards(pomdp, hhist);
p = _plot_results(pomdp, df);
pall = plot(p.action, p.econ, p.other, layout=(3, 1), size=(1100, 800), margin=5mm)
# savefig(pall, "results.pdf")