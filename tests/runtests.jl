using Test
using LiPOMDPs

@testset "LiSCPOMDP tests" begin
    @testset "POMDP" begin
        include("pomdp_tests.jl")
    end
end