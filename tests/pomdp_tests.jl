using Test

@testset "POMDP initialization" begin
    # Make sure that we can initialize LiPOMDP w/o errors
    @test_nowarn initialize_lipomdp()
end
