using ConstrainedParameters
using Base.Test

using ForwardDiff, StaticArrays

# write your own tests here
constrained_types = [PositiveVector, ProbabilityVector, RealVector]

#convstrained vector transforms
function cvt(::Type{q}, x::AbstractArray{T,1}) where {T <: Real, q <: ConstrainedVector}
  construct(q{T}, x, 0).x
end
function cvt(::Type{CovarianceMatrix{p,T2} where T2 <: Real}, x::AbstractArray{T,1}) where {T <: Real, p}
  cv = construct(CovarianceMatrix{p,T}, x, 0)
  ConstrainedParameters.update_Σ!(cv)
  out = similar(x)
  k = 0
  for i ∈ 1:p, j ∈ 1:i
    k += 1
    out[k] = cv.Σ[j,i]
  end
  out
end

@testset for q ∈ constrained_types
  p = rand(1:100)
  x = randn(p)
  cv = construct(q{p,Float64},x,0)
  cv
  @test log_jacobian!(cv) ≈ logabsdet(ForwardDiff.jacobian(x -> cvt(q{p}, x), x))[1]
  @test length(cv) == p
end
Base.logdet(A::Symmetric) = 2logdet(chol(A))

p = rand(1:15)
x = randn(round(Int,p*(p+1)/2))
cv = construct(CovarianceMatrix{p,Float64},x,0)
cv
S = randn(2p,p) |> x -> x' * x
y = @SVector randn(p);
μ = @SVector randn(p);
#The CovarianceMatrix log_jacobian! function drops the constant p*log(2) term.
@testset begin
  @test log_jacobian!(cv) + p*log(2) ≈ logabsdet(ForwardDiff.jacobian(x -> cvt(CovarianceMatrix{p}, x), x))[1]
  update_Σ!(cv)
  @test lpdf_InverseWishart(cv, 3.0I, p+1) ≈ -(p+1)*logdet(cv.Σ) - trace(3 * inv(cv.Σ)) / 2
  @test lpdf_InverseWishart(cv, S, p+1) ≈ -(p+1)*logdet(cv.Σ) - trace(S * inv(cv.Σ)) / 2
  @test lpdf_normal(y, μ, cv) ≈ -( logdet(cv.Σ)  + (y .- μ)' * inv(cv.Σ) * (y .- μ) ) / 2
  @test ConstrainedParameters.type_length(typeof(cv)) == round(Int, p*(p+1)/2)
  @test length(cv) == p^2
end
