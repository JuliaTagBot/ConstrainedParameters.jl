using ConstrainedParameters
using Base.Test

using ForwardDiff

# write your own tests here
constrained_types = [PositiveVector, ProbabilityVector, RealVector]


function cvt(::Type{q}, x::Vector{T}) where {T <: Real, q <: ConstrainedVector}
  construct(q{T}, x, 0).x
end
function cvt(::Type{CovarianceMatrix{p,T2} where T2 <: Real}, x::Vector{T}) where {T <: Real, p}
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
  @test log_jacobian!(cv) ≈ logabsdet(ForwardDiff.jacobian(x -> cvt(q{p}, x), x))[1]
end

p = rand(1:15)
x = randn(round(Int,p*(p+1)/2))
cv = construct(CovarianceMatrix{p,Float64},x,0)

#The CovarianceMatrix log_jacobian! function drops the constant p*log(2) term.
@test log_jacobian!(cv) + p*log(2) ≈ logabsdet(ForwardDiff.jacobian(x -> cvt(CovarianceMatrix{p}, x), x))[1]
