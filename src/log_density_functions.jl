


function lpdf_normal(x::Real, σ::Real)
  -log(σ) - (x/σ)^2/2
end
function lpdf_normal(x::Vector, σ::Real)
  -log(σ)*length(x) - sum( x .^ 2 ) / 2σ^2
end
function lpdf_normal(x::AbstractArray{<:Real,1}, μ::AbstractArray{<:Real,1}, σ::Real)
  -log(σ)*length(x) - sum( (x .- μ) .^ 2 ) / 2σ^2
end
function lpdf_normal{p}(x::AbstractArray{<:Real,1}, μ::ConstrainedVector{p, <: Real}, σ::Real)
  -log(σ)*p - sum( (x .- μ.x) .^ 2 ) / 2σ^2
end
function lpdf_normal{p}(x::ConstrainedVector{p, <: Real}, μ::AbstractArray{<:Real,1}, σ::Real)
  -log(σ)*p - sum( (x.x .- μ) .^ 2 ) / 2σ^2
end
function lpdf_normal{p}(x::ConstrainedVector{p, <: Real}, μ::ConstrainedVector{p, <: Real}, σ::Real)
  -log(σ)*p - sum( (x.x .- μ.x) .^ 2 ) / 2σ^2
end
function lpdf_normal(x::Vector, Σ::CovarianceMatrix)
  negative_log_root_det(Σ) - quad_form(x, Σ)/2
end
function lpdf_normal(x::Vector, Σ::AbstractArray{<: Real,2})
  Σi = chol(Symmetric(Σ))
  out = logdet(Σi)/2
  for i ∈ 1:size(Σi,1)
    out -= x[i]^2 * Σi[i,i]/2
    for j ∈ 1:i-1
      out -= x[i] * Σi[j,i] * x[j]
    end
  end
  out
end

lpdf_normal(x::Real, μ::Real, σ::Real) = lpdf_normal(x-μ, σ)
lpdf_normal(x::Real, μ::AbstractArray{<:Real,1}, σ::Real) = lpdf_normal( x - μ, σ )
lpdf_normal(x::AbstractArray{<:Real,1}, μ::Real, σ::Real) = lpdf_normal( x - μ, σ )

lpdf_normal(x::Real, μ::Real, σ::AbstractArray{<:Real,1}) = sum(lpdf_normal.(x-μ, σ))
lpdf_normal(x::Real, μ::Real, σ::ConstrainedVector) = sum(lpdf_normal.(x-μ, σ.x))
lpdf_normal(x::Real, μ::AbstractArray{<:Real,1}, σ::AbstractArray{<:Real,1}) = sum(lpdf_normal.( x .- μ, σ ))
lpdf_normal(x::Real, μ::ConstrainedVector, σ::AbstractArray{<:Real,1}) = sum(lpdf_normal.( x .- μ.x, σ ))
lpdf_normal(x::Real, μ::AbstractArray{<:Real,1}, σ::ConstrainedVector) = sum(lpdf_normal.( x .- μ, σ.x ))
lpdf_normal(x::Real, μ::ConstrainedVector, σ::ConstrainedVector) = sum(lpdf_normal.( x .- μ.x, σ.x ))
lpdf_normal(x::AbstractArray{<:Real,1}, μ::Real, σ::AbstractArray{<:Real,1}) = sum(lpdf_normal.( x .- μ, σ ))
lpdf_normal(x::ConstrainedVector, μ::Real, σ::AbstractArray{<:Real,1}) = sum(lpdf_normal.( x.x .- μ, σ ))
lpdf_normal(x::AbstractArray{<:Real,1}, μ::Real, σ::ConstrainedVector) = sum(lpdf_normal.( x .- μ, σ.x ))
lpdf_normal(x::ConstrainedVector, μ::Real, σ::ConstrainedVector) = sum(lpdf_normal.( x.x .- μ, σ.x ))
lpdf_normal(x::AbstractArray{<:Real,1}, μ::AbstractArray{<:Real,1}, σ::AbstractArray{<:Real,1}) = sum(lpdf_normal.( x .- μ, σ ))
lpdf_normal(x::ConstrainedVector, μ::AbstractArray{<:Real,1}, σ::AbstractArray{<:Real,1}) = sum(lpdf_normal.( x.x .- μ, σ ))
lpdf_normal(x::AbstractArray{<:Real,1}, μ::ConstrainedVector, σ::AbstractArray{<:Real,1}) = sum(lpdf_normal.( x .- μ.x, σ ))
lpdf_normal(x::AbstractArray{<:Real,1}, μ::AbstractArray{<:Real,1}, σ::ConstrainedVector) = sum(lpdf_normal.( x .- μ, σ.x ))
lpdf_normal(x::ConstrainedVector, μ::ConstrainedVector, σ::AbstractArray{<:Real,1}) = sum(lpdf_normal.( x.x .- μ.x, σ ))
lpdf_normal(x::ConstrainedVector, μ::AbstractArray{<:Real,1}, σ::ConstrainedVector) = sum(lpdf_normal.( x.x .- μ, σ.x ))
lpdf_normal(x::AbstractArray{<:Real,1}, μ::ConstrainedVector, σ::ConstrainedVector) = sum(lpdf_normal.( x .- μ.x, σ.x ))
lpdf_normal(x::ConstrainedVector, μ::ConstrainedVector, σ::ConstrainedVector) = sum(lpdf_normal.( x.x .- μ.x, σ.x ))

#lpdf_normal(x::Real, μ::Real, Σ::AbstractArray{<:Real,2}) = lpdf_normal(x - μ, Σ)
lpdf_normal(x::Real, μ::AbstractArray{<:Real,1}, Σ::AbstractArray{<:Real,2}) = lpdf_normal( x - μ, Σ )
lpdf_normal(x::AbstractArray{<:Real,1}, μ::Real, Σ::AbstractArray{<:Real,2}) = lpdf_normal( x - μ, Σ )
lpdf_normal(x::AbstractArray{<:Real,1}, μ::AbstractArray{<:Real,1}, Σ::AbstractArray{<:Real,2}) = lpdf_normal( x - μ, Σ )



function lpdf_InverseWishart{p,T}(Σ::CovarianceMatrix{p,T}, Λ::AbstractArray{<: Real,2}, ν::Real)
  negative_log_root_det(Σ)*(ν + p + 1) - trace_AΣinv(A, Σ) / 2
end
function lpdf_InverseWishart{p,T}(Σ::CovarianceMatrix{p,T}, Λ::Real, ν::Real)
  negative_log_root_det(Σ)*(ν + p + 1) - Λ * trace_inverse(Σ) / 2
end
lpdf_InverseWishart{p,T}(Σ::CovarianceMatrix{p,T}, Λ::UniformScaling, ν::Real) = lpdf_InverseWishart{p,T}(Σ, Λ.λ, ν)
