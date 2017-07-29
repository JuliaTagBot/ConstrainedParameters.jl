abstract type ConstrainedVector{p,T} <: Constrainedparameters{p,T,1} end

struct PositiveVector{p, T} <: ConstrainedVector{p, T}
  Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
  x::MVector{p,T}
end
struct ProbabilityVector{p, T} <: ConstrainedVector{p, T}
  Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
  x::MVector{p,T}
end
struct RealVector{p, T} <: ConstrainedVector{p, T}
  x::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
end
struct Simplex{p, q, T} <: ConstrainedVector{p, T}
  Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
  x::MVector{p,T}
  z::MVector{q,T}
  csx::MVector{q,T}
end


Base.:+(x::ConstrainedVector, y::Vector) = x.x .+ y
Base.:+(y::Vector, x::ConstrainedVector) = x.x .+ y
Base.:+(x::ConstrainedVector, y::Real) = x.x .+ y
Base.:+(y::Real, x::ConstrainedVector) = x.x .+ y
Base.:+(y::ConstrainedVector, x::ConstrainedVector) = x.x .+ y.x
Base.:-(x::ConstrainedVector, y::Vector) = x.x .- y
Base.:-(y::Vector, x::ConstrainedVector) = y .- x.x
Base.:-(x::ConstrainedVector, y::Real) = x.x .- y
Base.:-(y::Real, x::ConstrainedVector) = y .- x.x
Base.:-(x::ConstrainedVector, y::ConstrainedVector) = x.x .- y.x
Base.:*(A::AbstractArray{<:Real,2}, x::ConstrainedVector) = A * x.x
Base.:*(x::ConstrainedVector, A::AbstractArray{<:Real,2}) = x.x * A
Base.convert(::Type{Vector}, A::ConstrainedVector) = A.x
Base.show(io::IO, ::MIME"text/plain", Θ::ConstrainedVector) = print(Θ.x)
Base.size(x::ConstrainedVector) = size(x.x)
Base.getindex(x::ConstrainedVector, i::Int) = x.x[i]


PositiveVector(Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}) where {T} = PositiveVector{length(x), T}(Θ, MVector{length(x)}(log.(Θ)))
function update!(x::PositiveVector{p,T} where {T <: Real}) where {p}
  for i ∈ 1:p
    x.x[i] = exp(x.Θ[i])
  end
end
function log_jacobian!(x::PositiveVector)
  sum(x.Θ)
end
type_length(::Type{PositiveVector{p,T}}) where {p,T} = p
Base.getindex(x::PositiveVector, i::Int) = exp(x.Θ[i])
function Base.setindex!(x::PositiveVector, v::Real, i::Int)
  x.x[i] = v
  x.Θ[i] = log(v)
end
function construct(::Type{PositiveVector{p,T}}, Θv::Vector{T}, i::Int) where {p, T}
  v = view(Θv, i + (1:p))
  PositiveVector{p, T}(v, MVector{p}(exp.(v)))
end
function construct(::Type{PositiveVector{p,T}}, Θv::Vector{T}, i::Int, vals::Vector{T}) where {p, T}
  pv = PositiveVector{p, T}(view(Θv, i + (1:p)), MVector{p}(vals))
  pv.Θ .= log.(pv.x)
  pv
end



ProbabilityVector(Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}) where {T} = ProbabilityVector{length(x), T}(Θ, MVector{length(x)}(logit.(Θ)))
function update!(x::ProbabilityVector{p, T} where {T <: Real}) where {p}
  for i ∈ 1:p
    x.x[i] = logistic(x.Θ[i])
  end
end
function log_jacobian!(x::ProbabilityVector)
  sum(log.(x.x) .+ log.(1 .- x.x))
end
type_length(::Type{ProbabilityVector{p,T}}) where {p,T} = p
Base.getindex(x::ProbabilityVector, i::Int) = x.x[i]
function Base.setindex!(x::ProbabilityVector, v::Real, i::Int)
  x.x[i] = v
  x.Θ[i] = logit(v)
end
function construct(::Type{ProbabilityVector{p,T}}, Θv::Vector{T}, i::Int) where {p, T}
  v = view(Θv, i + (1:p))
  ProbabilityVector{p, T}(v, MVector{p}(logistic.(v)))
end
function construct(::Type{ProbabilityVector{p,T}}, Θv::Vector{T}, i::Int, vals::Vector{T}) where {p, T}
  pv = ProbabilityVector{p, T}(view(Θv, i + (1:p)), MVector{p}(vals))
  pv.Θ .= logit.(vals)
  pv
end

function Base.setindex!(x::RealVector, v::Real, i::Int)
  x.x[i] = v
end
function update!(x::RealVector)
end
@generated log_jacobian!(x::RealVector{p, T} where {p}) where {T} = zero(T)

type_length(::Type{RealVector{p,T}}) where {p,T} = p
function construct(::Type{RealVector{p,T}}, Θ::Vector{T}, i::Int) where {p, T}
  RealVector{p, T}(view(Θ, i + (1:p)))
end
function construct(::Type{RealVector{p,T}}, Θ::Vector{T}, i::Int, vals::Vector{T}) where {p, T}
  rv = RealVector{p, T}(view(Θ, i + (1:p)))
  copy!(rv.x, vals)
  rv
end


#Simplex(Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}) where {T} = Simplex{length(x), T}(Θ, MVector{length(x)}(SimplexTransform.(Θ)))
function update!(x::Simplex{p, q, T} where {T <: Real}) where {p, q}
  for i ∈ eachindex(x.Θ)
    x.z[i] = logistic(x.Θ[i] - log( p - i ) )
  end
  x.csx[1] = x.x[1] = x.z[1]
  for i ∈ 2:q
    x.x[i] = (1 - x.csx[i-1]) * x.z[i]
    x.csx[i] = x.x[i] + x.csx[i-1]
  end
  x.x[end] = 1 - x.csx[end]
end
function log_jacobian!(x::Simplex{p, q, T} where {p,T}) where {q}
  out = log(x.z[1]) + log(1 - x.z[1])
  for i ∈ 2:q
    out += log(x.z[i]) + log(1 - x.z[i]) + log(1 - x.csx[i - 1])
  end
  out
end
type_length(::Type{Simplex{p,q,T}} where {p,T}) where {q} = q
Base.getindex(x::Simplex, i::Int) = x.x[i]

#Using setindex! is strongly discouraged.

function Base.setindex!(x::Simplex, v::Vector{ <: Real}, i::Int)
  x.x .*= (1 .- v) ./ (1 .- x.x[i])
  x.x[i] = v
  set_prob!(x)
end
function set_prob!(x::Simplex, π::AbstractArray{<:Real,1})
  x.x .= π
  set_prob!(x)
end
function set_prob!(x::Simplex{p,q,T}) where {p,q,T}
  x.csx[1] = x.z[1] = x.x[1]
  y[i] = logit(x.z[1]) + log(p - 1)
  for i ∈ 2:q
    x.csx[i] = x.csx[i-1] + x.x[i]
    x.z[i] = x.x[i] / (1 - x.csx[i-1])
    y[i] = logit(x.z[i]) + log(p - i)
  end
end
function Simplex(v::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}, q = length(v)) where {T}
  Simplex{q+1,q,T}(v, MVector{q+1}(Vector{T}(q+1)), MVector{q}(Vector{T}(q)), MVector{q}(Vector{T}(q)))
end
function construct(::Type{Simplex{p,q,T}} where {p}, Θv::Vector{T}, i::Int) where {q, T}
  out = Simplex(view(Θv, i + (1:q)), q)
  update!(out)
  out
end
function construct(::Type{Simplex{p,q,T}}, Θv::Vector{T}, i::Int, vals::Vector{T}) where {p, q, T}
  out = Simplex(view(Θv, i + (1:q)))
  set_prob!(out, vals)
  out
end

#struct LowerBoundVector{p, T}
#  Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
#  x::Vector{T}
#  L::Vector{T}
#end
