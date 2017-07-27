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


#struct LowerBoundVector{p, T}
#  Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
#  x::Vector{T}
#  L::Vector{T}
#end
