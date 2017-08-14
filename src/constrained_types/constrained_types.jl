
abstract type Data end
abstract type parameters{T,N} <: AbstractArray{T,N} end
abstract type parameter{T} <: parameters{T,1} end
abstract type Constrainedparameters{p,T,N} <: parameters{T,N} end
VectorView{T,P} = Union{SubArray{T,1,P,Tuple{UnitRange{Int}},true}, SlidingVector{T,P}}
mutable_vector{T} = Union{MVector{p,T} where p, Vector{T}, SizedArray}

function update!(A::AbstractArray)
end
log_jacobian!(A::AbstractArray{T}) where {T} = zero(T)
Base.IndexStyle(::parameters) = IndexLinear()

type_length(::Type{Vector{T}} where {T}) = 0
param_type_length(::Type{Vector{T}} where {T}) = Val{0}
#type_length{p,T}(::Type{MVector{p,T}}) = p

function Base.show(io::IO, ::MIME"text/plain", Θ::T) where {T <: parameters}
  for j in 2:length(fieldnames(T))
    println(getfield(Θ, j))
  end
end
function Base.show(io::IO, Θ::T) where {T <: parameters}
  for j in 2:length(fieldnames(T))
    println(getfield(Θ, j))
  end
end

abstract type SquareMatrix{p, T} <: Constrainedparameters{p, T, 2} end

update!(Θ::Constrainedparameters) = nothing

@generated function Base.size(A::T) where {T <: SquareMatrix}
  p = T.parameters[1]
  l = round(Int, p * (p + 1) / 2)
  (l, )
end

function Base.size(A::SquareMatrix{p,<:Real}) where {p}
  (p, p)
end
