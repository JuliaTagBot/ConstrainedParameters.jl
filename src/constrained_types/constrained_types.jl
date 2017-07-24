
abstract type Data end
abstract type parameters{T,N} <: AbstractArray{T,N} end


abstract type Constrainedparameters{p,T,N} <: parameters{T,N} end
function update!(A::AbstractArray)
end
log_jacobian!{T}(A::AbstractArray{T}) = zero(T)
Base.IndexStyle(::parameters) = IndexLinear()

type_length{T}(::Type{Vector{T}}) = 0
#type_length{p,T}(::Type{MVector{p,T}}) = p

function Base.show{T <: parameters}(io::IO, ::MIME"text/plain", Θ::T)
  for j in 2:length(fieldnames(T))
    println(getfield(Θ, j))
  end
end
function Base.show{T <: parameters}(io::IO, Θ::T)
  for j in 2:length(fieldnames(T))
    println(getfield(Θ, j))
  end
end

abstract type SquareMatrix{p, T} <: Constrainedparameters{p, T, 2} end

update!(Θ::Constrainedparameters) = nothing

@generated function Base.size{T <: SquareMatrix}(A::T)
  p = T.parameters[1]
  l = round(Int, p * (p + 1) / 2)
  (l, )
end

function Base.size{p}(A::SquareMatrix{p,<:Real})
  (p, p)
end
