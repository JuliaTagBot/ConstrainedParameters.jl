abstract type ConstrainedVector{p,T} <: Constrainedparameters{p,T,1} end
abstract type CTV{p} end
struct PositiveVector{p} <: CTV{p} end
struct ProbabilityVector{p} <: CTV{p} end
struct RealVector{p} <: CTV{p} end
struct Simplex{p,q} <: CTV{p} end

Base.@pure PositiveVector(p) = PositiveVector{p}()
Base.@pure ProbabilityVector(p) = ProbabilityVector{p}()
Base.@pure RealVector(p) = RealVector{p}()
Base.@pure Simplex(p) = Simplex{p,p-1}()

struct PosiVec{p, T, P <: AbstractArray{T,1}, V <: VectorView{T,P}, D <: StaticArray{Tuple{p}, T, 1} } <: ConstrainedVector{p, T}
  Θ::V
  x::D
end
struct ProbVec{p, T, P <: AbstractVector{T}, V <: VectorView{T,P}, D <: StaticArray{Tuple{p}, T, 1}} <: ConstrainedVector{p, T}
  Θ::V
  x::D
end
struct RealVec{p, T, P <: AbstractVector{T}, V <: VectorView{T,P}} <: ConstrainedVector{p, T}
  x::V
end
struct Simp{p, q, T, P <: AbstractVector{T}, V <: VectorView{T,P}, D <: StaticArray{Tuple{p}, T, 1}, R <: StaticArray{Tuple{q}, T, 1}} <: ConstrainedVector{p, T}
  Θ::V
  x::D
  z::R
  csx::R
end


Base.:+(x::ConstrainedVector, y::AbstractArray{<:Real,1}) = x.x + y
Base.:+(y::AbstractArray{<:Real,1}, x::ConstrainedVector) = x.x + y
Base.:+(x::ConstrainedVector, y::Real) = x.x .+ y
Base.:+(y::Real, x::ConstrainedVector) = x.x .+ y
Base.:+(y::ConstrainedVector, x::ConstrainedVector) = x.x + y.x
Base.:-(x::ConstrainedVector, y::AbstractArray{<:Real,1}) = x.x - y
Base.:-(y::AbstractArray{<:Real,1}, x::ConstrainedVector) = y - x.x
Base.:-(x::ConstrainedVector, y::Real) = x.x .- y
Base.:-(y::Real, x::ConstrainedVector) = y .- x.x
Base.:-(x::ConstrainedVector, y::ConstrainedVector) = x.x - y.x
Base.:*(A::AbstractArray{<:Real,2}, x::ConstrainedVector) = A * x.x
Base.:*(x::ConstrainedVector, A::AbstractArray{<:Real,2}) = x.x * A
Base.convert(::Type{Vector}, A::ConstrainedVector) = A.x
Base.show(io::IO, ::MIME"text/plain", Θ::ConstrainedVector) = print(Θ.x)
Base.size(x::ConstrainedVector) = size(x.x)
Base.getindex(x::ConstrainedVector, i::Int) = x.x[i]


type_length(::Type{PositiveVector{p}}) where p = p
length(::PositiveVector{p}) where p = p
length(::PosiVec{p}) where p = p
type_length(::Type{ProbabilityVector{p}}) where p = p
length(::ProbabilityVector{p}) where p = p
length(::ProbVec{p}) where p = p
type_length(::Type{RealVector{p}}) where p = p
length(::RealVector{p}) where p = p
length(::RealVec{p}) where p = p
type_length(::Type{Simplex{p,q}} where p) where q = q
length(::Simplex{p,q} where p) where q = q
length(::Simp{p,q} where p) where q = q

function update!(x::PosiVec{p,T,V,D} where {T <: Real,V<:mutable_vector,D}) where {p}
  @inbounds for i ∈ 1:p
    x.x[i] = exp(x.Θ[i])
  end
end
log_jacobian(x::PosiVec) = sum(x.Θ)

Base.getindex(x::PosiVec, i::Int) = exp(x.Θ[i])
function Base.setindex!(x::PosiVec{p,T,V,D} where {p,T,V <: mutable_vector, D <: mutable_vector}, v::Real, i::Int)
  x.x[i] = v
  x.Θ[i] = log(v)
end


@inline PosiVec(v::V, x::D, ::Type{Val{p}}) where {p, T, P <: AbstractArray{T,1}, V <: VectorView{T,P}, D <: StaticArray{Tuple{p}, T, 1} } = PosiVec{p, T, P, V, D}(v, x)
function construct(::PositiveVector{p}, Θv::V, i::Int) where {p, T, V <: mutable_vector{T}}
  v = view( Θv, 1+i-p:i )
  PosiVec(v, MVector{p}( exp.(v) ), Val{p})
end
function construct(::PositiveVector{p}, Θv::SVector{q,T}, i::Int) where {p,q,T}
  v = view( Θv, 1+i-p:i )
  PosiVec(v, SVector{p,T}(exp.(v)), Val{p})
end
function construct(::PositiveVector{p}, Θv::V, i::Int, vals::A) where {p, T, V <: mutable_vector{T}, A <: AbstractArray{T,1}}
  pv = PosiVec(view( Θv, 1+i-p:i ), MVector{p}(vals), Val{p})
  pv.Θ .= log.(pv.x)
  pv
end


function update!(x::ProbVec{p}) where {p}
  @inbounds for i ∈ 1:p
    x.x[i] = logistic(x.Θ[i])
  end
end
lj(x::Real) = log(x) + log(1 - x)
log_jacobian(x::ProbVec) = sum(lj, x.x)

Base.getindex(x::ProbVec, i::Int) = x.x[i]
function Base.setindex!(x::ProbVec{p,T,V,D} where {p,T,V <: mutable_vector,D<:mutable_vector}, v::Real, i::Int)
  x.x[i] = v
  x.Θ[i] = logit(v)
end
@inline ProbVec(v::V, x::D, ::Type{Val{p}}) where {p, T, P <: AbstractArray{T,1}, V <: VectorView{T,P}, D <: StaticArray{Tuple{p}, T, 1} } = ProbVec{p, T, P, V, D}(v, x)
function construct(::ProbabilityVector{p}, Θv::V, i::Int) where {p, T, V <: mutable_vector{T}}
  v = view(Θv, 1+i-p:i)
  ProbVec(v, MVector{p}( logistic.(v) ), Val{p})
end
function construct(::ProbabilityVector{p}, Θv::SVector{q,T}, i::Int) where {p, q, T}
  v = view(Θv, 1+i-p:i)
  ProbVec(v, SVector{p}( logistic.(v) ), Val{p})
end

function construct(::ProbabilityVector{p}, Θv::V, i::Int, vals::A) where {p, T, V <: mutable_vector{T}, A <: AbstractArray{T,1}}
  pv = ProbVec(view(Θv, 1+i-p:i), MVector{p}(vals), Val{p})
  pv.Θ .= logit.(vals)
  pv
end

function Base.setindex!(x::RealVec, v::Real, i::Int)
  x.x[i] = v
end
function update!(x::RealVec)
end
@generated log_jacobian(x::RealVec{p, T} where {p}) where {T} = zero(T)

@inline RealVec(v::V, ::Type{Val{p}}) where {p, T, P <: AbstractVector{T}, V <: VectorView{T,P}} = RealVec{p, T, P <: AbstractVector{T}, V <: VectorView{T,P}}(v)
function construct(::Type{RealVector{p}}, Θ::V, i::Int) where {p, T, V <: mutable_vector{T}}
  RealVec(view(Θ, 1+i-p:i), Val{p})
end
function construct(::Type{RealVector{p}}, Θ::SVector{q,T}, i::Int) where {p, q, T}
  RealVec(view(Θ, 1+i-p:i), Val{p})
end
function construct(::Type{RealVector{p}}, Θv::V, i::Int, vals::A) where {p, T, V <: mutable_vector{T}, A <: AbstractArray{T,1}}
  rv = RealVec(view(Θv, 1+i-p:i), Val{p})
  copy!(rv.x, vals)
  rv
end


#Simplex(Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}) where {T} = Simplex{length(x), T}(Θ, MVector{length(x)}(SimplexTransform.(Θ)))
function update!(x::Simp{p, q, T, V, D, R} where {T <: Real, V, D <: mutable_vector, R <: mutable_vector}) where {p, q}
  @inbounds for i ∈ 1:q
    x.z[i] = logistic(x.Θ[i] - log( p - i ) )
  end
  x.csx[1] = x.x[1] = x.z[1]
  @inbounds for i ∈ 2:q
    x.x[i] = (1 - x.csx[i-1]) * x.z[i]
    x.csx[i] = x.x[i] + x.csx[i-1]
  end
  x.x[end] = 1 - x.csx[end]
end
function log_jacobian(x::Simp{p, q, T, V, D, R} where {p,T,V,D,R}) where {q}
  out = log(x.z[1]) + log(1 - x.z[1])
  @inbounds for i ∈ 2:q
    out += log(x.z[i]) + log(1 - x.z[i]) + log(1 - x.csx[i - 1])
  end
  out
end

Base.getindex(x::Simp, i::Int) = x.x[i]

#Using setindex! is strongly discouraged.

function Base.setindex!(x::Simp, v::Vector, i::Int)
  x.x .*= (1 .- v) ./ (1 .- x.x[i])
  x.x[i] = v
  set_prob!(x)
end
function set_prob!(x::Simp, π::AbstractVector)
  x.x .= π
  set_prob!(x)
end
function set_prob!(x::Simp{p,q,T,V,D,R} where {T,V<:mutable_vector,D<:mutable_vector,R<:mutable_vector}) where {p,q}
  x.csx[1] = x.z[1] = x.x[1]
  y[i] = logit(x.z[1]) + log(p - 1)
  @inbounds for i ∈ 2:q
    x.csx[i] = x.csx[i-1] + x.x[i]
    x.z[i] = x.x[i] / (1 - x.csx[i-1])
    y[i] = logit(x.z[i]) + log(p - i)
  end
end
@inline function Simp(::Type{Val{p}}, ::Type{Val{q}}, Θv::V, i::Int) where {p, q, T, P, V <: VectorView{T,P}}
    Simp{p, q, T, P, V, MVector{p,T}, MVector{q,T}}(view(Θv, 1+i-q:i), MVector{p,T}(Vector{T}(p)), MVector{q,T}(Vector{T}(q)), MVector{q,T}(Vector{T}(q)))
end
function construct(::Type{Simplex{p, q}}, Θv::V, i::Int) where {p, q, V}
    s = Simp(Val{p}, Val{q}, Θv, i)
    update!(s)
    s
end
function construct(::Simplex{p,q}, Θv::V, i::Int, vals::A) where {p, q, T, V <: mutable_vector{T}, A <: AbstractArray{T}}
  out = Simp(Val{p}, Val{q}, Θv, i)
  set_prob!(out, vals)
  out
end
#struct LowerBoundVector{p, T}
#  Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
#  x::Vector{T}
#  L::Vector{T}
#end
