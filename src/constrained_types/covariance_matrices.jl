struct CovarianceMatrix{p,l,o} <: CTV{p} end
Base.@pure CovarianceMatrix(p) = CovarianceMatrix{p,div(p*(p+1),2),div(p*(p-1),2)}()
Base.length(::CovarianceMatrix{p,l,o} where {p,o}) where l = l

abstract type UpperTriangle{p,T} <: AbstractArray{T,2} end
struct UpperTriangleView{p,T,P <: AbstractVector{T}, V <: VectorView{T,P}} <: UpperTriangle{p,T}
  diag::MVector{p,T}
  off_diag::V
end
struct UpperTriangleVector{p,T,o} <: UpperTriangle{p,T}
  diag::MVector{p,T}
  off_diag::MVector{o,T}
end

struct CovMat{p, l, T <: Real, P <: AbstractVector{T}, V <: VectorView{T,P}, o} <: SquareMatrix{p, T}
  Λ::V#length p
  U::UpperTriangleVector{p,T,o}
  Σ::Symmetric{T,SizedArray{Tuple{p,p},T,2,2}}
  U_inverse::UpperTriangleView{p,T,P,V}
end
length(::CovMat{p,l} where p) where l = l
length(::CovarianceMatrix{p,l} where p) where l = l

@generated off_diag_count(::Type{Val{p}}) where p = Val{div(p*(p-1),2)}
@generated val_square(::Type{Val{p}}) where p = Val{abs2(p)}

Base.IndexStyle(::UpperTriangle) = IndexLinear()
Base.getindex(A::UpperTriangle, i::Int) = A.off_diag[i]
function Base.setindex!(A::UpperTriangle, v, i::Int)
  @inbounds A.off_diag[i] = v
end
function Base.size(::UpperTriangle{p}) where {p}
  (p,p)
end
sub2triangle(i_1::Int, i_2::Int) = i_1 + div(i_2*(i_2-1),2)
function Base.getindex(A::UpperTriangle, i_1::Int, i_2::Int)
  @inbounds i_1 == i_2 ? A.diag[i_1] : A.off_diag[sub2triangle(i_1, i_2-1)]
end
function Base.setindex!(A::UpperTriangle{p,T}, v::T, i_1::Int, i_2::Int) where {T,p}
  if i_1 == i_2
    @inbounds A.diag[i_1] = v
  else
    @inbounds A.off_diag[sub2triangle(i_1, i_2-1)] = v
  end
end

#Only updates U_inverse.
#This is all that is needed for most probability density functions.
#In case someone wants to access any other representations, they must explicitly call a function to evaluate it.
#May eventually implement some sort of lazy evaluation, eg Tim Holy's MappedArrays.
function update!(Θ::CovMat{p}) where {p}
  @inbounds for i ∈ 1:p
    Θ.U_inverse.diag[i] = exp(Θ.Λ[i])
  end
end
function construct(::CovarianceMatrix{p,l,o}, Θv::V, i::Int, CovMat::SizedArray{Tuple{p,p},T,2,2} = Symmetric(SizedArray{Tuple{p,p},T,2,2}(Array{T,2}(p,p)))) where {p, T, V <: AbstractVector{T}, l, o}
  ind_end = p-l+i
  Λ = view(Θv, 1-l+i:ind_end)
  U = UpperTriangleVector{p,T}(MVector{p}(Vector{T}(p)), MVector{o,T}(Vector{T}(o)))
  U_inverse = UpperTriangleView{p,T}(MVector{p,T}(exp.(Λ)), view(Θv, 1+ind_end:i))
  CovMat{p, l, T, P, typeof(Λ), o}(Λ, U, CovMat, U_inverse)
end


function construct(CM::CovarianceMatrix{p,l,o}, Θv::Vector{T}, i::Int, CovMat::Array{T,2}) where {p, T}
  construct(CM, Θv, i, Symmetric(CovMat))
end
function construct(CM::CovarianceMatrix, Θv::Vector{T}, i::Int, CovMat::Symmetric{T,Array{T,2}}) where {p, T}
  Θ = CovMat.uplo != 'U' ? construct(CM, Θv, i, Symmetric(CovMat.data')) : construct(CM, Θv, i, CovMat)
  set_Σ!(Θ)
  Θ
end


function log_jacobian!(Θ::CovMat{p, l, T} where l) where {p, T}
  l_jac = zero(T)
  @inbounds for i ∈ 1:p
    l_jac += (i - 2p - 1) * Θ.Λ[i]
  end
  l_jac
end
function chol!(U::UpperTriangle{p}, Σ::Symmetric) where {p}
  @inbounds for i ∈ 1:p
    U[i,i] = Σ[i,i]
    for j ∈ 1:i-1
      U[j,i] = Σ[j,i]
      for k ∈ 1:j-1
        U[j,i] -= U[k,i] * U[k,j]
      end
      U[j,i] /= U[j,j]
      U[i,i] -= U[j,i]^2
    end
    U[i,i] = √U[i,i]
  end
end
###This happens when someone sets an index of the covariance matrix.
calc_U_from_Σ!(Θ::CovMat) = chol!(Θ.U, Θ.Σ)
function calc_Σij!(Θ::CovMat, i::Int, j::Int)
  @inbounds Θ.Σ.data[j,i] = Θ.U[1,i] * Θ.U[1,j]
  @inbounds for k ∈ 2:j
    Θ.Σ.data[j,i] += Θ.U[k,i] * Θ.U[k,j]
  end
end
function calc_Σ!(Θ::CovMat{p}) where p
  for i ∈ 1:p, j ∈ 1:i
    calc_Σij!(Θ, i, j)
  end
end
function calc_invΣij(Θ::CovMat{p,T}, i::Int, j::Int) where {p,T}
  out = zero(T)
  @inbounds for k ∈ i:p
    out += Θ.U_inverse[i,k] * Θ.U_inverse[j,k]
  end
  out
end
function inv!(U_inverse::UpperTriangle{p}, U::UpperTriangle{p}) where p
  @inbounds for i ∈ 1:p
    U_inverse.diag[i] = 1 / U.diag[i]
    for j ∈ i+1:p
      triangle_index = sub2triangle(i,j-1)
      U_inverse.off_diag[triangle_index] = U[i,j] * U_inverse.diag[i]
      for k ∈ i+1:j-1
        U_inverse.off_diag[triangle_index] += U[k,j] * U_inverse[i,k]
      end
      U_inverse.off_diag[triangle_index] /= -U.diag[j]
    end
  end
end
function calc_U_inverse_from_U!(Θ::CovMat)
  inv!(Θ.U_inverse, Θ.U)
  Θ.Λ .= log.(Θ.U_inverse.diag)
end
function calc_U_from_U_inverse!(Θ::CovMat)
  inv!(Θ.U, Θ.U_inverse)
end
function set_Σ!(Θ::CovMat)
  calc_U_from_Σ!(Θ)
  calc_U_inverse_from_U!(Θ)
end
function update_Σ!(Θ::CovMat)
  calc_U_from_U_inverse!(Θ)
  calc_Σ!(Θ)
end

#Note, accessing the covariance matrix brings you here, where you calculate Σij; if you want access to the cached value you need to reference Θ.Σ[i,j]. Note that the cache is not updated often.
function Base.getindex(Θ::CovMat, i::Int, j::Int)
  i > j ? calc_Σij!(Θ,i, j) : calc_Σij!(Θ, j, i)
end
function Base.getindex(Θ::CovMat{p}, k::Int) where p
  Θ[ind2sub((p,p), k)...]
end
#Strongly discouraged from calling the following method. But...if you have to, it is here.
function Base.setindex!(Θ::CovMat{p}, v::T, k::Int) where p
  Θ[ind2sub((p,p), k)] = v
end
function Base.setindex!(Θ::CovMat, v::T, i::Int, j::Int)
  update_Σ!(Θ)
  i > j ? setindex!(Θ.Σ.data, v, j, i) : setindex!(Θ.Σ.data, v, i, j)
  set_Σ!(Θ)
end


function quad_form(x::AbstractVector, Θ::CovMat{p}) where p
  out = (x[1] * Θ.U_inverse.diag[1])^2
  @inbounds for i ∈ 2:p
    dot_prod = x[i] * Θ.U_inverse.diag[i]
    triangle_index = div((i-1)*(i-2),2)
    for j ∈ 1:i-1
      dot_prod += x[j] * Θ.U_inverse.off_diag[triangle_index + j]
    end
    out += dot_prod^2
  end
  out
end
function trace_AΣinv(A::AbstractArray{<:Real,2}, Σ::CovMat)
  2*htrace_AΣinv(A,Σ)
end
function htrace_AΣinv(A::AbstractArray{<:Real,2}, Σ::CovMat{p}) where p
  out = zero(T)
  @inbounds for i ∈ 1:p
    out += A[i,i] * calc_invΣij(Σ, i, i) / 2
    for j ∈ 1:i-1
      out += A[j,i] * calc_invΣij(Σ, i, j)
    end
  end
  out
end
Base.det(U::UpperTriangle) = prod(U.diag)
Base.logdet(U::UpperTriangle) = sum(log, U.diag)
Base.trace(U::UpperTriangle) = sum(U.diag)
Base.det(Θ::CovMat) = det(Θ.U_inverse)^-2
Base.logdet(Θ::CovMat) = 2hlogdet(Θ)
inv_det(Θ::CovMat) = det(Θ.U_inverse)^2
inv_root_det(Θ::CovMat) = det(Θ.U_inverse)
root_det(Θ::CovMat) = 1/det(Θ.U_inverse)
hlogdet(Θ::CovMat) = -sum(Θ.Λ)
nhlogdet(Θ::CovMat) = sum(Θ.Λ)
function trace_inverse(Θ::CovMat{p,l,T} where l) where {p,T}
  out = zero(T)
  for i ∈ 1:p
    out += calc_invΣij(Θ, i, i)
  end
  out
end
function Base.:+(Θ::CovMat, A::AbstractArray{<: Real,2})
  update_Σ!(Θ)
  Θ.Σ + A
end
function Base.:+(A::AbstractArray{<: Real,2}, Θ::CovMat)
  update_Σ!(Θ)
  Θ.Σ + A
end
#Would you want to output a regular matrix, a symmetric matrix, or a covariance matrix?
function Base.:+(Θ_1::CovMat{p}, Θ_2::CovMat{p}) where p
  update_Σ!(Θ_1)
  update_Σ!(Θ_2)
  #CovMat(T, p, Θ_1.Σ + Θ_2.Σ)
  Θ_1.Σ + Θ_2.Σ
end
function Base.:*(Θ::CovMat, A::AbstractArray)
  update_Σ!(Θ)
  Θ.Σ * A
end
function Base.:*(A::AbstractArray, Θ::CovMat)
  update_Σ!(Θ)
  A * Θ.Σ
end
function Base.:*(Θ_1::CovMat{p}, Θ_2::CovMat{p}) where p
  update_Σ!(Θ_1)
  update_Σ!(Θ_2)
  #CovMat(T, p, Θ_1.Σ + Θ_2.Σ)
  Θ_1.Σ * Θ_2.Σ
end
function Base.show(io::IO, ::MIME"text/plain", Θ::CovMat)
  update_Σ!(Θ)
  println(Θ.Σ)
end
type_length(::Type{CovMat{p, l}} where p) where l = l
param_type_length(::Type{CovMat{p,l}} where p) where l = Val{l}
function convert(::Type{Symmetric}, A::CovMat{p,T}) where {p,T}
  update_Σ!(A)
  A.Σ
end
function convert(::Type{Array{T,2}}, A::CovMat{p,l,T} where {p,l}) where T
    update_Σ!(A)
    convert(Array{T,2}, A.Σ)
end
