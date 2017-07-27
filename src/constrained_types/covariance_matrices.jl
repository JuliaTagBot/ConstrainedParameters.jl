abstract type UpperTriangle{p,T} <: AbstractArray{T,2} end
struct UpperTriangleView{p,T} <: UpperTriangle{p,T}
  diag::MVector{p,T}
  off_diag::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
end
struct UpperTriangleVector{p,T} <: UpperTriangle{p,T}
  diag::MVector{p,T}
  off_diag::Vector{T}
end

struct CovarianceMatrix{p, T <: Real} <: SquareMatrix{p, T}
  Λ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}#length p
  U::UpperTriangleVector{p,T}
  Σ::Symmetric{T,Array{T, 2}}
  U_inverse::UpperTriangleView{p,T}
end

Base.IndexStyle(::UpperTriangle) = IndexLinear()
Base.getindex(A::UpperTriangle, i::Int) = A.off_diag[i]
function Base.setindex!(A::UpperTriangle, v, i::Int)
  A.off_diag[i] = v
end
function Base.size(::UpperTriangle{p,Float64}) where {p}
  (p,p)
end
sub2triangle(i_1::Int, i_2::Int) = i_1 + round(Int, i_2*(i_2-1)/2)
function Base.getindex(A::UpperTriangle, i_1::Int, i_2::Int)
  if i_1 == i_2
    A.diag[i_1]
  else
    A.off_diag[sub2triangle(i_1, i_2-1)]
  end
end
function Base.setindex!(A::UpperTriangle{p,T}, v::T, i_1::Int, i_2::Int) where {T,p}
  if i_1 == i_2
    A.diag[i_1] = v
  else
    A.off_diag[sub2triangle(i_1, i_2-1)] = v
  end
end


function update_U_inverse!(Θ::CovarianceMatrix{p,T} where {T<:Real}) where {p}
  for i ∈ 1:p
    Θ.U_inverse.diag[i] = exp(Θ.Λ[i])
  end
end
function build(A::Type{CovarianceMatrix{p,T}}, Θv::Vector{T}, i::Int, CovMat::Symmetric{T,Array{T,2}} = Symmetric(Array{T}(p,p))) where {p, T}
  Λ = view(Θv, i + (1:p))
  U = UpperTriangleVector{p,T}(MVector{p}(Vector{T}(p)), Vector{T}(round(Int,type_length(A))))
  U_inverse = UpperTriangleView{p,T}(MVector{p}(exp.(Λ)), view(Θv, i + (1+p:type_length(A))))
  CovarianceMatrix{p, T}(Λ, U, CovMat, U_inverse)
end
function construct(A::Type{CovarianceMatrix{p,T}}, Θv::Vector{T}, i::Int, CovMat::Array{T,2}) where {p, T}
  construct(A, Θv, i, Symmetric(CovMat))
end
function construct(A::Type{CovarianceMatrix{p,T}}, Θv::Vector{T}, i::Int, CovMat::Symmetric{T,Array{T,2}}) where {p, T}
  if CovMat.uplo != 'U'
    CovMat = Symmetric(CovMat.data')
  end
  Θ = build(A, Θv, i, CovMat)
  set_Σ!(Θ)
  Θ
end
construct(A::Type{CovarianceMatrix{p,T}}, Θv::Vector{T}, i::Int) where {p, T} = build(A, Θv, i)


#@generated function Base.length{p,T}(::Type{CovarianceMatrix{p,T}})
#  round(Int, p*(p+1)/2)
#end
update!(Θ::CovarianceMatrix) = update_U_inverse!(Θ)
function log_jacobian!(Θ::CovarianceMatrix{p, T}) where {p, T}
  l_jac = zero(T)
  for i ∈ 1:p
    l_jac += (i - 2p - 1) * Θ.Λ[i]
  end
  l_jac
end
function chol!(U::UpperTriangle{p,T}, Σ::Symmetric{T,Array{T, 2}}) where {p,T}
  for i ∈ 1:p
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
function calc_U_from_Σ!(Θ::CovarianceMatrix{p, T}) where {p,T}
  chol!(Θ.U, Θ.Σ)
end
function calc_Σij!(Θ::CovarianceMatrix, i::Int, j::Int)
  Θ.Σ.data[j,i] = Θ.U[1,i] * Θ.U[1,j]
  for k ∈ 2:j
    Θ.Σ.data[j,i] += Θ.U[k,i] * Θ.U[k,j]
  end
  Θ.Σ.data[j,i]
end
function calc_Σ!(Θ::CovarianceMatrix{p,<:Real}) where {p}
  for i ∈ 1:p, j ∈ 1:i
    calc_Σij!(Θ, i, j)
  end
end
function calc_invΣij(Θ::CovarianceMatrix{p,T}, i::Int, j::Int) where {p,T}
  out = zero(T)
  for k ∈ i:p
    out += Θ.U_inverse[i,k] * Θ.U_inverse[j,k]
  end
  out
end
function inv!(U_inverse::UpperTriangle{p,T}, U::UpperTriangle{p,T}) where {p,T}
  for i ∈ 1:p
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
function calc_U_inverse_from_U!(Θ::CovarianceMatrix)
  inv!(Θ.U_inverse, Θ.U)
  Θ.Λ .= log.(Θ.U_inverse.diag)
end
function calc_U_from_U_inverse!(Θ::CovarianceMatrix)
  inv!(Θ.U, Θ.U_inverse)
end
function set_Σ!(Θ::CovarianceMatrix)
  calc_U_from_Σ!(Θ)
  calc_U_inverse_from_U!(Θ)
end
function update_Σ!(Θ::CovarianceMatrix)
  calc_U_from_U_inverse!(Θ)
  calc_Σ!(Θ)
end

#Note, accessing the covariance matrix brings you here, where you calculate Σij; if you want access to the cached value you need to reference Θ.Σ[i,j]. Note that the cache is not updated often.
function Base.getindex(Θ::CovarianceMatrix, i::Int, j::Int)
  i > j ? calc_Σij!(Θ,i, j) : calc_Σij!(Θ, j, i)
end
function Base.getindex(Θ::CovarianceMatrix{p,T}, k::Int) where {p,T}
  Θ[ind2sub((p,p), k)...]
end
#Strongly discouraged from calling the following method. But...if you have to, it is here.
function Base.setindex!(Θ::CovarianceMatrix{p,T}, v::T, k::Int) where {p,T}
  Θ[ind2sub((p,p), k)] = v
end
function Base.setindex!(Θ::CovarianceMatrix{p,T}, v::T, i::Int, j::Int) where {p,T}
  update_Σ!(Θ)
  i > j ? setindex!(Θ.Σ.data, v, j, i) : setindex!(Θ.Σ.data, v, i, j)
  set_Σ!(Θ)
end


function quad_form(x::Vector{Real}, Θ::CovarianceMatrix)
  out = (x[1] * Θ.U_inverse.diag[1])^2
  for i ∈ 2:p
    dot_prod = x[i] * Θ.U_inverse.diag[i]
    triangle_index = round(Int, (i-1)*(i-2)/2)
    for j ∈ 1:i-1
      dot_prod += x[j] * Θ.U_inverse.off_diag[triangle_index + j]
    end
    out += dot_prod^2
  end
  out
end
function trace_AΣinv(A::AbstractArray{<:Real}, Σ::CovarianceMatrix{p,T}) where {p,T}
  2*htrace_AΣinv(A,Σ)
end
function htrace_AΣinv(A::AbstractArray{<:Real}, Σ::CovarianceMatrix{p,T}) where {p,T}
  out = zero(T)
  for i ∈ 1:p
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
Base.det(Θ::CovarianceMatrix) = det(Θ.U_inverse)^-2
Base.logdet(Θ::CovarianceMatrix) = 2hlogdet(Θ)
inv_det(Θ::CovarianceMatrix) = det(Θ.U_inverse)^2
inv_root_det(Θ::CovarianceMatrix) = det(Θ.U_inverse)
root_det(Θ::CovarianceMatrix) = 1/det(Θ.U_inverse)
hlogdet(Θ::CovarianceMatrix) = -sum(Θ.Λ)
nhlogdet(Θ::CovarianceMatrix) = sum(Θ.Λ)
function trace_inverse(Θ::CovarianceMatrix{p,T}) where {p,T}
  out = 0
  for i ∈ 1:p
    out += calc_invΣij(Θ, i, i)
  end
  out
end
function Base.:+(Θ::CovarianceMatrix, A::AbstractArray{Real,2})
  update_Σ!(Θ)
  Θ.Σ + A
end
function Base.:+(A::AbstractArray{Real,2}, Θ::CovarianceMatrix)
  update_Σ!(Θ)
  Θ.Σ + A
end
#Would you want to output a regular matrix, a symmetric matrix, or a covariance matrix?
function Base.:+(Θ_1::CovarianceMatrix{p,T}, Θ_2::CovarianceMatrix{p,T}) where {p,T}
  update_Σ!(Θ_1)
  update_Σ!(Θ_2)
  #CovarianceMatrix(T, p, Θ_1.Σ + Θ_2.Σ)
  Θ_1.Σ + Θ_2.Σ
end
function Base.:*(Θ::CovarianceMatrix, A::AbstractArray)
  update_Σ!(Θ)
  Θ.Σ * A
end
function Base.:*(A::AbstractArray, Θ::CovarianceMatrix)
  update_Σ!(Θ)
  A * Θ.Σ
end
function Base.:*(Θ_1::CovarianceMatrix{p,T}, Θ_2::CovarianceMatrix{p,T}) where {p,T}
  update_Σ!(Θ_1)
  update_Σ!(Θ_2)
  #CovarianceMatrix(T, p, Θ_1.Σ + Θ_2.Σ)
  Θ_1.Σ * Θ_2.Σ
end
function Base.show(io::IO, ::MIME"text/plain", Θ::CovarianceMatrix)
  update_Σ!(Θ)
  println(Θ.Σ)
end
@generated type_length(::Type{CovarianceMatrix{p,T}}) where {p,T} = round(Int, p * (p+1)/2)
function convert(::Type{Symmetric{T,Array{T,2}}}, A::CovarianceMatrix{p,T}) where {p,T}
  update_Σ!(A)
  A.Σ
end
function convert(::Type{Array{T,2}}, A::CovarianceMatrix{p,T}) where {p,T}
  convert(Array{T,2}, convert(Symmetric{T,Array{T,2}}, A))
end
