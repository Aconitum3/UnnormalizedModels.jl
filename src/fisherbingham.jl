abstract type AbstractFisherBingham <: ContinuousMultivariateDistribution end

struct FisherBingham{T<:Real} <: AbstractFisherBingham
    μ::Vector{T}
    A::Symmetric{T}
end

function FisherBingham(μ::AbstractVector{T},A::AbstractMatrix{T}) where {T<:Real}
    size(A,1) == Base.length(μ) || throw(DimensionMismatch("The dimensions of mu and A are inconsistent."))
    issymmetric(A) || error("SymmetricException: A is not Symmetric.")
    return FisherBingham(μ,Symmetric(A))
end

function FisherBingham(μ::AbstractVector{<:Real},A::AbstractMatrix{<:Real})
    R = Base.promote_eltype(μ,A)
    return FisherBingham(convert(AbstractArray{R}, μ), convert(AbstractMatrix{R}, A))
end


# Fisher-Bingham Distribution with A_{d,d}=0
struct ZeroCornerFisherBingham{T<:Real} <: AbstractFisherBingham
    μ::Vector{T}
    A::Symmetric{T}
end

function ZeroCornerFisherBingham(μ::AbstractVector{T},A::AbstractMatrix{T}) where {T<:Real}
    d = size(A,1)
    d == Base.length(μ) || throw(DimensionMismatch("The dimensions of mu and A are inconsistent."))
    issymmetric(A) || error("SymmetricException: A is not Symmetric.")
    A[d,d] == 0 || error("ZeroCornerException: A[d,d] must be Zero.")
    return ZeroCornerFisherBingham(μ,Symmetric(A))
end

function ZeroCornerFisherBingham(μ::AbstractVector{<:Real},A::AbstractMatrix{<:Real})
    R = Base.promote_eltype(μ,A)
    return ZeroCornerFisherBingham(convert(AbstractArray{R}, μ), convert(AbstractMatrix{R}, A))
end

struct FisherBinghamCanon{T<:Real} <: AbstractFisherBingham
    η::Vector{T}
end

struct ZeroCornerFisherBinghamCanon{T<:Real} <: AbstractFisherBingham
    η::Vector{T}
end

### Basic statistics

params(d::FisherBingham) = (d.μ,d.A)
params(d::ZeroCornerFisherBingham) = (d.μ,d.A)

function meanform(F::FisherBinghamCanon)
    η = F.η
    m = length(η)
    d = (isqrt(1+8m)-2)÷2
    μ = -η[1:d]
    A = -avech(η[d+1:end]) .* (ones(d,d) + I(d))/2
    return FisherBingham(μ,A)
end

function meanform(F::ZeroCornerFisherBinghamCanon)
    η = F.η
    m = length(η) + 1
    d = (isqrt(1+8m)-2)÷2
    μ = -η[1:d]
    A = -avech([η[d+1:end];0]) .* (ones(d,d) + I(d))/2
    return FisherBingham(μ,A)
end

t(D::Type{FisherBingham},x) = t(MvNormal,x)
t(D::Type{ZeroCornerFisherBingham},x) = t(FisherBingham,x)[1:end-1]
jac_t(D::Type{FisherBingham},x) = jac_t(MvNormal,x)
jac_t(D::Type{ZeroCornerFisherBingham},x) = jac_t(FisherBingham,x)[1:end-1,:]

hess_t(D::Type{FisherBingham},x) = hess_t(MvNormal,x)
hess_t(D::Type{ZeroCornerFisherBingham},x) = hess_t(FisherBingham,x)[1:end-1,:]

function fit_sm(D::Type{FisherBingham},X;w::Union{Function, Nothing}=nothing)
    η = _SSM_ExpFam_core(x -> jac_t(D,x),x -> hess_t(D,x),X,w=w)
    return FisherBinghamCanon(η) |> meanform
end

function fit_smom(D::Type{FisherBingham},jac_f,hess_f,X)
    η = _SSMoM_ExpFam_core(x -> jac_t(D,x),jac_f,hess_f,X)
    return FisherBinghamCanon(η) |> meanform
end

fit_smom(D::Type{FisherBingham},f,X) = _SSMoM_ExpFam_core(x -> jac_t(D,x),f,X) |> FisherBinghamCanon |> meanform

function fit_sm(D::Type{ZeroCornerFisherBingham},X;w::Union{Function, Nothing}=nothing)
    η = _SSM_ExpFam_core(x -> jac_t(D,x),x -> hess_t(D,x),X,w=w)
    return ZeroCornerFisherBinghamCanon(η) |> meanform
end

function fit_smom(D::Type{ZeroCornerFisherBingham},jac_f,hess_f,X)
    η = _SSMoM_ExpFam_core(x -> jac_t(D,x),jac_f,hess_f,X)
    return ZeroCornerFisherBinghamCanon(η) |> meanform
end

fit_smom(D::Type{ZeroCornerFisherBingham},f,X) = _SSMoM_ExpFam_core(x -> jac_t(D,x),f,X) |> ZeroCornerFisherBinghamCanon |> meanform



