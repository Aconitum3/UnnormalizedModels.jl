### Basic Statistics
meanform(η::Vector{<:Real}) = VonMisesFisher(-vec(η))

t(D::Type{VonMisesFisher},x) = x
jac_t(D::Type{VonMisesFisher},x) = I(length(x))
hess_t(D::Type{VonMisesFisher},x) = zeros(length(x),length(x)^2)

fit_sm(D::Type{VonMisesFisher},X;w::Union{Function, Nothing}=nothing) = _SSM_ExpFam_core(x -> jac_t(D,x),x -> hess_t(D,x),X,w=w) |> meanform

fit_smom(D::Type{VonMisesFisher},jac_f,hess_f,X) = _SSMoM_ExpFam_core(x -> jac_t(D,x),jac_f,hess_f,X) |> meanform
fit_smom(D::Type{VonMisesFisher},f,X) = _SSMoM_ExpFam_core(x -> jac_t(D,x),f,X) |> meanform