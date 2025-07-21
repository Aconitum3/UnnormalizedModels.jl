# Stein's Method of Moments 

# Estimator for the canonical parameter eta of Exponential Family on $\mathbb{R}^d$
function _SMoM_ExpFam_core(jac_t::Function,jac_f::Function,hess_f::Function,X::AbstractMatrix)
    d,n = size(X)
    @assert n > 0 "X is empty."

    m = size(jac_f(X[:,1]), 1)
    A = zeros(m,m)
    b = zeros(m)

    lap_f(x) = vec(mapslices(v -> tr(reshape(v,d,d)),hess_f(x);dims=2))
    for i in 1:n
        x = X[:,i]
        A += jac_f(x) * jac_t(x)'
        b += lap_f(x)
    end
    
    return vec(A \ b)
end

function _SMoM_ExpFam_core(jac_t::Function,f::Function,X::AbstractMatrix)
    n = size(X,2)
    @assert n > 0 "X is empty."
    m = size(jac_t(X[:,1]),1)

    jac_f(x) = jacobian(f,x)[1]
    hess_f(x) = reduce(vcat,[vec(hessian(u -> f(u)[i],x))' for i in 1:m])
    
    return _SMoM_ExpFam_core(jac_t,jac_f,hess_f,X)
end

function SMoM_ExpFam(f,t,X)
    n = size(X,2)
    @assert n > 0 "X is empty."

    jac_t(x) = jacobian(t,x)[1]
    return _SMoM_ExpFam_core(jac_t,f,X)
end

# Estimator for the canonical parameter eta of Exponential Family on $\mathbb{S}^{d-1}$
function _SSMoM_ExpFam_core(jac_t::Function,jac_f::Function,hess_f::Function,X::AbstractMatrix)
    d, n = size(X)
    @assert n > 0 "X is empty."

    m = size(jac_f(X[:,1]), 1)
    
    P(u) = I(d) - u*u'
    
    A = zeros(m,m)
    b = zeros(m)
    for i in 1:n
        x = X[:,i]
        
        jac_f_val = jac_f(x)
        hess_f_val = hess_f(x)
        lap_f_val = mapslices(v -> tr(reshape(v,d,d)),hess_f_val;dims=2)
        
        
        A += jac_f_val * P(x) * jac_t(x)'
        b += (1-d) * jac_f_val * x - hess_f_val * kron(x,x) + lap_f_val
    end
    
    return vec(A \ b)
end

function _SSMoM_ExpFam_core(jac_t::Function,f::Function,X)
    n = size(X,2)
    @assert n > 0 "X is empty."
    
    m = size(jac_t(X[:,1]),1)

    jac_f(x) = jacobian(f,x)[1]
    hess_f(x) = reduce(vcat,[vec(hessian(u -> f(u)[i],x))' for i in 1:m])
    
    return _SSMoM_ExpFam_core(jac_t,jac_f,hess_f,X)
end

function SSMoM_ExpFam(f,t,X)
    n = size(X,2)
    @assert n > 0 "X is empty."
    
    jac_t(x) = jacobian(t,x)[1]
    
    return _SSMoM_ExpFam_core(jac_t,f,X)
end

function fit_smom(D::Type{<:ContinuousDistribution},w,∇w,hess_w,X)
    function hess_wt(x)
            w_val = w(x)
            ∇w_val = ∇w(x)
            hess_w_val = hess_w(x)
            t_val = t(D,x)
            jac_t_val = jac_t(D,x)
            hess_t_val = hess_t(D,x)
            m, d = size(jac_t_val)
        return reduce(vcat,[vec( hess_w_val * t_val[i] + w_val * reshape(hess_t_val[i,:],d,d) + ∇w_val * jac_t_val[i,:]' + jac_t_val[i,:] * ∇w_val' )' for i in 1:m])
    end
        
    return fit_smom(D,x -> w(x)*jac_t(D,x) + t(D,x)*∇w(x)', hess_wt, X)
end

function fit_smom(D::Type{<:ContinuousDistribution},X;w::Union{Function, Nothing}=nothing)
    ∇w(x) = gradient(w,x)[1]
    if isnothing(w) || isnothing(∇w(X[:,1]))
        return fit_smom(D,x -> jac_t(D,x),x -> hess_t(D,x),X)
    else
        hess_w(x) = hessian(w,x)
        return fit_smom(D,w,∇w,hess_w, X)
    end
end