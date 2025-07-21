# Estimator for the canonical parameter eta of Exponential Family on $\mathbb{R}^{d}$

function _SM_ExpFam_core(jac_t::Function,lap_t::Function,w::Union{Function,Nothing},∇w::Union{Function,Nothing},X::AbstractMatrix)
    n = size(X,2)
    @assert n > 0 "X is empty."

    m = size(jac_t(X[:,1]),1)
    
    A = zeros(m,m)
    b = zeros(m)

    for i in 1:n
        x = X[:,i]
        
        jac_t_val = jac_t(x)
        lap_t_val = lap_t(x)
        
        if isnothing(w)
            A += jac_t_val * jac_t_val'
            b += lap_t_val
        else
            w_val = w(x)
            A += w_val * jac_t_val * jac_t_val'
            b += w_val * lap_t_val  +  jac_t_val * ∇w(x)
        end
    end
    
    return vec(A \ b)
end

function _SM_ExpFam_core(jac_t::Function,lap_t::Function,X::AbstractMatrix; w::Union{Function, Nothing}=nothing)
    n = size(X,2)
    @assert n > 0 "X is empty."
    
    ∇w(x) = gradient(w,x)[1]
    
    if isnothing(w) || isnothing(∇w(X[:,1]))
        return _SM_ExpFam_core(jac_t,lap_t,nothing,nothing,X)
    else
        return _SM_ExpFam_core(jac_t,lap_t,w,∇w,X)
    end
end

function SM_ExpFam(t, X; w::Union{Function, Nothing}=nothing)
    n = size(X,2)
    @assert n > 0 "X is empty."

    m = length(t(X[:,1]))
    
    jac_t(x) = jacobian(t,x)[1]
    lap_t(x) = vec(sum(reduce(hcat,[hessian(u -> t(u)[i],x) |> diag for i in 1:m]), dims=1))
    
    return _SM_ExpFam_core(jac_t,lap_t,X;w=w)
end

# Estimator for the canonical parameter eta of Exponential Family on $\mathbb{S}^{d-1}$
function _SSM_ExpFam_core(jac_t::Function,hess_t::Function,X::AbstractMatrix)
    d, n = size(X)
    @assert n > 0 "X is empty."

    m = size(jac_t(X[:,1]),1)
    
    P(x) = I(d) - x*x'
    
    A = zeros(m,m)
    b = zeros(m)

    for i in 1:n
        x = X[:,i]
        
        jac_t_val = jac_t(x)
        hess_t_val = hess_t(x)
        lap_t_val = mapslices(v -> tr(reshape(v,d,d)),hess_t_val;dims=2)
        P_val = P(x)
        
        A += jac_t_val * P_val * jac_t_val'
        b += (1-d) * jac_t_val * x - hess_t_val * kron(x,x) + lap_t_val
    end
    
    return vec(A \ b)
end

function _SSM_ExpFam_core(jac_t::Function,hess_t::Function,w::Union{Function,Nothing},∇w::Union{Function,Nothing},X::AbstractMatrix)
    d, n = size(X)
    @assert n > 0 "X is empty."

    m = size(jac_t(X[:,1]),1)
    
    P(x) = I(d) - x*x'
    
    A = zeros(m,m)
    b = zeros(m)

    for i in 1:n
        x = X[:,i]
        
        jac_t_val = jac_t(x)
        hess_t_val = hess_t(x)
        lap_t_val = mapslices(v -> tr(reshape(v,d,d)),hess_t_val;dims=2)
        P_val = P(x)

        if isnothing(w)
            A += jac_t_val * P_val * jac_t_val'
            b += (1-d) * jac_t_val * x - hess_t_val * kron(x,x) + lap_t_val
        else
            w_val = w(x)
            A += w_val * jac_t_val * P_val * jac_t_val'
            b += w_val * ( (1-d) * jac_t_val * x - hess_t_val * kron(x,x) + lap_t_val ) + jac_t_val * P_val * ∇w(x)
        end
    end
    return vec(A \ b)
end

function _SSM_ExpFam_core(jac_t::Function,hess_t::Function,X;w::Union{Function,Nothing}=nothing) 
    n = size(X,2)
    @assert n > 0 "X is empty"
    
    ∇w(x) = gradient(w,x)[1]
    
    if isnothing(w) || isnothing(∇w(X[:,1]))
        return _SSM_ExpFam_core(jac_t,hess_t,nothing,nothing,X)
    else
        return _SSM_ExpFam_core(jac_t,hess_t,w,∇w,X)
    end
end

function SSM_ExpFam(t,X;w::Union{Function,Nothing}=nothing) 
    n = size(X,2)
    @assert n > 0 "X is empty"

    m = length(t(X[:,1]))
    
    jac_t(x) = jacobian(t,x)[1]
    hess_t(x) = reduce(vcat,[vec(hessian(u -> t(u)[i],x))' for i in 1:m])

    return _SSM_ExpFam_core(jac_t,hess_t,X;w=w)
end