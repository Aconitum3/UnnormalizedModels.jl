### Basic statistics

t(D::Type{MvNormal},x) = [x;vech(x*x')]

function reshape_MvN(η)
    d = (isqrt(1+8length(η))-2)÷2
    Λμ = -η[1:d]
    Λ = avech(η[d+1:end]) .* (ones(d,d) + I(d))
    return MvNormalCanon(Λμ,Λ)
end

function jac_t(D::Type{MvNormal},x)
    d = length(x)
    m = 2d + (d^2-d)÷2
    res = zeros(m,d)
    for i in 1:d
        res[i,i] = 1
    end

    for i in 1:d
        for j in i:d
            k(i,j) = (j^2-j)÷2 + i + d
            if i == j
                res[k(i,i),i] = 2x[i]
            else
                res[k(i,j),i] = x[j]
                res[k(i,j),j] = x[i]
            end
        end
    end
    return res
end

function hess_t(D::Type{MvNormal},x)
    d = length(x)
    m = 2d + (d^2-d)÷2
    res = zeros(m,d^2)

    function vecE(i,j)
        res_ = zeros(d^2)
        if i == j
            res_[(i-1)d+j] = 2
        else
            res_[(i-1)d+j] = 1
            res_[(j-1)d+i] = 1
        end
        return res_
    end

    k = d+1
    for i in 1:d
        for j in 1:i
            res[k,:] = vecE(i,j)
            k+=1
        end
    end
    return res
end

function lap_t(D::Type{MvNormal},x)
    d = length(x)
    return [zeros(d);vech(diagm(2ones(d)))]
end

fit_sm(D::Type{MvNormal},X; w::Union{Function, Nothing}=nothing) = _SM_ExpFam_core(x -> jac_t(D,x),x -> lap_t(D,x),X;w=w) |> reshape_MvN |> Distributions.meanform

fit_smom(D::Type{MvNormal},jac_f,hess_f,X) = _SMoM_ExpFam_core(x -> jac_t(D,x),jac_f,hess_f,X) |> reshape_MvN |> Distributions.meanform
fit_smom(D::Type{MvNormal},f,X) = _SMoM_ExpFam_core(x -> jac_t(D,x),f,X) |> reshape_MvN |> Distributions.meanform
