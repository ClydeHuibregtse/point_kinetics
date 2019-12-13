using DifferentialEquations, Sundials, Plots;
using LinearAlgebra, BenchmarkTools;
using Profile;



module point_kinetics
    abstract type AbstractInsert end
    # (p::AbstractInsert)(t::Float64)::Float64


    struct StepInsert <: AbstractInsert
        ρ::Float64
        t::Float64
    end

    (insert::StepInsert)(t::Float64)::Float64 = t > insert.t ? insert.ρ : 0.

    struct TanhInsert <: AbstractInsert
        ρ::Float64
        t::Float64
        k::Float64
    end
    (insert::TanhInsert)(t::Float64)::Float64 = (tanh((t - insert.t) * insert.k) + 1.) / 2. * insert.ρ


    struct PKparams
        ρ::AbstractInsert
        Λ::Float64
        β::Float64
        lams::Array{Float64, 1}
        bets::Array{Float64, 1}
    end

    function pk!(du, u, p, t)
        ρ = p.ρ
        Λ = p.Λ
        β =  p.β
        lams = p.lams
        bets = p.bets

        prec_conc = @view u[2:end]
        d_prec_conc = @view du[2:end]

        du[1] = (ρ(t) - β)*u[1] / Λ + lams' *  prec_conc

        d_prec_conc .= bets .* u[1] ./ Λ .- lams .* prec_conc
    end

    export pk!, PKparams
end
using .point_kinetics;


tspan = (0.,10.)

delayed_neutron_fractions = [9, 87, 70, 140, 60, 55] * 1.e-5
precursor_tcs = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]
mean_generation_time = 1.e-5
delayed_neutron_fractions ./ sum(delayed_neutron_fractions)
# step_reac = point_kinetics.StepInsert(1.e-3, 5.)
step_reac = point_kinetics.TanhInsert(1.e-3, 5., 10)


# plot(1:0.1:10.,ext_reac(1:0.1:10.))

p = point_kinetics.PKparams(step_reac, mean_generation_time,
        sum(delayed_neutron_fractions), precursor_tcs, delayed_neutron_fractions)

u0 = delayed_neutron_fractions ./ (mean_generation_time .* precursor_tcs)
u0 = vcat([1.],u0)

prob = ODEProblem(point_kinetics.pk!, u0, tspan, p)
#
sol = solve(prob, KenCarp3(), saveat=0.:0.1:10., atol=1e-12, rtol=1e-9)

plot(sol, vars=(1))



@profile for _ in 1:1000 solve(prob, CVODE_BDF(), save_everystep=false, atol=1e-12, rtol=1e-9)end
Juno.profiler()
Profile.clear()
using ForwardDiff;
using Calculus;
# step!(integrator)
# integrator


function step_function(u,t)
    du = similar(u);
    point_kinetics.pk!(du, u, p, t);
    du
end

# solve!(integrator)
# integrator.t

integrator =  init(prob, CVODE_BDF())
jac = ForwardDiff.jacobian(_u -> step_function(_u, integrator.t), integrator.u)

for step in 1:40
    step!(integrator)
    jac = ForwardDiff.jacobian(_u -> step_function(_u, integrator.t), integrator.u)
    max_val, min_val = max(jac...), min(jac...)
    if step % 10 == 0
        # println(integrator.u[1])
        # println(integrator.t)
        eigvals_jac = eigvals(jac)
        max_eigval = max(abs.(eigvals_jac)...)
        min_eigval = min(abs.(eigvals_jac)...)
        sr = max_eigval / min_eigval
        println("STEP: $step, Maximum jacobian value: $max_val, Minimum jacobian value: $min_val, Stiffness ratio: $sr")
    end
end
# @profile for i in 1:100 solve(prob, Tsit5(), save_everystep=false) end
#
# using Traceur;
# @trace sol = solve(prob, Tsit5())
#
#
# Juno.profiler()
# Profile.clear()

using Traceur;
using BenchmarkTools;

function pk_test!(du, u, p, t)
    ρ = p.ρ
    Λ = p.Λ
    β =  p.β
    lams = p.lams
    bets = p.bets

    # n = @view u[1]
    prec_conc = @view u[2:end]

    # dn = @view du[1]
    # d_prec_conc = @view du[2:end]

    du[1] = (ρ(t) - β)*u[1] / Λ + lams' *  prec_conc


    # d_prec_conc .= bets .* u[1] ./ Λ .- lams .* prec_conc
end

u = u0
du = similar(u)

# @trace point_kinetics.pk!(du, u, p, 0.0)

@btime pk_test!(du, u, p, 0.0)
@allocated pk_test!(du, u, p, 0.0)
@code_llvm pk_test!(du, u, p, 0.0)
@code_warntype pk_test!(du, u, p, 0.0)
a = 1

# @benchmark solve(prob, TRBDF2())
# @benchmark solve(prob, ImplicitEuler())
# @benchmark solve(prob, Kvaerno5())
#
# sol = solve(prob, TRBDF2())
# plot(sol, vars=(1))
#
#
#
#
#
#
# using DiffEqSensitivity;
# ls_prob = ODELocalSensitivityProblem(pk!, u0, tspan, p)
# param_sol = solve(ls_prob, DP8())
#
# x,dp = extract_local_sensitivities(param_sol)
# dp[1]
# plot(param_sol.t, dp[1]')
#
#
# using ForwardDiff, Calculus;
# function test(p)
#     prob = ODEProblem(pk!, u0, tspan, p)
#     solve(prob, TRBDF2())
# end
#
# ForwardDiff.jacobian(test, p)
#
#
#
# function f(du,u,p,t)
#   du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
#   du[2] = dy = -p[3]*u[2] + u[1]*u[2]
# end
# using ForwardDiff, Calculus
# function test_f(p)
#   prob = ODEProblem(f,eltype(p).([1.0,1.0]),eltype(p).((0.0,10.0)),p)
#   solve(prob,Vern9(),abstol=1e-14,reltol=1e-14,save_everystep=false)[end]
# end
#
# p = [1.5,1.0,3.0]
# fd_res = ForwardDiff.jacobian(test_f,p)
# calc_res = Calculus.finite_difference_jacobian(test_f,p)


using LinearAlgebra;
using Plots;
##### Analytical solution #####
rho = 1e-3
betas = [9, 87, 70, 140, 60, 55] * 1.e-5
sum_B = sum(betas)
lambdas = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]
Λ = 1.e-5

Ψ0 = [1.; betas ./ (Λ .* lambdas)]



jac = [(rho - sum_B)/Λ lambdas[1] lambdas[2] lambdas[3] lambdas[4] lambdas[5] lambdas[6]
        betas[1]/Λ -lambdas[1]      0           0           0          0        0
        betas[2]/Λ     0        -lambdas[2]     0           0          0        0
        betas[3]/Λ     0            0     -lambdas[3]       0          0        0
        betas[4]/Λ     0            0           0       -lambdas[4]    0        0
        betas[5]/Λ     0            0           0           0     -lambdas[5]   0
        betas[6]/Λ     0            0           0           0          0    -lambdas[6]
        ]

eigs = eigvals(jac)
function sub_sum(start, K, eigs, m_depth, j)
    if m_depth == 0
        return 1
    end
    sum = 0.
    for i_l in start:K
        if i_l == j continue
        end
        sum += eigs[i_l + 1] * sub_sum(i_l + 1, K, eigs, m_depth - 1, j)
    end
    return sum
end

function B(m,j, eigs;K=6)
    if m < 0 || m > K
        return 0
    elseif m == 0
        return 1
    else
        return sub_sum(0, K, eigs, m, j)
    end
end

function α(k,t, eigs; K=6)
    out = 0.
    for j in 0:K
        out += (exp(eigs[j+1]*t) * B(K-k, j, eigs)) / (prod([(eigs[i+1] - eigs[j+1]) for i in 0:K if i != j]))
        # out += (exp(eigs[j+1]*t))# * B(K-k, j, eigs)) / (prod([(eigs[i+1] - eigs[j+1]) for i in 0:K if i != j]))
    end

    out *= (-1.)^k

    return out

end

holder(t) = α(5,t,eigs)
ForwardDiff.derivative(holder, 0)

function Ψ(t)
    betas = [9, 87, 70, 140, 60, 55] * 1.e-5
    sum_B = sum(betas)
    lambdas = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]
    Λ = 1.e-5
    rho = 1.e-3
    # rho = sum_B

    Ψ_0 = [1.; betas ./ (Λ .* lambdas)]
    K=6
    jac = [(rho - sum_B)/Λ lambdas[1] lambdas[2] lambdas[3] lambdas[4] lambdas[5] lambdas[6]
            betas[1]/Λ -lambdas[1]      0           0           0          0        0
            betas[2]/Λ     0        -lambdas[2]     0           0          0        0
            betas[3]/Λ     0            0     -lambdas[3]       0          0        0
            betas[4]/Λ     0            0           0       -lambdas[4]    0        0
            betas[5]/Λ     0            0           0           0     -lambdas[5]   0
            betas[6]/Λ     0            0           0           0          0    -lambdas[6]
            ]
    eigs = eigvals(jac)

    α_array = zeros(size(jac))
    for k in 0:K
        # println(α(k,t,eigs))
        # println(k)
        α_array += α(k,t,eigs) .* (jac)^k
    end
    # println(α_array)

    out = α_array * Ψ_0

    return out
end

using ForwardDiff;
(Ψ(1.0000001) - Ψ(1))

ForwardDiff.derivative(Ψ, 1)

Ψ(0.001)
[1.; betas ./ (Λ .* lambdas)]
jac
α(0,0,eigs)
B(3,0,eigs)
eigs
anal_sol = hcat(map(Ψ, 0:.01:10)...)
plot(0:.01:10, anal_sol[1,:])


K = 6
j = 0
prod([(eigs[i+1] - eigs[j+1]) for i in 0:K if i != j])

w0, w1, w2, w3, w4, w5, w6 = eigs

#3,0
(w1*(w2*(w3 + w4 + w5 + w6) + w3*(w4 + w5 + w6) + w4*(w5 + w6) + w5*w6)) +
    (w2*(w3*(w4 + w5 + w6) + w4*(w5 + w6) + w5*w6) +w3*(w4*(w5+w6) + w5*w6) + w4*w5*w6)
#5,0
(w1*(w2*(w3*(w4*(w5+w6) + w5*w6) + w4*w5*w6) + w3*w4*w5*w6) + w2*w3*w4*w5*w6)
#2,0
(w0*(w2 + w3 +w4 + w5 + w6) + w2*(w3 + w4 + w5 + w6) + w3*(w4 + w5 + w6) + w4*(w5 + w6) + w5*w6)
sum(eigs[2:end])

prod(eigs[3:end])*eigs[1]





λ = lambdas[1]
A = [(rho - sum_B) / Λ λ
    sum_B / Λ λ]
w0, w1 = eigvals(A)

function n(t)
    rho / (Λ * (w0 - w1)) * (exp(w0*t) - exp(w1*t)) + (w1*exp(w0*t) - w0*exp(w1*t)) / (w1 - w0)

end

plot(0:.01:1,n.(0:.01:1))

ForwardDiff.derivative(n, 1)
