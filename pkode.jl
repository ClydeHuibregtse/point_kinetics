using DifferentialEquations, Plots;
using LinearAlgebra, BenchmarkTools;
using Profile;
using Winston;

module point_kinetics
    struct PKparams
        ρ
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

workspace()
reload("point_kinetics")
using .point_kinetics;

#
tspan = (0.,100.)

delayed_neutron_fractions = [9, 87, 70, 140, 60, 55] * 1.e-5
precursor_tcs = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]
mean_generation_time = 1.e-5
#
ext_reac(t) =    t > 50. ? -1.e-2 : 0. #shutdown
ext_reac(t) = (tanh.((t .- 50.).*1e-1) .+ 1.) .* 40.e-4
# ext_reac(t) = (tanh.((t .- 50.).*1e-1) .+ 1.) .* 5.e-4


# plot(ext_reac(1:0.1:100))

p = point_kinetics.PKparams(ext_reac, mean_generation_time,
                    sum(delayed_neutron_fractions), precursor_tcs, delayed_neutron_fractions)

u0 = delayed_neutron_fractions ./ (mean_generation_time .* precursor_tcs)
u0 = vcat([1.],u0)

prob = ODEProblem(point_kinetics.pk!, u0, tspan, p)
#
sol = solve(prob, Tsit5())
#
# plot(sol[1:end], vars=(1))
#
# @profile for i in 1:100 solve(prob, Tsit5(), save_everystep=false) end
#
#
# using Traceur;
# @trace sol = solve(prob, Tsit5())
#
# Juno.profiler()
# Profile.clear()
#
#
#
#
#
#
#
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
