using DifferentialEquations, Sundials, Plots;
using LinearAlgebra, BenchmarkTools;
using Profile;


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
using .point_kinetics;

#
tspan = (0.,10.)

delayed_neutron_fractions = [9, 87, 70, 140, 60, 55] * 1.e-5
precursor_tcs = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]
mean_generation_time = 1.e-5
delayed_neutron_fractions ./ sum(delayed_neutron_fractions)
# ext_reac(t) =    t > 50. ? 1.e-3 : 0.
ext_reac(t) =    t > 5. ? 4.21e-3 : 0.
# ext_reac(t) = (tanh.((t .- 50.).*1e-1) .+ 1.) .* 40.e-4
ext_reac(t) = (tanh.((t .- 5.)*100) .+ 1.) .* 5.e-4


# plot(1:0.1:10.,ext_reac(1:0.1:10.))

p = point_kinetics.PKparams(ext_reac, mean_generation_time,
                    sum(delayed_neutron_fractions), precursor_tcs, delayed_neutron_fractions)

u0 = delayed_neutron_fractions ./ (mean_generation_time .* precursor_tcs)
u0 = vcat([1.],u0)

prob = ODEProblem(point_kinetics.pk!, u0, tspan, p)
#
sol = solve(prob, CVODE_BDF(), saveat=4.9:0.001:5.1)

plot(sol, vars=(1))

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
ForwardDiff.jacobian(step_function, integrator.u)

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
#
# using Traceur;
# @trace sol = solve(prob, Tsit5())
#
# Juno.profiler()
# Profile.clear()

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
