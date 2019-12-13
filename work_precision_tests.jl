
using DifferentialEquations, BenchmarkTools;
include("./pkode.jl")
using Main.point_kinetics
using Statistics;
using Serialization;
using Sundials;
using LSODA;
using ODEInterface;


solvers = Dict("SDIRK" => [ImplicitEuler, Trapezoid, TRBDF2, SDIRK2,
                Kvaerno3, KenCarp3, Cash4, Hairer4, Hairer42, Kvaerno4, KenCarp4,
                Kvaerno5, KenCarp5],
                )

solvers = (Rosenbrock23, KenCarp3, KenCarp5, CVODE_BDF, lsoda, TRBDF2) #radau,

function build_all_cases()

        ## Generatre all tolerance options
        # tolerances = [10.0^(-i) for i in 2:12]
        tolerances = [10.0^(-i) for i in 1:12]
        # tolerances = [10.0^(-i) for i in 12:12]

        ## Compile all relevant solver options
        # solver_algs = solvers["SDIRK"]
        solver_algs = solvers

        ## External Reactivity Functions
        inserted_react = 1.e-3
        reactivities = [t -> t > 5. ? inserted_react : 0.]
        # reactivities = [t -> (tanh((t - 50.)*1e-4) + 1.) * inserted_react / 2.  t -> t > 50. ? inserted_react : 0.]
        # reactivities = [t -> (tanh((t - 50.)*1e-1) + 1.) * 40.e-4  , t -> t > 50. ? 8.e-3 : 0.]

        cases = [(c1, c2, c3, [t for t in 0:0.1:10]) for c1 in reactivities
                                for c2 in solver_algs for c3 in tolerances]
        cases
end

plot(map(t -> t > 5. ? 1.e-3 : 0.,  0:0.1:10))
# function solve_wrapper(alg, u0, p, tspan)
#         problem = ODEProblem(point_kinetics.pk!, u0, tspan, p);
#         solve(problem, alg(), abstol=atol, reltol=atol*1.e3, save_everystep=false);
# end

function benchmark_ODE_case(case)
        ρ, alg, atol, saveat = case
        max_rho = ρ(10.)
        println("Beginning $alg, with atol $atol. ρ(10)=$max_rho")

        ## Fast Reactor Constants
        delayed_neutron_fractions = [9, 87, 70, 140, 60, 55] * 1.e-5
        precursor_tcs = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]
        mean_generation_time = 1.e-5

        ## ODE Parameter Definition
        p = point_kinetics.PKparams(ext_reac, mean_generation_time,
                            sum(delayed_neutron_fractions), precursor_tcs, delayed_neutron_fractions)

        ## Initial Condition Definition
        u0 = delayed_neutron_fractions ./ (mean_generation_time .* precursor_tcs)
        u0 = vcat([1.],u0)

        ## Timespan Setting
        tspan = (0., 10.)

        problem = ODEProblem(point_kinetics.pk!, u0, tspan, p)
        sol = solve(problem, alg(), abstol=atol, reltol=atol*1.e3, saveat=saveat)

        bm = @benchmark solve($problem, $alg(), abstol=$atol, reltol=$atol*1.e3, save_everystep=false)

        case, bm, sol
end

cases = build_all_cases()

function calc_all_benchmarks()
        benchmarks = Array{Any, 1}(undef, length(cases))
        idx = 1
        num_cases = length(cases)
        for case in cases

                case, bm, sol = benchmark_ODE_case(case)

                alg = case[2]
                tol = case[3]
                median_time = median(bm.times)
                # median_time = 0
                println("$idx / $num_cases  Completed: (alg, tol, median time) = $alg, $tol, $median_time")

                benchmarks[idx] = (case, bm, sol)

                idx += 1

        end
        # cur_dir = pwd()
        # open("$cur_dir/benchmarks/SDIRKs/benchmarks.jld", "w") do io
        #         write(io, benchmarks)
        # end;
        benchmarks
end

# test = calc_all_benchmarks()
# plot(test[1][end], vars=(1))

benchmarks = calc_all_benchmarks()

serialize("benchmarks/benchmars.dat", benchmarks)

#https://benchmarks.juliadiffeq.org/html/StiffODE/VanDerPol.html


using Plots;

# scatter([mean(benchmarks[i][2].times) for i in 1:length(benchmarks)], [benchmarks[i][1][3] for i in 1:length(benchmarks)], yaxis=:log)


function group_by_alg(alg)

        tols = Array{Float64, 1}(undef, 12)
        times = Array{Float64, 1}(undef, 12)

        idx = 1
        for (case, bm, sol) in benchmarks
                if case[2] != alg
                        continue
                end
                tols[idx] = case[3]
                times[idx] = mean(bm.times)
                idx += 1

        end
        tols, times
end

function group_powers_by_alg(alg)
        sols = Array{Array{Float64, }, 1}(undef, 12)

        idx = 1
        for (case, bm, sol) in benchmarks
                reac = case[1](100)
                tol = case[3]
                println(case[2])

                if case[2] != lsoda
                        nreject = sol.destats.nreject
                        maxeig = sol.destats.maxeig
                else
                        nreject = 0
                        maxeig = 0
                end
                println("CASE: reac: $reac, tol: $tol, num_reject: $nreject, maxeig: $maxeig")
                # println(case[1], case[3], sol.destats.nreject, sol.destats.maxeig)
                if case[2] != alg
                        continue
                end

                sols[idx] = hcat(sol.u...)[1,:]
                idx += 1
        end
        println(sols)
        hcat(sols...)
end


function build_work_precision_plot(benchmarks, solvers)
        idx = 0
        for solver in solvers
                tols, times = group_by_alg(solver)
                if idx == 0
                        scatter(tols[6:end], times[6:end], xaxis=:log, label=solver)
                else
                        scatter!(tols[6:end], times[6:end], xaxis=:log, label=solver)
                end
                idx += 1
        end
        current()
end



benchmarks
solvers
# plot(1.:0.1:100, benchmarks[1][end](1.:0.1:100)[1,:])

build_work_precision_plot(benchmarks, solvers[2:end])




sols = group_powers_by_alg(CVODE_BDF)
tols, times = group_by_alg(CVODE_BDF)
tols2, times2 = group_by_alg(lsoda)
tols3, times3 = group_by_alg(Rosenbrock23)
tols3, times3 = group_by_alg(Rosenbrock23)
# plot(sols[450:600,11:11:end])
plot(sols[:,:])
# plot!(map(t -> (t > 50. ? 1e-3 : 0.), 0:0.1:100)[450:550])

# map(t -> (t > 50. ? 1e-3 : 0.), 0:0.1:100)[450:550]


scatter(tols[1:end-4], times[1:end-4], xaxis=:log)
scatter!(tols2[1:end-4], times2[1:end-4], xaxis=:log)
scatter!(tols3[1:end-4], times3[1:end-4], xaxis=:log)
