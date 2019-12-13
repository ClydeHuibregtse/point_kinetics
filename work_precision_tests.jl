
using DifferentialEquations, BenchmarkTools;
include("./pkode.jl")
using Main.point_kinetics
using Statistics;
using Serialization;
using Sundials;
using LSODA;
using ODEInterfaceDiffEq;
using ODE;
using DoubleFloats;
# Pkg.add("ODEInterface")
using ODEInterface;



solvers = (Rosenbrock23, KenCarp3, KenCarp5, CVODE_BDF, lsoda, TRBDF2, rodas, radau5, dopri5, RadauIIA5)
# solvers = ( rodas, radau5, dopri5, RadauIIA5)

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
        reactivities = [point_kinetics.StepInsert(inserted_react, 5.), point_kinetics.TanhInsert(inserted_react, 5., 10.)]
        # reactivities = [t -> (tanh((t - 50.)*1e-4) + 1.) * inserted_react / 2.  t -> t > 50. ? inserted_react : 0.]
        # reactivities = [t -> (tanh((t - 50.)*1e-1) + 1.) * 40.e-4  , t -> t > 50. ? 8.e-3 : 0.]

        cases = [(c1, c2, c3, [t for t in 0:0.01:10]) for c1 in reactivities
                                for c2 in solver_algs for c3 in tolerances]
        cases
end


function benchmark_ODE_case(case)
        ρ, alg, atol, saveat = case
        max_rho = ρ(10.)
        println("Beginning $alg, with atol $atol. ρ(10)=$max_rho")

        ## Fast Reactor Constants (#GANOPOL)
        delayed_neutron_fractions = [9, 87, 70, 140, 60, 55] * 1.e-5
        precursor_tcs = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]
        mean_generation_time = 1.e-5

        ## ODE Parameter Definition
        p = point_kinetics.PKparams(ρ, mean_generation_time,
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

serialize("benchmarks/benchmarks.dat", benchmarks)

#https://benchmarks.juliadiffeq.org/html/StiffODE/VanDerPol.html

# benchmarks = deserialize("benchmarks/benchmarks.dat")
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
                # println(case[2])

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
        # println(sols)
        hcat(sols...)
end


function build_work_precision_plot(benchmarks, solvers, test_sol)
        idx = 0

        wp = true
        final_tp = true

        for solver in solvers
                tols, times = group_by_alg(solver)
                powers = group_powers_by_alg(solver)
                errors = abs.(powers .- test_sol)
                println(size(errors))
                mean_errors = mean(errors, dims=1)

                if wp
                        if final_tp
                                if idx == 0
                                        plot(errors[end,7:end], times[7:end], xaxis=:log, shape=:circle, label=solver)
                                        # plot(errors[end,1:6], times[1:6], xaxis=:log, shape=:circle, label=solver)
                                else
                                        plot!(errors[end,7:end], times[7:end], xaxis=:log, shape=:circle,label=solver)
                                        # plot!(errors[end,1:6], times[1:6], xaxis=:log, shape=:circle,label=solver)
                                end
                                idx += 1
                        else

                                # println(size(mean_errors[:,7end]), size(times[6:end]))
                                if idx == 0
                                        # plot(mean_errors[:,7:end]', times[7:end], xaxis=:log, shape=:circle, label=solver)
                                        plot(mean_errors[:,1:6]', times[1:6], xaxis=:log, shape=:circle, label=solver)
                                else
                                        # plot!(mean_errors[:,7:end]', times[7:end], xaxis=:log, shape=:circle,label=solver)
                                        plot!(mean_errors[:,1:6]', times[1:6], xaxis=:log, shape=:circle,label=solver)
                                end
                                idx += 1
                        end
                else
                        # errors += 1e-100
                        tol = tols[1]
                        tspan = 5.:.01:10.
                        plot(tspan, errors[501:end,1], label="$tol", yaxis=:log)

                        for i in 2:length(tols)
                                tol = tols[i]
                                plot!(tspan, errors[501:end, i], label="$tol", yaxis=:log)
                        end
                        # yaxis!(:log)
                        xlabel!("Time (s)")
                        ylabel!("Error")
                        savefig("plots/timeseries-error/$solver.png")
                end

        end
        xlabel!("Final Timepoint Error")
        ylabel!("Median Runtime (ns)")
        savefig("plots/work-precision/final_tp_low_tolerance_100pcm.png")
        current()
end




test_sol = reshape(hcat(point_kinetics.Ψ.(0:.01:5., 1e-3)...)[1,:], (501,1))
test_sol = vcat(ones(500,1), test_sol)

build_work_precision_plot(benchmarks, solvers[1:end], test_sol)


function build_a_sol_plot(rho)
        tspan = [t for t in 0:.01:5]
        plot_tspan = [t for t in 0:0.01:10]
        sol = hcat(point_kinetics.Ψ.(tspan, rho)...)[1,:]
        sol = vcat(ones(500,1), reshape(sol, (501,1)))
        println(size(sol))
        plot(plot_tspan, sol, label="100pcm Insertion")
        xlabel!("Time (s)")
        ylabel!("Normalized Neutron Density")

        savefig("plots/analytical-sols/100pcm_fp_error.png")
        current()
end

build_a_sol_plot(1.e-3)
