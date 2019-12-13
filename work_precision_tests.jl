
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

length(benchmarks)
step_benchmarks = benchmarks[1:Int(length(benchmarks)/2)]
tanh_benchmarks = benchmarks[Int(length(benchmarks)/2)+1:end]


serialize("benchmarks/benchmarks.dat", benchmarks)

#https://benchmarks.juliadiffeq.org/html/StiffODE/VanDerPol.html

benchmarks = deserialize("benchmarks/benchmarks.dat")
using Plots;

# scatter([mean(benchmarks[i][2].times) for i in 1:length(benchmarks)], [benchmarks[i][1][3] for i in 1:length(benchmarks)], yaxis=:log)


function group_by_alg(benchmarks, alg)

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

function group_powers_by_alg(benchmarks, alg)
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
# tanh_benchmarks
# plot(tanh_benchmarks[1][1][1].(0:.01:10.))
# benchmarks[1:Int(length(benchmarks)/2)]
# benchmarks[Int(length(benchmarks)/2)+1:end]
function build_work_precision_plot(benchmarks, solvers, test_sol)
        idx = 0

        wp = false
        final_tp = false
        high = true
        tanh = true

        # benchmarks = tanh ? benchmarks[1:Int(length(benchmarks)/2)] : benchmarks[Int(length(benchmarks)/2)+1:end] end

        plot()
        for solver in solvers
                if solver == dopri5 continue end# || solver == Rosenbrock23 || solver == KenCarp5 || solver == KenCarp3 continue end
                tols, times = group_by_alg(benchmarks, solver)
                powers = group_powers_by_alg(benchmarks, solver)
                errors = abs.(powers .- test_sol)

                mean_errors = mean(errors, dims=1)

                if wp
                        if final_tp
                                if high
                                        plot!(errors[end,1:6], times[1:6], xaxis=:log, shape=:circle,label=solver)
                                else
                                        plot!(errors[end,7:end], times[7:end], xaxis=:log, shape=:circle,label=solver)
                                end
                        else
                                if high
                                        plot!(mean_errors[:,1:6]', times[1:6], xaxis=:log, shape=:circle,label=solver)
                                else
                                        plot!(mean_errors[:,7:end]', times[7:end], xaxis=:log, shape=:circle,label=solver)
                                end
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
                        savefig("plots/timeseries-error/$solver-tanh.png")
                end

        end
        if final_tp
                xlabel!("Final Timepoint Error")
                prefix = "final_tp"
        else
                xlabel!("Mean Error")
                prefix = ""
        end
        ylabel!("Median Runtime (ns)")
        if high
                if tanh
                        savefig("plots/work-precision/$prefix-high_tolerance_100pcm_tanh.png")
                else
                        savefig("plots/work-precision/$prefix-high_tolerance_100pcm.png")
                end
        else
                if tanh
                        savefig("plots/work-precision/$prefix-low_tolerance_100pcm_tanh.png")
                else
                        savefig("plots/work-precision/$prefix-low_tolerance_100pcm.png")
                end
        end
        current()
end




test_sol = reshape(hcat(point_kinetics.Ψ.(0:.01:5., 1e-3)...)[1,:], (501,1))
test_sol = vcat(ones(500,1), test_sol)

build_work_precision_plot(tanh_benchmarks, solvers[1:end], test_sol)


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

benchmarks[1][1][3]




function count_steps(benchmarks)
        step_info = Dict{String, Any}()
        for (case, bm, sol) in benchmarks
                if case[2] == lsoda continue end

                num_accepted = sol.destats.naccept
                num_rejected = sol.destats.nreject
                num_total = num_accepted + num_rejected

                solver, tol = case[2:3]

                if !haskey(step_info, "$solver")
                        step_info["$solver"] = Dict()
                end

                step_info["$solver"][tol] = [num_total num_accepted num_rejected]
        end
        return step_info
end


tanh_steps = count_steps(tanh_benchmarks)
step_steps = count_steps(step_benchmarks)

collect(keys(step_steps))
# vcat(collect(values(step_steps[radau5]))...)
step_steps["TRBDF2"]
function plot_steps(steps)
        plot()
        for alg in collect(keys(steps))
                if alg == "Rosenbrock23" continue end
                alg_steps = sort(steps[alg])
                tols = collect(keys(alg_steps))
                tol_steps = vcat(collect(values(alg_steps))...)
                plot!(tols, tol_steps[:,1], label=alg, xaxis=:log)
        end
        xlabel!("Absolute Tolerance")
        ylabel!("Number of Total Solve Steps")
        savefig("plots/step_plots/tanhinserts.png")
        current()
end

plot_steps(tanh_steps)
