
using DifferentialEquations, BenchmarkTools;
include("./pkode.jl")
using Main.point_kinetics
using JLD;
using Statistics;


# ImplicitMidpoint()


solvers = Dict("SDIRK" => [ImplicitEuler, Trapezoid, TRBDF2, SDIRK2,
                Kvaerno3, KenCarp3, Cash4, Hairer4, Hairer42, Kvaerno4, KenCarp4,
                Kvaerno5, KenCarp5],
                )

function build_all_cases()

        ## Generatre all tolerance options
        tolerances = [10.0^(-i) for i in 2:12]

        ## Compile all relevant solver options
        solver_algs = solvers["SDIRK"]

        ## External Reactivity Functions
        reactivities = [t -> tanh(t - 50.) * 1.e-4]

        # cases = [(c1, c2, c3, c4, [1:0.1:100]) for c1 in reactivities
        #                         for c2 in solver_algs for c3 in tolerances for c4 in tolerances]
        cases = [(c1, c2, c3, [t for t in 1:0.1:100]) for c1 in reactivities
                                for c2 in solver_algs for c3 in tolerances]
        # cases = [(c1, c2, c3, [t for t in 1:0.1:100])  for c1 in reactivities for c3 in tolerances for c2 in solver_algs]
        cases
end

function benchmark_ODE_case(case)
        # ρ, alg, atol, rtol, saveat = case
        ρ, alg, atol, saveat = case


        ## Fast Reactor Constants
        delayed_neutron_fractions = [9, 87, 70, 140, 60, 55] * 1.e-5
        precursor_tcs = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]
        mean_generation_time = 1.e-5

        ## ODE Parameter Definition
        p = [ρ, mean_generation_time, sum(delayed_neutron_fractions)]#, precursor_tcs, delayed_neutron_fractions]
        p = vcat(p, precursor_tcs, delayed_neutron_fractions)

        ## Initial Condition Definition
        u0 = delayed_neutron_fractions ./ (mean_generation_time .* precursor_tcs)
        u0 = vcat([1.],u0)

        ## Timespan Setting
        tspan = (0., 100.)

        problem = ODEProblem(point_kinetics.pk!, u0, tspan, p)
        sol = solve(problem, alg(), abstol=atol, saveat=saveat, save_everystep=false)

        bm = @benchmark solve($problem, $alg(), abstol=$atol, saveat=$saveat, save_everystem=false) #reltol=$rtol, saveat=$saveat)

        case, bm, sol
end

cases = build_all_cases()


function calc_all_benchmarks()
        benchmarks = Array{Any, 1}(undef, length(cases))
        idx = 1
        for case in cases

                case, bm, sol = benchmark_ODE_case(case)

                alg = case[2]
                tol = case[3]
                median_time = median(bm.times)
                println("$idx / 143  Completed: (alg, tol, median time) = $alg, $tol, $median_time")

                benchmarks[idx] = (case, bm, sol)

                idx += 1
        end
        save("benchmarks/SDIRKs/benchmarks.jld", "bm", benchmarks)
        benchmarks
end

benchmarks = calc_all_benchmarks()
