
using DifferentialEquations, BenchmarkTools;
# using DSP, Knet, ImageFiltering;
using ImageFiltering;
using Plots;
using CuArrays, Flux;
include("./pkode.jl")
using Main.point_kinetics

# nxn arrays
function heat_transfer!(dT, T, p, t)
    @views h, K, C, Q, A = p

    n = size(T)[1]
    for j in 2:n-1, i in 2:n-1
        @views dT[i,j] = A[i,j] * (T[i,j] - T[i-1,j])
        @views dT[i,j] += A[i,j] * (T[i,j] - T[i+1,j])
        @views dT[i,j] += A[i,j] * (T[i,j] - T[i,j+1])
        @views dT[i,j] += A[i,j] * (T[i,j] - T[i,j-1]) ## Image filtering kernel ops?
        @views dT[i,j] *= -1. / C[i,j] * h


        @views dT[i,j] += Q[i,j]

        # dT[i,j] = -1. / c * hA * sum( (T[i,j] - T[i+xi, j+yi]) for xi in -1:1, yi in -1:1 if abs(xi) != abs(yi))
        # dT[i,j] = -1. / c * hA * sum( (T[i,j] - T[x,y]) for (x,y) in find_neighbors(i,j)) #+ q[i*n + j]

    end
    dT .*= C
end

function heat_transfer_conv!(dT, T, p, t)
    @views h, K, C, Q, A = p

    @views dT[2:end-1, 2:end-1] .= imfilter(T, K)[2:end-1, 2:end-1] .+ Q[2:end-1, 2:end-1]
end

function heat_transfer_CNN!(dT, T, p, t)
    @views heat_transfer_NN = p[1]

    act_dT = @view dT[2:end-1, 2:end-1, :, :]
    act_dT .= heat_transfer_NN(T)
end


# struct PKF/ullParams{T}
    # ρ::T
    # t::Float64
    # NN::






n = 12
T = zeros(n,n)

T[5,5] = 100.


C = ones(n,n)
# C[5,2:end-1] .= 1e-1

Q = zeros(n,n)

# Q[5,5] = 1.
A = ones(n,n)

K =  [ 0 1 0; 1 -4 1; 0 1 0.]
p =  [1.,K, C, Q, A]

tspan = (0.,10.)

prob = ODEProblem(heat_transfer!, T, tspan, p)
@benchmark solve(prob)



K_ch = CuArray(reshape(K, (3,3,1,1)))
T_ch = CuArray(reshape(T, (n,n,1,1)))
bias = CuArray(zeros(1))
h = 1.
C_ch = CuArray(reshape(C[2:end-1,2:end-1], (n-2, n-2, 1, 1)))
Q_ch = CuArray(reshape(Q[2:end-1, 2:end-1], (n-2, n-2, 1, 1)))

Diagonal(α, β) = x -> α .* x .+ β

diag_layer = Diagonal(1. ./ C_ch .* h, Q_ch)
conv_layer = Conv(K_ch, bias)


heat_transfer_NN = Chain(conv_layer, diag_layer)
p_cnn = [heat_transfer_NN]


cnn_prob = ODEProblem(heat_transfer_CNN!, T_ch, tspan, p_cnn)
cnn_sol = solve(cnn_prob)


# dT = similar(T_ch)
# a = @benchmark heat_transfer_CNN!(dT, T_ch, p_cnn, 0)
# a

cnn_out = convert(Array{Float64}, cnn_sol)


@gif for t in 1:size(cnn_sol)[end]
    heatmap(cnn_out[:,:,1,1,t])
end



using BenchmarkTools;

function scale_problem_size()
    exp_range = 4:8

    naive = Array{BenchmarkTools.Trial, 1}(undef, length(exp_range))
    imfilt = Array{BenchmarkTools.Trial, 1}(undef, length(exp_range))
    cnn = Array{BenchmarkTools.Trial, 1}(undef, length(exp_range))
    idx = 1
    for n in 2 .^(exp_range)
        println(n)
        # Build empty array of temperature values
        T = zeros(n,n)
        # Store at some arbitrary centered index a large temperature to be
        # transferred throughout the space.
        T[Int(floor(n/2)),Int(floor(n//2))] = 1.

        # Specify arbitrary thermal conductivities
        C = ones(n,n)

        # Assume no heat generation (system is insulated)
        Q = zeros(n,n)

        # Unused at the moment, but corresponds to the contact area between cells
        A = ones(n,n)

        # Heat transfer kernal
        K =  [ 0 1 0; 1 -4 1; 0 1 0.]

        # Define parameters for ODE solve
        p =  (1.,K, C, Q, A)

        # 10 second run
        tspan = (0.,10.)


        ### Define CNN

        # put everything into CuArrays
        h = 1.
        T_ch = CuArray(reshape(T, (n,n,1,1)))
        K_ch = CuArray(reshape(K, (3,3,1,1)))
        C_ch = CuArray(reshape(C[2:end-1,2:end-1], (n-2, n-2, 1, 1)))
        Q_ch = CuArray(reshape(Q[2:end-1, 2:end-1], (n-2, n-2, 1, 1)))
        bias = CuArray(zeros(1))

        # Define a diagonal layer for linear applications
        Diagonal(α, β) = x -> α .* x .+ β
        diag_layer = Diagonal(1. ./ C_ch .* h, Q_ch)

        # Define the convolutional layer with the kernal defined as K to enforce
        # the heat transferring
        conv_layer = Conv(K_ch, bias)

        # String the two layers together to create the full heat transfer derivative
        heat_transfer_NN = Chain(conv_layer, diag_layer)

        # Place the NN into a parameter tuple (unneccessary, but will change to full
        # parameter object come implementation with PK)
        p_cnn = (heat_transfer_NN)


        naive_prob = ODEProblem(heat_transfer!, T, tspan, p)
        n_bm = @benchmark solve($naive_prob, save_everystep=false)
        n_median = median(n_bm.times)
        println("Complete naive approach at size $n: median time = $n_median")

        filt_prob = ODEProblem(heat_transfer_conv!, T, tspan, p)
        filt_bm = @benchmark solve($filt_prob, save_everystep=false)
        filt_median = median(filt_bm.times)
        println("Complete imfilter approach at size $n: median time = $filt_median")

        CNN_prob = ODEProblem(heat_transfer_CNN!, T_ch, tspan, p_cnn)
        CNN_bm = @benchmark solve($CNN_prob, save_everystep=false)
        CNN_median = median(CNN_bm.times)
        println("Complete CNN approach at size $n: median time = $CNN_median")

        naive[idx] = n_bm
        imfilt[idx] = filt_bm
        cnn[idx] = CNN_bm
        idx += 1
    end
    return naive, imfilt, cnn

end


scale_problem_size()

    # @gif for t in 1:length(conv_sol)
        #     heatmap(conv_sol[t])
        # end

n = 16
    # Build empty array of temperature values
T = zeros(n,n)
# Store at some arbitrary centered index a large temperature to be
# transferred throughout the space.
T[Int(floor(n/2)),Int(floor(n//2))] = 1.

# Specify arbitrary thermal conductivities
C = ones(n,n)

# Assume no heat generation (system is insulated)
Q = zeros(n,n)

# Unused at the moment, but corresponds to the contact area between cells
A = ones(n,n)

# Heat transfer kernal
K =  [ 0 1 0; 1 -4 1; 0 1 0.]

# Define parameters for ODE solve
p =  (1.,K, C, Q, A)

# 10 second run
tspan = (0.,10.)

naive_prob = ODEProblem(heat_transfer!, T, tspan, p)
n_bm = @benchmark solve($naive_prob, Rosenbrock23(), save_everystep=false)
n_median = median(n_bm.times)
