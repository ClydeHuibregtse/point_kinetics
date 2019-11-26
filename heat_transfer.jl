using DifferentialEquations, BenchmarkTools;
# using DSP, Knet, ImageFiltering;
using ImageFiltering;
using Plots;
using CuArrays, Flux;

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



n = 12
T = zeros(n,n)
# T[Int(floor(n/2)+1),Int(floor(n/2)+1)] = 100.
# T[Int(floor(n/2)+1),Int(floor(n/2)+50)] = 100.

C = ones(n,n)
Q = zeros(n,n)
Q[5,5] = 1.
A = ones(n,n)

K =  [ 0 1 0; 1 -4 1; 0 1 0.]
p =  [1.,K, C, Q, A]

tspan = (0.,10.)


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





@gif for t in 1:length(loop_sol)
    heatmap(loop_sol[t])
end

@gif for t in 1:length(conv_sol)
    heatmap(conv_sol[t])
end

cnn_out = convert(Array{Float64}, cnn_sol)


@gif for t in 1:size(cnn_sol)[end]
    heatmap(cnn_out[:,:,1,1,t])
end








# loop_prob = ODEProblem(heat_transfer!, T, tspan, p)
# loop_sol = solve(loop_prob)
#
# conv_prob = ODEProblem(heat_transfer_conv!, T, tspan, p)
# conv_sol = solve(conv_prob)

# @benchmark solve(loop_prob)
#
# @benchmark solve(conv_prob)
#
# @benchmark solve(cnn_prob)
