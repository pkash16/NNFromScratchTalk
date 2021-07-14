### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ e14a0598-df23-11eb-18ac-f72bae136d78
using LinearAlgebra

# ╔═╡ a77e4c14-07ad-446b-af45-9be5a0abd38c
using Statistics

# ╔═╡ 9c3993a9-f108-4501-ac58-15d703827fc1
using Plots

# ╔═╡ bbd2fc03-6317-4598-93f5-e0300219d91d
using PlutoUI

# ╔═╡ 0eeead2a-4db1-4929-a691-cac2d0802fff
using MLDatasets

# ╔═╡ 70e36e9d-ca13-4c82-8b75-f5e68af60391
using Flux

# ╔═╡ 1211d685-5011-4140-9ea7-6f2fedfbb2b7
using MIRT

# ╔═╡ 3c04f103-f945-4168-96f5-c26f139819ad
html"<button onclick='present()'>present</button>"

# ╔═╡ 5f50c28c-aaff-4fda-a039-5258e76783c0
md"""
# Introduction
Building a Neural Network from Scratch in Julia


## What is a Neural Network?
$(Resource("https://iq.opengenus.org/content/images/2019/03/fc.jpg"))

A simple fully connected layer example output:

``a_{21} = \begin{bmatrix} a_{01} & a_{02} & a_{03} \end{bmatrix} * \begin{bmatrix} w_{11}^{01}\\ w_{11}^{02}\\ w_{11}^{03} \end{bmatrix} + b_{11}``
"""

# ╔═╡ 3127081d-347f-4a8c-892b-2985386b0e1c
begin
	a11 = 0.58
	a12 = 0.35
	a13 = 3.62
	a14 = 8.32
	w1 = 0.2
	w2 = 0.5
	w3 = 1
	w4 = 0.003
	b21 = 0.25
	md"""

	### Example
	Given variables ``a_{11}, a_{12}, a_{13}, a_{14}`` and weights ``w_1, w_2, w_3, w_4`` for node ``a_{21}``, as well as bias ``b_{21}``, find the output of ``a_{21}``


	"""
	
end

# ╔═╡ e9f05b98-4e41-411a-9828-4f95256cc1d5
example1 = 0  #missing

# ╔═╡ c981f970-e767-42b2-8303-007c65e593f4
md"""
## Basic Neural Network Design
Initializing Parameters: Fully connected layers!
Let's define a function `init_param` to randomly create a dictionary holding weights and biases for a series of fullly connected layers.
"""

# ╔═╡ 02acb10f-6967-46fe-a1b5-4877ccb92000
function init_param(layer_dimensions)

	param = Dict()

	for l=2:length(layer_dimensions)

		param[string("W_" , string(l-1))] = rand(layer_dimensions[l] ,
				layer_dimensions[l-1])*0.1
		param[string("b_" , string(l-1))] = zeros(layer_dimensions[l] , 1)

	end

	return param
end

# ╔═╡ b9ce8c31-0d4e-4390-bfd9-bf873dbaec68
md"""
Feel free to play around with the `init_param` function to undestand how it works.

Example inputs:

	init_param((3,3,3)) # 2 fully connected layers, input size 3 output size 3
	init_param((1,20,20,14,1)) # 4 fully connected layers, input size 1 output size 1

"""

# ╔═╡ d8c12772-2b0a-45e8-b997-a9c4fb7379bc
#play around with me here!
init_param((4, 5, 4, 1)) 

# ╔═╡ 19ed0f3b-4bc3-4f45-bc12-b18b233302cf
md"""

Example: Find the output of the fully connected network given parameters `params2` and `input`. Hint: the dot product can be extended to another simple linear algebra operation for a simple operation.

"""

# ╔═╡ e8f66c9f-1c06-4db4-ba9b-c9257ff7651a
begin
	
	params2 = init_param((3,5,3)) #do not edit
	input = randn(3,1) # do not edit
	
	example2 =  0#missing
	
end

# ╔═╡ ffaa7a50-6adc-43e7-b3fd-e072032ed078
md"""
## Activation Functions

So far we have played around with fully-connected layers and their inputs and outputs. We've learned that we can fully characterize the outputs of fully connected layers as matrix-vector multiplications and additions. Essentially, we can consider the output of a fully-connected neural network as a `linear operator`, and we can solve for the optimal parameters using a `least squares` approach.


To improve performance, we would like to add some `non-linearity` to our data pathway. To do this, we can introduce what is known as an `activation function` at the end of each node. We will discuss 2 activation functions below.

#### Rectified Linear Unit (ReLU)
``y`` = ReLu(``x``) = 
``max(0, x)``

#### Sigmoid Function
``y`` = sigmoid(``x``) = \
``\frac{1}{1 + e^{-x}}``


#### Implementing sigmoid and relu: Using Julia's `dot` notation
Implement the relu and the sigmoid functions. Use the `dot` notation to allow inputs as vectors/matrices. 

Example of `dot` notation:
	
	y = abs(x) #takes in a single element
	y = abs.(x) #takes in a vector or matrix of elements

"""

# ╔═╡ 4d81c782-4cd6-4d47-9a1a-fbbf4144a107
#takes the input X and returns output tuple sigmoid(X), X
function sigmoid(X)
	sigma = 1 ./(1 .+ exp.(.-X))
	return sigma , X 
	#return 0, X # missing
end


# ╔═╡ 82adc6bb-8c29-4b16-8a62-6f09eaac2c25
begin
example3_input = randn(20,1)
example3 = sigmoid(example3_input)
md""" ##### Write the sigmoid function: """
end

# ╔═╡ 11fc4e5e-1935-4952-8bd9-a170718d4eea
#takes the input X and returns output tuple relu(X), X
function relu(X)
	rel = max.(0,X)
	return rel , X
	#return 0 , X # missing
end


# ╔═╡ 2ede9ea2-a763-4d96-b42e-fe3a7c0ec53c
begin
example4_input = randn(20,1)
example4 = relu(example4_input)
md""" ##### Write the ReLU function: """
end

# ╔═╡ 9eebff00-03e6-4554-9036-108a887bd90a
md"""
 ## Implementing the forward pass

Implement the `forward_linear` function. This function takes weights w, input A, and bias b and returns output w*A + b, and a cache (A, w, b)

"""

# ╔═╡ 06b02c90-b1a7-4a2a-975d-10079b19cdd9
function forward_linear(A , w , b)

    Z = w*A .+ b
    cache = (A , w , b)

    return Z,cache

end

# ╔═╡ 7dd03c52-0c7c-4fb9-984c-ee41f01a8d4e
md"""

The `calculate_activation_forward` function takes into account the `forward_linear` function as well as the activation functions we defined earlier, and returns output `activation(forward_linear(input))`
"""

# ╔═╡ 4dfc2eb3-638a-446d-bd1f-04cd8bb9bd54
function calculate_activation_forward(A_pre , W , b , function_type)

    if (function_type == "sigmoid")

        Z , linear_step_cache = forward_linear(A_pre , W , b)
        A , activation_step_cache = sigmoid(Z)

    elseif (function_type == "relu")

        Z , linear_step_cache = forward_linear(A_pre , W , b)
        A , activation_step_cache = relu(Z)

    end

    cache = (linear_step_cache , activation_step_cache)
    return A , cache

end

# ╔═╡ df208c60-816b-4f84-a041-d363b87d1c90
md"""

Finally, we define the `model_forward_step` function that takes in our `params` and runs through in a loop the activation_forwards. We also append the `cache` for all the steps. The reason we hold onto the `cache` intermediate variables for all values is because of the `backpropogation` step which we will define later, and is necesary for calculating gradients.

"""

# ╔═╡ 7e0b5831-7506-4823-9324-5623211a8b2f

function model_forward_step(X , params)

    all_caches = []
    A = X
    L = length(params)/2

    for l=1:L-1
        A_pre = A
        A , cache = calculate_activation_forward(A_pre , params[string("W_" , string(Int(l)))] , params[string("b_" , string(Int(l)))] , "relu")
        push!(all_caches , cache)
    end 
    A_l , cache = calculate_activation_forward(A , params[string("W_" , string(Int(L)))] , params[string("b_" , string(Int(L)))] , "sigmoid")
    push!(all_caches , cache)


    return A_l , all_caches 

end

# ╔═╡ 60491abb-1564-4fcc-a9e9-949dad66f977
md"""
## Cost Functions
A sample cost function is the binary cross-entropy loss:

J = ``-\frac{1}{m}\sum_i^m{Y^ilog(\hat{Y}^i) + (1 - Y^i)log(1 - \hat{Y}^i)}``
"""

# ╔═╡ a6d871bf-bb1b-4b5d-b488-1945183b8582
function cost_function_bce(AL , Y)
    
    cost = -mean(Y.*log.(AL) + (1 .- Y).*log.(1 .- AL))

    return cost 

end

# ╔═╡ 90917a6b-67e9-44e2-932c-34e7f7c6a94d
md"""
Another cost function is the least squares loss: \
``J = \frac{1}{m} \sum_i^m (\hat{Y}^i - Y^i)^2``
"""

# ╔═╡ f20e9d8f-defc-4081-bab0-0cc71dc53dcf
function cost_function_least_squares(Al, Y)
	cost = mean( (Al .- Y) .^ 2)
	return cost
end

# ╔═╡ 2251dff8-1877-4163-ad36-a005fe898c32
md"""
## An Introduction to backpropogation

What is backpropogation?

This explanation has not yet been written. But the code is fully described below. Before JC this will be fixed!


"""

# ╔═╡ 5a1dc433-9cf5-4880-b2d7-552c09e0c6fb

function backward_linear_step(dZ , cache)

    A_prev , W , b = cache

    m = size(A_prev)[2]
    
    dW = dZ * (A_prev')/m
    db = sum(dZ , dims = 2)/m
    dA_prev = (W')* dZ
    return dW , db , dA_prev 

end

# ╔═╡ 913aa2fb-ef40-4d1a-a684-fdad9591319e
begin
	
	function backward_relu(dA , cache_activation)
    return dA.*(cache_activation.>0)
end 

function backward_sigmoid(dA , cache_activation)
    return dA.*(sigmoid(cache_activation)[1].*(1 .- sigmoid(cache_activation)[1]))
end
	
end

# ╔═╡ 0e3bc617-d8ad-4587-ad8b-e1dafec98d10
function backward_activation_step(dA , cache , activation)

    linear_cache , cache_activation = cache
    if (activation == "relu")

        dZ = backward_relu(dA , cache_activation)
        dW , db , dA_prev = backward_linear_step(dZ , linear_cache)

    elseif (activation == "sigmoid")

        dZ = backward_sigmoid(dA , cache_activation)
        dW , db , dA_prev = backward_linear_step(dZ , linear_cache)

    end 

    return dW , db , dA_prev

end 

# ╔═╡ 2d150849-439a-4e3a-90cc-abf6802c5a45
md"""
## Putting the Pieces together: Modeling the backpropogation
"""

# ╔═╡ 0dc0ac84-2763-4401-9e8f-1805d3fa65b1
function model_backwards_step(A_l , Y , caches)

    grads = Dict()

    L = length(caches)

    m = size(A_l)[2]

    Y = reshape(Y , size(A_l))
    #dA_l = (-(Y./A_l) .+ ((1 .- Y)./( 1 .- A_l)))
	dA_l = -2*Y .+ 2*A_l
	
    current_cache = caches[L]
    grads[string("dW_" , string(L))] , grads[string("db_" , string(L))] , grads[string("dA_" , string(L-1))] = backward_activation_step(dA_l , current_cache , "sigmoid")
    for l=reverse(0:L-2)
        current_cache = caches[l+1]
        grads[string("dW_" , string(l+1))] , grads[string("db_" , string(l+1))] , grads[string("dA_" , string(l))] = backward_activation_step(grads[string("dA_" , string(l+1))] , current_cache , "relu")

    end 

    return grads 

end

# ╔═╡ ef50b40d-e0a8-4cf6-99f6-32db41e2921a
function update_param(parameters , grads , learning_rate)

    L = Int(length(parameters)/2)

    for l=0:(L-1)

        parameters[string("W_" , string(l+1))] -= learning_rate.*grads[string("dW_" , string(l+1))]
        parameters[string("b_",string(l+1))] -= learning_rate.*grads[string("db_",string(l+1))]

    end 

    return parameters

end

# ╔═╡ f8173203-a448-458e-870b-52302a822fe5
md"""
## Training the neural network and checking the accuracy
"""

# ╔═╡ 328afa94-d6bb-4197-9fda-ff5b61efc62f
function check_accuracy(A_l, test_y)
	num_correct = 0
	num_total = size(A_l, 2)
	
	correct_indices = []
	incorrect_indices = []
	
	for idx = 1:size(A_l, 2)
		_, network_idx = findmax(A_l[:,idx])
		_, corr_idx = findmax(test_y[:,idx])
		
		if corr_idx == network_idx
			push!(correct_indices, (idx, network_idx, corr_idx))
			num_correct = num_correct + 1
		else
			push!(incorrect_indices, (idx, network_idx, corr_idx))
		end
	end
	return num_correct / num_total * 100, correct_indices, incorrect_indices
end

# ╔═╡ 6f09f404-3d5b-480d-be59-50c0edc88be3
function train_nn(layers_dimensions , X , Y , learning_rate , n_iter)

    params = init_param(layers_dimensions)
    costs = []
    iters = []
    accuracy = []
	
    for i=1:n_iter
        A_l , caches  = model_forward_step(X , params)
 		cost = cost_function_least_squares(A_l, Y)
		
        acc, _, _ = check_accuracy(A_l , Y)
        grads  = model_backwards_step(A_l , Y , caches)
        params = update_param(params , grads , learning_rate)
		
        println("Iteration ->" , i)
        println("Cost ->" , cost)
        println("Accuracy -> " , acc)
        push!(iters , i)
        push!(costs , cost)
        push!(accuracy , acc)
        
    end 
	
	return params , costs, accuracy

end

# ╔═╡ abce2d41-7074-4df0-834f-aa96dee6f34f
begin
	train_x, train_y = MNIST.traindata()
	
	truncate = 1000
	train_x = train_x[:, :, 1:truncate]
	train_y = train_y[1:truncate]
	test_x, test_y = MNIST.testdata()
	
	md"""
	## Testing the network on a toy dataset: MNIST

	The `MLDatasets` package allows us to quickly pull training and testing data from the MNIST dataset. The code hidden in this block loads the MNIST data and stores it in the variables `train_x`, `train_y`, `test_x`, and `test_y`
	"""
end

# ╔═╡ f68e451c-9a23-416c-abcd-11b635186808
begin
	train_y_onehot = Flux.onehotbatch(train_y, [0,1,2,3,4,5,6,7,8,9])
	params_MNIST, costs_MNIST, accuracy_MNIST = train_nn((784, 16, 16, 10), reshape(train_x, 28*28, :), train_y_onehot, 0.25, 2000)
end

# ╔═╡ 25000b12-98c8-4aa0-a270-69aa10ec4f05
plot(costs_MNIST)

# ╔═╡ 8eea3830-51f8-4d55-a782-a1690db94b4d
begin
A_l, _ = model_forward_step(reshape(test_x, 28*28, :), params_MNIST)
test_y_onehot = Flux.onehotbatch(test_y, [0,1,2,3,4,5,6,7,8,9])
acc, correct_indices, incorrect_indices = check_accuracy(A_l, test_y_onehot)
end

# ╔═╡ e747ecdd-bb4e-4991-9086-efcafa1242f4
md"""
Accuracy: $(acc) %
"""

# ╔═╡ 3648337e-4f84-4f4f-8cc7-1416877476d4
md"""
## Visualization of Results
"""

# ╔═╡ 88a16479-d006-4d84-ac09-60c3b28d9f8f
md"""
Let's visualize some of the correct and incorrect indices.
"""

# ╔═╡ 10b0b1d9-2d8d-4d16-98ff-695e0a3dec6f
md"""
### Network Failures
"""

# ╔═╡ b6a1c6cc-26bc-4a95-9236-3e4a5d7bfcd9
@bind incorrect_idx Slider(1:size(incorrect_indices,1), show_value=true)

# ╔═╡ 1a1c4f5f-5a48-45f6-830b-bc66f4229c7a
jim(test_x[:, :, incorrect_indices[incorrect_idx][1]])

# ╔═╡ c3beb59b-3089-4a71-9267-dce52f17a3c0
md"""
Network Predicted: $(incorrect_indices[incorrect_idx][2] - 1) \
Actual Label: $(incorrect_indices[incorrect_idx][3] - 1)
"""

# ╔═╡ 6d90f99d-f37a-479e-9353-4d43cd812fa8
md"""
### Network Successes
"""

# ╔═╡ 4bafac2a-a127-4f7c-bce7-af6513c5ab8c
@bind correct_idx Slider(1:size(correct_indices,1), show_value=true)

# ╔═╡ 829925cd-4cfa-436c-b733-0d212bf3f481
jim(test_x[:, :, correct_indices[correct_idx][1]])

# ╔═╡ 50ea4196-b225-4d66-87db-08199c47eae0
md"""
## 
"""

# ╔═╡ 2cc0008b-591a-4a4e-b502-b7a9d217e2e2
md"""
### Helper Functions
"""

# ╔═╡ 81245ae8-b79b-4adb-8335-85e680d71862
hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]))

# ╔═╡ f6cf25d2-9c46-4961-89de-6331c4952b3f
answer(text) = Markdown.MD(Markdown.Admonition("hint", "Answer", [text]))

# ╔═╡ 0fb74ac1-7014-41d5-b8dc-825768bdaf16
almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]))

# ╔═╡ b9fb0441-06f8-4da4-a1c5-7dd61155a0d7
still_missing(text=md"Replace `missing` with your answer.") = Markdown.MD(Markdown.Admonition("warning", "Here we go!", [text]))

# ╔═╡ cd200ce8-ffb8-49fd-baeb-564719d2efd5
keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]))

# ╔═╡ d0886291-a616-4374-9c3a-4b6822a1bbf8
not_defined(variable_name) = Markdown.MD(Markdown.Admonition("danger", "Oopsie!", [md"Make sure that you define a variable called **$(Markdown.Code(string(variable_name)))**"]))

# ╔═╡ 33e527ab-4a9b-4251-b0cc-0722f74f2209
yays = [md"You did it!" md"Well done!" md"Nice job!" md"Cool stuff."]

# ╔═╡ 89bfc20f-9c16-4f43-b4a6-e972e159b0f3
correct(text=rand(yays)) = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]))

# ╔═╡ 0b84ee1f-5b75-4998-9b32-879f5e044450
begin
	example1_correct = [w1 w2 w3 w4] * [a11;a12;a13;a14] .+ b21
	example1_nobias = [w1 w2 w3 w4] * [a11;a12;a13;a14]
	
	if example1 == example1_nobias
		almost(md"Don't forget that the node has a `bias` term ``b_{21}`` that needs to be added to the output.")
	elseif example1 == example1_correct
		correct()
	elseif example1 == 0
		still_missing()
	else
		keep_working(md"Remember that the output of a fully connected node can be considered as a vector-vector dot product plus the ``bias`` term similar to the example above.")	
	end
end

# ╔═╡ 1e9032b1-d05a-4aa0-af68-1db938bd825a
begin
	example2_correct = params2["W_2"] * (params2["W_1"] * input + params2["b_1"]) + params2["b_2"]
	example2_nobias = params2["W_2"] * (params2["W_1"] * input)
	
	if example2 == example2_correct
		correct()
	elseif example2 == 0
		still_missing()
	else
		keep_working(md"Remember that the output of a fully connected layer output can be considered as a matrix-vector product plus the ``bias`` vector.")	
	end
end

# ╔═╡ 81ebabf9-9de3-439f-9df3-b20da1fb86e9
begin
	
function sigmoid_correct(X)
	sigma = 1 ./(1 .+ exp.(.-X))
	return sigma , X 
end
	
	

example3_correct = sigmoid_correct(example3_input)

if example3 == example3_correct
	correct()
elseif example3 == (0, example3_input)
	still_missing()
else
	keep_working(md"Take a look at the sigmoid function again. Remember to use `dot` notation and to return the tuple sigmoid(X), X ")	
end

	
end

# ╔═╡ cc5bae6c-81d5-4481-8b81-fe588a95ca7f
begin
	
		
function relu_correct(X)
	rel = max.(0,X)
	return rel , X
end


example4_correct = relu_correct(example4_input)

if example4 == example4_correct
	correct()
elseif example4 == (0, example4_input)
	still_missing()
else
	keep_working(md"Take a look at the sigmoid function again. Remember to use `dot` notation and to return the tuple sigmoid(X), X ")	
end

	
end
	

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MIRT = "7035ae7a-3787-11e9-139a-5545ed3dc201"
MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Flux = "~0.12.1"
MIRT = "~0.14.1"
MLDatasets = "~0.5.7"
Plots = "~1.18.1"
PlutoUI = "~0.7.9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AVSfldIO]]
deps = ["FileIO"]
git-tree-sha1 = "f47f9477b6aaf284212bd6745ed7050616bdaec4"
uuid = "b6189060-daf9-4c28-845a-cc0984b81781"
version = "0.2.1"

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[BFloat16s]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "4af69e205efc343068dc8722b8dfec1ade89254a"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.1.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BinDeps]]
deps = ["Libdl", "Pkg", "SHA", "URIParser", "Unicode"]
git-tree-sha1 = "1289b57e8cf019aede076edab0587eb9644175bd"
uuid = "9e28174c-4ba2-5203-b857-d8d62c4213ee"
version = "1.0.2"

[[BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[Blosc]]
deps = ["Blosc_jll"]
git-tree-sha1 = "84cf7d0f8fd46ca6f1b3e0305b4b4a37afe50fd6"
uuid = "a74b3585-a348-5f62-a45c-50e91977d574"
version = "0.7.0"

[[Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "e747dac84f39c62aff6956651ec359686490134e"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.0+0"

[[BufferedStreams]]
deps = ["Compat", "Test"]
git-tree-sha1 = "5d55b9486590fdda5905c275bb21ce1f0754020f"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.0.0"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c3598e525718abcc440f69cc6d5f60dda0a1b61e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.6+5"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "DataStructures", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "MacroTools", "Memoize", "NNlib", "Printf", "Random", "Reexport", "Requires", "SparseArrays", "Statistics", "TimerOutputs"]
git-tree-sha1 = "6893a46f357eabd44ce0fc1f9a264120a1a3a732"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "2.6.3"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e2f47f6d8337369411569fd45ae5753ca10394c6"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.0+6"

[[CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "85c579fa131b5545eef874a5b413bb3b783e21c6"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "0.8.21"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "dcc25ff085cf548bc8befad5ce048391a7c07d40"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.10.11"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random", "StaticArrays"]
git-tree-sha1 = "ed268efe58512df8c7e224d2e170afd76dd6a417"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.13.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "32a2b8af383f11cbb65803883837a149d10dfe8a"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.10.12"

[[ColorVectorSpace]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "StatsBase"]
git-tree-sha1 = "4d17724e99f357bfd32afa0a9e2dda2af31a9aea"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.8.7"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dc7dedc2c2aa9faf59a55c622760a25cbefbe941"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.31.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "6d1c23e740a586955645500bbec662476204a52c"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.1"

[[CustomUnitRanges]]
git-tree-sha1 = "537c988076d001469093945f3bd0b300b8d3a7f3"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.1"

[[DSP]]
deps = ["FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "2a63cb5fc0e8c1f0f139475ef94228c7441dc7d0"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.6.10"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataDeps]]
deps = ["BinaryProvider", "HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "4f0e41ff461d42cfc62ff0de4f1cd44c6e6b3771"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.7"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "214c3fcac57755cfda163d91c58893a8723f93e9"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.0.2"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "92d8f9f208637e8d2d28c664051a00569c01493d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.1.5+1"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "LibVPX_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3cc57ad0a213808473eafef4845a74766242e05f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.3.1+4"

[[FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "70a0cfd9b1c86b0209e38fbfe6d8231fd606eeaf"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.1"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "f985af3b9f4e278b1d24434cbb546d6092fca661"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.3"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3676abafff7e4ff07bbd2c42b3d8201f31653dcc"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.9+8"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "256d8e6188f3f1ebfa1a5d17e072a0efafa8c5bf"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.10.1"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "693210145367e7685d8604aee33d9bfb85db8b31"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.11.9"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Flux]]
deps = ["AbstractTrees", "Adapt", "CUDA", "CodecZlib", "Colors", "DelimitedFiles", "Functors", "Juno", "LinearAlgebra", "MacroTools", "NNlib", "Pkg", "Printf", "Random", "Reexport", "SHA", "Statistics", "StatsBase", "Test", "ZipFile", "Zygote"]
git-tree-sha1 = "287705d01ab510afe075b0165a159b9e5a4bf082"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.12.1"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "35895cf184ceaab11fd778b4590144034a167a2f"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.1+14"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "e2af66012e08966366a43251e1fd421522908be6"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.18"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "cbd58c9deb1d304f5a245a0b7eb841a2560cfec6"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.1+5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Functors]]
deps = ["MacroTools"]
git-tree-sha1 = "a7bb2af991c43dcf5c3455d276dd83976799634f"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.1"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GPUArrays]]
deps = ["AbstractFFTs", "Adapt", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "df5b8569904c5c10e84c640984cfff054b18c086"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "6.4.1"

[[GPUCompiler]]
deps = ["DataStructures", "ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Scratch", "Serialization", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "ef2839b063e158672583b9c09d2cf4876a8d3d55"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.10.0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "b83e3125048a9c3158cbb7ca423790c7b1b57bea"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.57.5"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e14907859a1d3aee73a019e7b3c98e9e7b8b5b3e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.57.3+0"

[[GZip]]
deps = ["Libdl"]
git-tree-sha1 = "039be665faf0b8ae36e089cd694233f5dee3f7d6"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.5.1"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "15ff9a14b9e1218958d3530cc288cf31465d9ae2"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.3.13"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "47ce50b742921377301e15005c96e979574e130b"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.1+0"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "2c1cf4df419938ece72de17f368a021ee162762e"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HDF5]]
deps = ["Blosc", "Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "1d18a48a037b14052ca462ea9d05dee3ac607d23"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.15.5"

[[HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "fd83fa0bde42e01952757f01149dd968c06c4dba"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.0+1"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "MbedTLS", "Sockets"]
git-tree-sha1 = "c7ec02c4c6a039a98a15f955462cd7aea5df4508"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.8.19"

[[IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "95215cd0076a150ef46ff7928892bc341864c73c"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.3"

[[IdentityRanges]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be8fcd695c4da16a1d6d0cd213cb88090a150e3b"
uuid = "bbac6d45-d8f3-5730-bfe4-7a449cd117ca"
version = "0.3.1"

[[ImageCore]]
deps = ["AbstractFFTs", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "db645f20b59f060d8cfae696bc9538d13fd86416"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.8.22"

[[ImageFiltering]]
deps = ["CatIndices", "ColorVectorSpace", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageCore", "LinearAlgebra", "OffsetArrays", "Requires", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "bf96839133212d3eff4a1c3a80c57abc7cfbf0ce"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.6.21"

[[ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "IdentityRanges", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "d966631de06f36c8cd4bec4bb2e8fa731db16ed9"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.8.12"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "1470c80592cf1f0a35566ee5e93c5f8221ebc33a"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.3"

[[Intervals]]
deps = ["Dates", "Printf", "RecipesBase", "Serialization", "TimeZones"]
git-tree-sha1 = "323a38ed1952d30586d0fe03412cde9399d3618b"
uuid = "d8418881-c3e1-53bb-8760-2df7ec849ed5"
version = "1.5.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LLVM]]
deps = ["CEnum", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "f57ac3fd2045b50d3db081663837ac5b4096947e"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "3.9.0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[LibVPX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "12ee7e23fa4d18361e7c2cde8f8337d4c3101bc7"
uuid = "dd192d2f-8180-539f-9fb4-cc70b1dcf69a"
version = "1.10.0+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LinearMaps]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "374b717add8a818363241b403c62b218a3368dd2"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "2.7.0"

[[LinearMapsAA]]
deps = ["LinearAlgebra", "LinearMaps", "SparseArrays", "Test"]
git-tree-sha1 = "40e6d84c41fdb4a62ba2a0c934ac8f01fca82ad6"
uuid = "599c1a8e-b958-11e9-0d14-b1e6b2ecea07"
version = "0.6.5"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "5c62992f3d46b8dce69bdd234279bb5a369db7d5"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.1"

[[MIRT]]
deps = ["AVSfldIO", "Distributed", "FFTW", "FileIO", "FillArrays", "ImageFiltering", "ImageTransformations", "Interpolations", "LinearAlgebra", "LinearMapsAA", "MIRTio", "MIRTjim", "NFFT", "Random", "Reexport", "SharedArrays", "SparseArrays", "SpecialFunctions", "Wavelets"]
git-tree-sha1 = "9d522b1f4ceb4b0b8f4411b0efb3c6a09740aca3"
uuid = "7035ae7a-3787-11e9-139a-5545ed3dc201"
version = "0.14.1"

[[MIRTio]]
deps = ["HDF5", "Test"]
git-tree-sha1 = "d3804313cb3b31818f158fcef2002e5b9138a7c9"
uuid = "274281c4-ae57-11e9-2014-65d3d5d9358c"
version = "0.3.1"

[[MIRTjim]]
deps = ["FFTViews", "LaTeXStrings", "MosaicViews", "Plots", "REPL"]
git-tree-sha1 = "a3397e47de9ec20b12fd31b0e5bacdcd06f77080"
uuid = "170b2178-6dee-4cb0-8729-b3e8b57834cc"
version = "0.7.0"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "c253236b0ed414624b083e6b72bfe891fbd2c7af"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+1"

[[MLDatasets]]
deps = ["BinDeps", "ColorTypes", "DataDeps", "DelimitedFiles", "FixedPointNumbers", "GZip", "MAT", "Requires"]
git-tree-sha1 = "c7eaf044f72245ffa50722c6d0813da9df0ff2f0"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.5.7"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[MappedArrays]]
git-tree-sha1 = "18d3584eebc861e311a552cbb67723af8edff5de"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.0"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Mocking]]
deps = ["ExprTools"]
git-tree-sha1 = "916b850daad0d46b8c71f65f719c49957e9513ed"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.1"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NFFT]]
deps = ["CUDA", "Distributed", "FFTW", "Graphics", "LinearAlgebra", "Random", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "44a74e77a99450c344edea4cae94d45a2f66ead8"
uuid = "efe261a4-0d2b-5849-be55-fc731d526b0d"
version = "0.6.1"

[[NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "7e6f31cfa39b1ff1c541cc8580b14b0ff4ba22d0"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.23"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2bf78c5fd7fa56d2bbf1efbadd45c1b8789e6f57"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.2"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fa5e78929aebc3f6b56e1a88cf505bb00a354c4"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.8"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "c8abc88faa3f7a3950832ac5d6e690881590d6dc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.0"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "501c20a63a34ac1d015d5304da0e645f42d91c9f"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.11"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "b93181645c1209d912d5632ba2d0094bc00703ad"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.18.1"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[Polynomials]]
deps = ["Intervals", "LinearAlgebra", "OffsetArrays", "RecipesBase"]
git-tree-sha1 = "0b15f3597b01eb76764dd03c3c23d6679a3c32c8"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "1.2.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
git-tree-sha1 = "37d210f612d70f3f7d57d488cb3b6eff56ad4e41"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.0"

[[RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "2a7a2469ed5d94a98dea0e85c46fa653d76be0cd"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.3.4"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rotations]]
deps = ["LinearAlgebra", "StaticArrays", "Statistics"]
git-tree-sha1 = "2ed8d8a16d703f900168822d83699b8c3c1a5cd8"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.0.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a50550fa3164a8c46747e62063b4d774ac1bcf49"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.5.1"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "a43a7b58a6e7dc933b2fa2e0ca653ccf8bb8fd0e"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.6"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2f6792d523d7448bbe2fec99eca9218f06cc746d"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.8"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "000e168f5cc9aded17b6999a560b7c11dda69095"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.0"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "8ed4a3ea724dac32670b062be3ef1c1de6773ae8"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.4.4"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "52c5f816857bfb3291c7d25420b1f4aca0a74d18"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.0"

[[TimeZones]]
deps = ["Dates", "Future", "LazyArtifacts", "Mocking", "Pkg", "Printf", "RecipesBase", "Serialization", "Unicode"]
git-tree-sha1 = "81753f400872e5074768c9a77d4c44e70d409ef0"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.5.6"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "209a8326c4f955e2442c07b56029e88bb48299c7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.12"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "7c53c35547de1c5b9d46a4797cf6d8253807108c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.5"

[[URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wavelets]]
deps = ["DSP", "FFTW", "LinearAlgebra", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "e5903fb2bf93697a79d01383618ea0855256a337"
uuid = "29a6e085-ba6d-5f35-a997-948ac2efa89a"
version = "0.9.3"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "c3a5637e27e914a7a445b8d0ad063d701931e9f7"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "531474afbc343c3c7cb9b71c2771813c6defd550"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.14"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "acc685bcf777b2202a904cdcb49ad34c2fa1880c"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.14.0+4"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7a5780a0d9c6864184b3a2eeeb833a0c871f00ab"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "0.1.6+4"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d713c1ce4deac133e3334ee12f4adff07f81778f"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2020.7.14+2"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "487da2f8f2f0c8ee0e83f39d13037d6bbf0a45ab"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.0.0+3"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═e14a0598-df23-11eb-18ac-f72bae136d78
# ╠═a77e4c14-07ad-446b-af45-9be5a0abd38c
# ╠═9c3993a9-f108-4501-ac58-15d703827fc1
# ╠═bbd2fc03-6317-4598-93f5-e0300219d91d
# ╠═0eeead2a-4db1-4929-a691-cac2d0802fff
# ╠═70e36e9d-ca13-4c82-8b75-f5e68af60391
# ╠═1211d685-5011-4140-9ea7-6f2fedfbb2b7
# ╟─3c04f103-f945-4168-96f5-c26f139819ad
# ╟─5f50c28c-aaff-4fda-a039-5258e76783c0
# ╟─3127081d-347f-4a8c-892b-2985386b0e1c
# ╠═e9f05b98-4e41-411a-9828-4f95256cc1d5
# ╟─0b84ee1f-5b75-4998-9b32-879f5e044450
# ╟─c981f970-e767-42b2-8303-007c65e593f4
# ╟─02acb10f-6967-46fe-a1b5-4877ccb92000
# ╟─b9ce8c31-0d4e-4390-bfd9-bf873dbaec68
# ╟─d8c12772-2b0a-45e8-b997-a9c4fb7379bc
# ╟─19ed0f3b-4bc3-4f45-bc12-b18b233302cf
# ╠═e8f66c9f-1c06-4db4-ba9b-c9257ff7651a
# ╟─1e9032b1-d05a-4aa0-af68-1db938bd825a
# ╟─ffaa7a50-6adc-43e7-b3fd-e072032ed078
# ╟─82adc6bb-8c29-4b16-8a62-6f09eaac2c25
# ╠═4d81c782-4cd6-4d47-9a1a-fbbf4144a107
# ╟─81ebabf9-9de3-439f-9df3-b20da1fb86e9
# ╟─2ede9ea2-a763-4d96-b42e-fe3a7c0ec53c
# ╠═11fc4e5e-1935-4952-8bd9-a170718d4eea
# ╟─cc5bae6c-81d5-4481-8b81-fe588a95ca7f
# ╟─9eebff00-03e6-4554-9036-108a887bd90a
# ╠═06b02c90-b1a7-4a2a-975d-10079b19cdd9
# ╟─7dd03c52-0c7c-4fb9-984c-ee41f01a8d4e
# ╠═4dfc2eb3-638a-446d-bd1f-04cd8bb9bd54
# ╟─df208c60-816b-4f84-a041-d363b87d1c90
# ╠═7e0b5831-7506-4823-9324-5623211a8b2f
# ╟─60491abb-1564-4fcc-a9e9-949dad66f977
# ╠═a6d871bf-bb1b-4b5d-b488-1945183b8582
# ╟─90917a6b-67e9-44e2-932c-34e7f7c6a94d
# ╠═f20e9d8f-defc-4081-bab0-0cc71dc53dcf
# ╟─2251dff8-1877-4163-ad36-a005fe898c32
# ╠═5a1dc433-9cf5-4880-b2d7-552c09e0c6fb
# ╠═913aa2fb-ef40-4d1a-a684-fdad9591319e
# ╠═0e3bc617-d8ad-4587-ad8b-e1dafec98d10
# ╟─2d150849-439a-4e3a-90cc-abf6802c5a45
# ╠═0dc0ac84-2763-4401-9e8f-1805d3fa65b1
# ╠═ef50b40d-e0a8-4cf6-99f6-32db41e2921a
# ╟─f8173203-a448-458e-870b-52302a822fe5
# ╠═328afa94-d6bb-4197-9fda-ff5b61efc62f
# ╠═6f09f404-3d5b-480d-be59-50c0edc88be3
# ╟─abce2d41-7074-4df0-834f-aa96dee6f34f
# ╠═f68e451c-9a23-416c-abcd-11b635186808
# ╠═25000b12-98c8-4aa0-a270-69aa10ec4f05
# ╠═8eea3830-51f8-4d55-a782-a1690db94b4d
# ╟─e747ecdd-bb4e-4991-9086-efcafa1242f4
# ╟─3648337e-4f84-4f4f-8cc7-1416877476d4
# ╟─88a16479-d006-4d84-ac09-60c3b28d9f8f
# ╟─10b0b1d9-2d8d-4d16-98ff-695e0a3dec6f
# ╟─b6a1c6cc-26bc-4a95-9236-3e4a5d7bfcd9
# ╟─1a1c4f5f-5a48-45f6-830b-bc66f4229c7a
# ╟─c3beb59b-3089-4a71-9267-dce52f17a3c0
# ╟─6d90f99d-f37a-479e-9353-4d43cd812fa8
# ╟─4bafac2a-a127-4f7c-bce7-af6513c5ab8c
# ╟─829925cd-4cfa-436c-b733-0d212bf3f481
# ╟─50ea4196-b225-4d66-87db-08199c47eae0
# ╟─2cc0008b-591a-4a4e-b502-b7a9d217e2e2
# ╟─81245ae8-b79b-4adb-8335-85e680d71862
# ╟─f6cf25d2-9c46-4961-89de-6331c4952b3f
# ╟─0fb74ac1-7014-41d5-b8dc-825768bdaf16
# ╟─b9fb0441-06f8-4da4-a1c5-7dd61155a0d7
# ╟─cd200ce8-ffb8-49fd-baeb-564719d2efd5
# ╟─89bfc20f-9c16-4f43-b4a6-e972e159b0f3
# ╟─d0886291-a616-4374-9c3a-4b6822a1bbf8
# ╟─33e527ab-4a9b-4251-b0cc-0722f74f2209
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
