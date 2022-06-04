class NeuralNetwork {
	network = []
	activation_functions = {
		sigmoid: [this.sigmoid, this.sigmoid_deriv],
		relu: [this.relu, this.relu_deriv],
		none: [this.none, this.none_deriv],
	}

	constructor(n_inputs, print=console) {
		this.n_inputs = n_inputs
		this.n_outputs = n_inputs
		this.print = print
	}

	/* Interface */
	add_layer(n_neurons, activation) {
		const layer = this.new_layer(
			n_neurons, this.activation_functions[activation])
		this.network.push(layer)
		this.n_outputs = n_neurons
		return this
	}
	train(data, l_rate, n_epoch) {
		for(let epoch = 0; epoch < n_epoch; epoch++) {
			let sum_error = 0
			for(const row of data) {
				const outputs = this.forward_propagate(row)
				const expected = new Array(
					this.n_outputs).fill(0)

				expected[row[row.length - 1]] = 1

				sum_error += expected.map((x, i) => {
					return Math.pow(x - outputs[i], 2)
				}).reduce((a, b) => a + b)
				this.backward_propagate_error(expected)
				this.update_weights(row, l_rate)
			}
			this.print.log(`>epoch=${epoch}, ` +
				       `lrate=${l_rate}, ` +
				       `error=${sum_error}`)
		}
		return this
	}
	predict(row) {
		const outputs = this.forward_propagate(row)
		return outputs.indexOf(Math.max(...outputs))
	}

	/* Inner functions */
	new_layer(n_neurons, activation) {
		return {
			neurons:
			this.new_array(n_neurons,
				       _ => this.new_neuron(this.n_outputs)),
			activation: activation[0],
			activation_deriv: activation[1],
		}
	}
	new_neuron(n_inputs) {
		return {
			weights: this.new_array(n_inputs + 1,
						_ => Math.random())
		}
	}
	new_array(len, init) {
		return new Array(len).fill(0).map(init)
	}
	output(weights, inputs) {
		const outputs = [weights[weights.length - 1]] /* Bias */
		for(let i = 0; i < weights.length - 1; i++)
			outputs.push(weights[i] * inputs[i])
		return outputs
	}
	forward_propagate(row) {
		let inputs = row
		for(const layer of this.network) {
			const new_inputs = []
			for(const neuron of layer.neurons) {
				const outputs = this.output(neuron.weights,
							    inputs)
				neuron.output = layer.activation(outputs)
				new_inputs.push(neuron.output)
			}
			inputs = new_inputs
		}
		return inputs
	}
	backward_propagate_error(expected) {
		for(let i = this.network.length - 1; i >= 0; i--) {
			const layer = this.network[i]
			const errors = []
			if(i == this.network.length - 1) {
				/* Output layer */
				for(const j in layer.neurons) {
					/* Each neuron in output layer */
					const neuron = layer.neurons[j]
					const error = (neuron.output -
						       expected[j])
					errors.push(error)
				}
			}
			else {
				/* Hidden layers */
				for(const j in layer.neurons) {
					/* Each neuron in current layer */
					let error = 0
					const next_layer = this.network[i + 1]
					for(const neuron of next_layer.neurons)
					{
						/* Each neuron in next layer */
						error += (neuron.weights[j] *
							  neuron.delta)
					}
					errors.push(error)
				}
			}
			for(const j in layer.neurons) {
				/* Each neuron in current layer */
				const neuron = layer.neurons[j]
				neuron.delta = (errors[j] *
						layer.activation_deriv(
							neuron.output))
			}
		}
	}
	update_weights(row, l_rate) {
		for(const i in this.network) {
			let inputs = row.slice(0, row.length - 1)
			if(i != 0) {
				inputs = this.network[i - 1].neurons.
					map(neuron => neuron.output)
			}
			const layer = this.network[i]
			for(const neuron of layer.neurons) {
				for(const j in inputs) {
					neuron.weights[j] -= (l_rate *
							      neuron.delta *
							      inputs[j])
				}
				neuron.weights[neuron.weights.length - 1] -= (
					l_rate * neuron.delta)
			}
		}
	}

	/* Activation functions */
	sigmoid(input) {
		return 1 / (1 + Math.exp(-input.reduce((a, b) => a + b)))
	}
	sigmoid_deriv(output) {
		return output * (1 - output)
	}
	relu(input) {
		return Math.max(0, ...input)
	}
	relu_deriv(output) {
		return output < 0 ? 0 : 1
	}
	none(input) {
		return input.reduce((a, b) => a + b)
	}
	none_deriv(output) {
		return 1
	}
}
