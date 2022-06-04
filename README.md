<div dir=rtl align=right>

# تۆڕێکی دەماریی (`Neural Network`) ساکار

کۆدەکانی فایلی `neural-network.js` وەرگێڕانێکە لە زمانی `Python` ڕا لە
ئەم سەرچاوەیە:

<div dir=ltr align=left>

[How to Code a Neural Network with Backpropagation In Python (from scratch)](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)

</div>

## بەکارهێنان

### دروست کردنی تۆڕێکی نوێ

<div dir=ltr align=left>

```js
const number_of_inputs = 10
const network = new NeuralNetwork(number_of_inputs)
```

</div>

### زیاد کردنی تەبەقێکی نوێی دەمارەکان

<div dir=ltr align=left>

```js
const number_of_neurons = 20
const activation_function = 'relu'
network.add_layer(number_of_neurons, activation_function)
```

</div>

### زیاد کردنی تەبەقی کۆتایی

<div dir=ltr align=left>

```js
const number_of_outputs = 2
const activation_function = 'none'
network.add_layer(number_of_outputs, activation_function)
```

</div>

### فێرکاریی تۆڕەکە

<div dir=ltr align=left>

```js
const data = [
      /* Structure: [Inputs... , Expected output]
      [0.1, 0.2, 0.1, 1],
      [0.9, 0.8, 0.7, 0]
]
const learning_rate = 0.1
const number_of_epochs = 10
network.train(data, learning_rate, number_of_epochs)
```

</div>

### تاقی کردنەوەی تۆڕەکە

<div dir=ltr align=left>

```js
const predicted = network.predict([0.1, 0.2, 0.15])
console.log(predicted)
```

</div>

## نموونەکان

- [ناسینەوەی ژمارەکان](https://allekok.github.io/neural-network/examples/numbers.html)

</div>
