# SupervisedLearning

Supervised learning is the machine learning task of inferring a function from labeled training data. A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples.

Credits for the underlying algorithms of the functions that make predictions go to [Andrew Ng](http://cs.stanford.edu/people/ang/) at Stanford University.

## Example

One example is the prediction of house prices (output value) along two dimensions (features): the size of house in square meters and the number of bedrooms.

The training data could look something like this:

| Size (m2)     | # bedrooms    | Price  |
| ------------- |:-------------:| ------:|
| 2104          | 3             | 399900 |
| 1600          | 3             | 329900 |
| 3000          | 4             | 539900 |
| 1940          | 4             | 239999 |

Using linear regression, we can now predict the price of a house with 2200 square meters and 3 bedrooms.

## Installation

Add this line to your application's Gemfile:

    gem 'supervised_learning'

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install supervised_learning

## Usage

### 1. Create a matrix of the training data

The **output value** (the type of value you want to predict) needs to be in the **last column**. The matrix has to have a) at least two columns (one feature and one output) and b) at least one row. The more data you feed it, the more accurate the prediction.

Consult the [Ruby API](http://www.ruby-doc.org/stdlib-2.1.2/libdoc/matrix/rdoc/Matrix.html) for information on how to build matrices. You can use both arrays of rows or columns to create a matrix.

```ruby
require 'matrix'

training_set = Matrix[ [2104, 3, 399900], [1600, 3, 329900], [3000, 4, 539900], [1940, 4, 239999] ]
```

### 2. Instantiate an object with the training data

```ruby
require 'supervised_learning' # if not automatically loaded

program = SupervisedLearning::LinearRegression.new(training_set)
```

### 3. Create a prediction in form of a matrix

This matrix has one row and the **columns follow the order of the training set**. It has one column less than the training set since the output value (the last column of the training set) is the value we want to predict.

```ruby
# Predict the price of a house of 2200 square meters with 3 bedrooms
prediction_set = Matrix[ [2200, 3] ]

program.predict(prediction_set)
=> 454115.66
```

### 4. Options

The lower bound running time of the `#predict` method is cubic / Ω(n3). Hence, on very large datasets (more than 1,000 columns / features), the `#predict` method can be slow. If this happens, use the `#predict_advanced` method, which uses gradient descent (approximation) instead of normalization (direct calculation). Gradient descent can be a bit less accurate and also requires some optional parameters.

The `#predict_advanced` method takes 4 parameters in the following order:

1. The prediction_set, similar to the `#prediction` method discussed above
2. The learning rate (how big of a "jump" each iteration makes). The smaller the learning rate, the less errors are to be expected. Unfortunately, a smaller learning rate requires more iterations (see below). Start at 0.001 and increase the parameter to 0.01, 0.1, 1, etc if the calculation is too slow.
3. The number of iterations. The more iterations, the longer accurate the result will but the slower the algorithm will run.
4. Whether you want print out values to see at which point convergence happens.

```ruby
# Predict the price of a house of 2200 square meters with 3 bedrooms
prediction_set = Matrix[ [2200, 3] ]

program.predict_advanced(prediction_set, 0.01, 600, false)
=> 443693.16
```

## Contributing

1. Fork it ( https://github.com/[my-github-username]/supervised_learning/fork )
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request
