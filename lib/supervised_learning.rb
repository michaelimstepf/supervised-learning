require 'supervised_learning/version'
require 'matrix'
require 'descriptive_statistics'
require 'matrix_extensions'

module SupervisedLearning
    
  # This class uses linear regression to make predictions based on a training set.
  # For datasets of less than 1000 columns, use #predict since this will give the most accurate prediction.
  # For larger datasets where the #predict method is too slow, use #predict_advanced.
  # The algorithms in #predict and #predict_advanced were provided by Andrew Ng (Stanford University).
  # @author Michael Imstepf
  class LinearRegression
    # Initializes a LinearRegression object with a training set
    # @param training_set [Matrix] training_set, each feature/dimension has one column and the last column is the output column (type of value #predict will return)
    # @raise [ArgumentError] if training_set is not a Matrix or does not have at least two columns and one row
    def initialize(training_set)      
      @training_set = training_set
      raise ArgumentError, 'input is not a Matrix' unless @training_set.is_a? Matrix
      raise ArgumentError, 'Matrix must have at least 2 columns and 1 row' unless @training_set.column_size > 1

      @number_of_features = @training_set.column_size - 1
      @number_of_training_examples = @training_set.row_size      

      @feature_set = @training_set.clone
      @feature_set.hpop # remove output set

      @output_set = @training_set.column_vectors.last      
    end

    # Makes prediction using normalization.
    # This algorithm is the most accurate one but with large
    # sets (more than 1000 columns) it might take too long to calculate.
    # @param prediction [Matrix] prediction    
    def predict(prediction)
      # add ones to feature set
      feature_set = Matrix.hconcat(Matrix.one(@number_of_training_examples, 1), @feature_set)

      validate_prediction_input(prediction)
            
      transposed_feature_set = feature_set.transpose # only transpose once for efficiency                  
      theta = (transposed_feature_set * feature_set).inverse * transposed_feature_set * @output_set

      # add column of ones to prediction
      prediction = Matrix.hconcat(Matrix.one(prediction.row_size, 1), prediction)
      
      result_vectorized = prediction * theta
      result = result_vectorized.to_a.first.to_f
    end

    # Makes prediction using gradient descent.
    # This algorithm is requires less computing power
    # than #predict but is less accurate since it uses approximation.
    # @param prediction [Matrix] prediction
    def predict_advanced(prediction, learning_rate = 0.01, iterations = 1000, debug = false) 
      validate_prediction_input(prediction)

      feature_set = normalize_feature_set(@feature_set)
      # add ones to feature set after normalization      
      feature_set = Matrix.hconcat(Matrix.one(@number_of_training_examples, 1), feature_set)

      # prepare theta column vector with zeros
      theta = Matrix.zero(@number_of_features+1, 1)

      iterations.times do        
        theta = theta - (learning_rate * (1.0/@number_of_training_examples) * (feature_set * theta - @output_set).transpose * feature_set).transpose
        if debug
          puts "Theta: #{theta}"
          puts "Cost: #{calculate_cost(feature_set, theta)}"
        end
      end

      # normalize prediction
      prediction = normalize_prediction(prediction)

      # add column of ones to prediction
      prediction = Matrix.hconcat(Matrix.one(prediction.row_size, 1), prediction)

      result_vectorized = prediction * theta
      result = result_vectorized[0,0]
    end   

    private

    # Validates prediction input.
    # @param prediction [Matrix] prediction
    # @raise [ArgumentError] if prediction is not a Matrix
    # @raise [ArgumentError] if prediction does not have the correct number of columns (@training set minus @output_set)
    # @raise [ArgumentError] if prediction has more than one row
    def validate_prediction_input(prediction)
      raise ArgumentError, 'input is not a Matrix' unless prediction.is_a? Matrix
      raise ArgumentError, 'input has more than one row' if prediction.row_size > 1
      raise ArgumentError, 'input has wrong number of columns' if prediction.column_size != @number_of_features 
    end

    # Normalizes feature set for quicker and more reliable calculation.
    # @param feature_set [Matrix] feature set
    # @return [Matrix] normalized feature set
    def normalize_feature_set(feature_set)
      # get mean for each column
      mean = []
      feature_set.column_vectors.each do |feature_set_column|
        mean << feature_set_column.mean
      end          
      # convert Array into Matrix of same dimension as feature_set for substraction
      # save for later usage      
      @mean = Matrix[mean].vcopy(@number_of_training_examples - 1)

      # substract mean from feature set
      feature_set = feature_set - @mean

      # get standard deviation for each column
      standard_deviation = []
      feature_set.column_vectors.each do |feature_set_column|
        standard_deviation << feature_set_column.standard_deviation
      end
      # convert Array into Matrix of same dimension as feature_set for substraction
      # save for later usage         
      @standard_deviation = Matrix[standard_deviation].vcopy(@number_of_training_examples - 1)

      # divide feature set by standard deviation
      feature_set = feature_set.element_division @standard_deviation
    end

    # Normalizes prediction.
    # @param prediction [Matrix] prediction
    # @return [Matrix] normalized prediction
    def normalize_prediction(prediction)      
      # convert prediction into Matrix of same dimension as @mean for substraction
      prediction = prediction.vcopy(@number_of_training_examples - 1)

      # substract mean
      prediction = prediction - @mean

      # divide feature set by standard deviation
      prediction = prediction.element_division @standard_deviation 
    end

    # Calculates cost of current theta.
    # The closer to 0, the more accurate the prediction will be.
    # @param feature_set [Matrix] feature set
    # @param theta [Matrix] theta
    # @return [Float] cost    
    def calculate_cost(feature_set, theta)
      cost_vectorized = 1.0/(2 * @number_of_training_examples) * (feature_set * theta - @output_set).transpose * (feature_set * theta - @output_set)
      cost_vectorized[0,0]
    end    
  end

  # This class uses logistic regression to make discrete predictions (true or false) based on a training set.
  # The algorithms in #predict were provided by Andrew Ng (Stanford University).
  # @author Michael Imstepf
  class LogisticRegression
    # Initializes a LogisticRegression object with a training set
    # @param training_set [Matrix] training_set, each feature/dimension has one column and the last column is the output column (type of value #predict will return)
    # @raise [ArgumentError] if training_set is not a Matrix or does not have at least two columns and one row
    def initialize(training_set)
      @training_set = training_set
      raise ArgumentError, 'input is not a Matrix' unless @training_set.is_a? Matrix
      raise ArgumentError, 'Matrix must have at least 2 columns and 1 row' unless @training_set.column_size > 1
    end

    private

    def calculate_sigmoid(z)
      matrix_with_ones = Matrix.one(1, z.column_size)
      matrix_with_eulers_number = Matrix.build(1, z.column_size) {Math::E}
      z_negative = z * -1
      matrix_with_ones.element_division (matrix_with_ones + matrix_with_eulers_number.element_exponentiation(z_negative))
    end

    def calculate_cost

    end
  end

end
