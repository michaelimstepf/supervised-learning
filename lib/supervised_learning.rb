require 'supervised_learning/version'
require 'matrix'
require 'descriptive_statistics'

module SupervisedLearning
    
  # This class uses linear regression to make predictions based on a training set.
  # For datasets of less than 1000 columns, use #predict since this will give the most accurate prediction.
  # For larger datasets where the #predict method is too slow, use #predict_advanced.
  # The algorithms in #predict and #predict_advanced were provided by Andrew Ng (Stanford University).
  # @author Michael Imstepf
  class LinearRegression
    # Initializes a LinearRegression object with a training set
    # @param training_set [Matrix] training_set, each feature/dimension has one column and the last column is the output column (type of value #predict will return)
    # @raise [ArgumentError] if training_set is not a Matrix and has at least two columns and one row
    def initialize(training_set)      
      @training_set = training_set
      raise ArgumentError, 'input is not a Matrix' unless @training_set.is_a? Matrix
      raise ArgumentError, 'Matrix must have at least 2 columns and 1 row' unless @training_set.column_size > 1

      @number_of_training_set_columns = @training_set.column_size
      @number_of_features = @number_of_training_set_columns - 1
      @number_of_training_examples = @training_set.row_size      

      @output_set = @training_set.column_vectors.last      
    end

    # Makes prediction using normalization.
    # This algorithm is the most accurate one but with large
    # sets (more than 1000 columns) it might take too long to calculate.
    # @param prediction [Matrix] prediction    
    def predict(prediction)
      feature_set = get_feature_set(@training_set, true)      

      validate_prediction_input(prediction)
            
      transposed_feature_set = feature_set.transpose # only transpose once for efficiency                  
      theta = (transposed_feature_set * feature_set).inverse * transposed_feature_set * @output_set

      # add column of ones to prediction
      prediction = get_feature_set(prediction, true)
      
      result_vectorized = prediction * theta
      result = result_vectorized.to_a.first.to_f
    end

    # Makes prediction using gradient descent.
    # This algorithm is requires less computing power
    # than #predict but is less accurate since it uses approximation.
    # @param prediction [Matrix] prediction
    def predict_advanced(prediction, learning_rate = 0.01, iterations = 1000, debug = false) 
      validate_prediction_input(prediction)

      feature_set = get_feature_set(@training_set, false) 
      feature_set = normalize_feature_set(feature_set)
      # add ones to feature set after normalization      
      feature_set = get_feature_set(feature_set, true)

      # prepare theta column vector with zeros
      theta = Array.new(@number_of_training_set_columns, 0)
      theta = Matrix.columns([theta])

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
      prediction = get_feature_set(prediction, true)

      result_vectorized = prediction * theta
      result = result_vectorized[0,0]
    end   

    private

    # Returns a feature set without output set (last column of training set)
    # and optionally adds a leading column of ones to a Matrix.
    # This column of ones is the first dimension of theta to easily calculate
    # the output of a function a*1 + b*theta_1 + c*theta_2 etc.    
    # Ruby's Matrix class has not built-in function for prepending,
    # hence some manual work is required.
    # @see http://stackoverflow.com/questions/9710628/how-do-i-add-columns-and-rows-to-a-matrix-in-ruby
    # @param matrix [Matrix] matrix
    # @param leading_ones [Boolean] whether to prepend a column of leading ones    
    # @return [Matrix] matrix
    def get_feature_set(matrix, leading_ones = false)
      # get array of columns
      existing_columns = matrix.column_vectors      

      columns = []
      columns << Array.new(existing_columns.first.size, 1) if leading_ones           
      # add remaining columns
      existing_columns.each_with_index do |column, index|
        # output column (last column of @training_set) needs to be skipped        
        # when called from #get_feature_set, matrix includes output column
        # when called from #prediction, matrix does not inlcude output column
        break if index + 1 > @number_of_features    
        columns << column.to_a
      end
      
      Matrix.columns(columns)      
    end

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
      # create Matrix with mean
      mean = []
      feature_set.column_vectors.each do |feature_set_column|
        # create Matrix of length of training examples for later substraction
        mean << Array.new(@number_of_training_examples, feature_set_column.mean)
      end          
      mean = Matrix.columns(mean)
      
      # save for later usage as Matrix and not as Vector
      @mean = Matrix[mean.row(0)]

      # substract mean from feature set
      feature_set = feature_set - mean

      # create Matrix with standard deviation
      standard_deviation = []
      feature_set.column_vectors.each do |feature_set_column|
        # create row vector with standard deviation
        standard_deviation << [feature_set_column.standard_deviation]
      end            
      # save for later usage
      @standard_deviation = Matrix.columns(standard_deviation)

      # Dividing these non-square matrices has to be done manually
      # (non square matrices have no inverse and can't be divided in Ruby)
      # iterate through each column    
      columns = []
      feature_set.column_vectors.each_with_index do |feature_set_column, index|
        # manually divide each row within column with standard deviation for that row
        columns << feature_set_column.to_a.collect { |value| value / @standard_deviation[0,index] }
      end
      # reconstruct training set
      feature_set = Matrix.columns(columns)      
      feature_set
    end

    # Normalizes prediction.
    # @param prediction [Matrix] prediction
    # @return [Matrix] normalized prediction
    def normalize_prediction(prediction)
      # substract mean
      prediction = prediction - @mean      

      # Dividing these non-square matrices has to be done manually
      # (non square matrices have no inverse and can't be divided in Ruby)
      # iterate through each column
      columns = []
      prediction.column_vectors.each_with_index do |prediction_column, index|
        # manually divide each row within column with standard deviation for that row
        columns << prediction_column / @standard_deviation[0,index]
      end      
      # reconstruct prediction
      prediction = Matrix.columns(columns) 
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
end
