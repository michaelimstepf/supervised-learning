require 'spec_helper'

describe SupervisedLearning::LinearRegression do
  training_set_one_feature = Matrix[ [2104,399900], [1600,329900], [2400,369000], [1416,232000], [3000,539900], [1985,299900], [1534,314900], [1427,198999], [1380,212000], [1494,242500], [1940,239999], [2000,347000], [1890,329999], [4478,699900], [1268,259900], [2300,449900], [1320,299900], [1236,199900], [2609,499998], [3031,599000], [1767,252900], [1888,255000], [1604,242900], [1962,259900], [3890,573900], [1100,249900], [1458,464500], [2526,469000], [2200,475000], [2637,299900], [1839,349900], [1000,169900], [2040,314900], [3137,579900], [1811,285900], [1437,249900], [1239,229900], [2132,345000], [4215,549000], [2162,287000], [1664,368500], [2238,329900], [2567,314000], [1200,299000], [852,179900], [1852,299900], [1203,239500] ]
  program_one_feature = SupervisedLearning::LinearRegression.new(training_set_one_feature)    

  training_set_two_features = Matrix[ [2104,3,399900], [1600,3,329900], [2400,3,369000], [1416,2,232000], [3000,4,539900], [1985,4,299900], [1534,3,314900], [1427,3,198999], [1380,3,212000], [1494,3,242500], [1940,4,239999], [2000,3,347000], [1890,3,329999], [4478,5,699900], [1268,3,259900], [2300,4,449900], [1320,2,299900], [1236,3,199900], [2609,4,499998], [3031,4,599000], [1767,3,252900], [1888,2,255000], [1604,3,242900], [1962,4,259900], [3890,3,573900], [1100,3,249900], [1458,3,464500], [2526,3,469000], [2200,3,475000], [2637,3,299900], [1839,2,349900], [1000,1,169900], [2040,4,314900], [3137,3,579900], [1811,4,285900], [1437,3,249900], [1239,3,229900], [2132,4,345000], [4215,4,549000], [2162,4,287000], [1664,2,368500], [2238,3,329900], [2567,4,314000], [1200,3,299000], [852,2,179900], [1852,4,299900], [1203,3,239500] ]
  program_two_features = SupervisedLearning::LinearRegression.new(training_set_two_features)

  describe '#initialize' do    
    context 'when training set is not a matrix' do
      it 'raises an exception' do
        expect {SupervisedLearning::LinearRegression.new([1, 2])}.to raise_exception(ArgumentError)
      end
    end

    context 'when training set is an empty Matrix' do
      it 'raises an exception' do
        expect {SupervisedLearning::LinearRegression.new(Matrix[])}.to raise_exception(ArgumentError)
      end
    end

    context 'when training only has one column' do
      it 'raises an exception' do
        expect {SupervisedLearning::LinearRegression.new(Matrix[[1]])}.to raise_exception(ArgumentError)
      end
    end
  end

  describe '#predict' do
    context 'when prediction set is not a matrix' do
      it 'raises an exception' do
        expect {program_one_feature.predict([1, 2])}.to raise_exception(ArgumentError)
      end
    end

    context 'when prediction has more than one row' do
      it 'raises an exception' do
        expect {program_two_features.predict(Matrix[ [1, 2], [3, 4] ])}.to raise_exception(ArgumentError)
      end
    end

    context 'when prediction has wrong amount of columns' do
      context 'when training set has one feature' do
        it 'raises an exception' do
          expect {program_one_feature.predict(Matrix[ [1, 2] ])}.to raise_exception(ArgumentError)
          expect {program_one_feature.predict(Matrix[ ])}.to raise_exception(ArgumentError)           
        end        
      end

      context 'when training set has two features' do
        it 'raises an exception' do
          expect {program_two_features.predict(Matrix[ [1, 2, 3] ])}.to raise_exception(ArgumentError)
          expect {program_two_features.predict(Matrix[ [1] ])}.to raise_exception(ArgumentError)                 
        end        
      end      
    end

    context 'when prediction has correct amount of columns' do
      context 'when training set has one feature' do
        it 'returns correct prediction' do
          expect(program_one_feature.predict(Matrix[ [1650] ]).to_i).to eq 293237
        end        
      end

      context 'when training set has two features' do
        it 'returns correct prediction' do
          expect(program_two_features.predict(Matrix[ [1650, 3] ]).to_i).to eq 293081
        end        
      end            
    end              
  end

  describe '#predict_advanced' do
    context 'when prediction set is not a matrix' do
      it 'raises an exception' do
        expect {program_one_feature.predict_advanced([1, 2])}.to raise_exception(ArgumentError)
      end
    end

    context 'when prediction has more than one row' do
      it 'raises an exception' do
        expect {program_two_features.predict_advanced(Matrix[ [1, 2], [3, 4] ])}.to raise_exception(ArgumentError)
      end
    end

    context 'when prediction has wrong amount of columns' do
      context 'when training set has one feature' do
        it 'raises an exception' do
          expect {program_one_feature.predict_advanced(Matrix[ [1, 2] ])}.to raise_exception(ArgumentError)
          expect {program_one_feature.predict_advanced(Matrix[ ])}.to raise_exception(ArgumentError)           
        end        
      end

      context 'when training set has two features' do
        it 'raises an exception' do
          expect {program_two_features.predict_advanced(Matrix[ [1, 2, 3] ])}.to raise_exception(ArgumentError)
          expect {program_two_features.predict_advanced(Matrix[ [1] ])}.to raise_exception(ArgumentError)                 
        end        
      end      
    end

    context 'when prediction has correct amount of columns' do
      context 'when training set has one feature' do
        it 'returns correct prediction' do
          expect(program_one_feature.predict_advanced(Matrix[ [1650] ], 0.01, 600, false).to_i).to be_within(200).of(293237)
        end        
      end

      context 'when training set has two features', :focus do
        it 'returns correct prediction' do
          expect(program_two_features.predict_advanced(Matrix[ [1650, 3] ], 0.01, 600, true).to_i).to be_within(200).of(293237)
        end        
      end            
    end              
  end      
end