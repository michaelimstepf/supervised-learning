require 'supervised_learning'
require 'pry'

RSpec.configure do |config|  
  # Run specs in random order to surface order dependencies. If you find an
  # order dependency and want to debug it, you can fix the order by providing
  # the seed, which is printed after each run.
  #     --seed 1234
  config.order = 'random'
  
  # when a focus tag is present in RSpec, only run tests with focus tag: http://railscasts.com/episodes/285-spork
  config.filter_run :focus
  config.run_all_when_everything_filtered = true  
end