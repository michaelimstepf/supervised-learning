# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'supervised_learning/version'

Gem::Specification.new do |spec|
  spec.name          = "supervised_learning"
  spec.version       = SupervisedLearning::VERSION
  spec.authors       = ["Michael Imstepf"]
  spec.email         = ["michael.imstepf@gmail.com"]
  spec.summary       = %q{A module to make predictions based on a set of training data.}
  spec.description   = %q{Supervised learning is the machine learning task of inferring a function from labeled training data. A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples.}
  spec.homepage      = "https://github.com/michaelimstepf/supervised-learning"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0")
  spec.executables   = spec.files.grep(%r{^bin/}) { |f| File.basename(f) }
  spec.test_files    = spec.files.grep(%r{^(test|spec|features)/})
  spec.require_paths = ["lib"]

  spec.add_development_dependency "bundler", "~> 1.6"
  spec.add_development_dependency "rake"
  spec.add_development_dependency "rspec"
  spec.add_runtime_dependency "descriptive_statistics"
  spec.add_runtime_dependency "matrix_extensions"
  spec.add_development_dependency "pry"
end
