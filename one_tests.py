import random
import numpy
from ComputationalGraphPrimer import *
from dl3 import *
import time

seed = 0           
random.seed(seed)
numpy.random.seed(seed)

for learning_rate in [0.1, 0.01, 0.001]:
      cgp = ComputationalGraphPrimer(
                  one_neuron_model = True,
                  expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
                  output_vars = ['xw'],
                  dataset_size = 5000,
                  learning_rate = learning_rate,
                  training_iterations = 40000,
                  batch_size = 8,
                  display_loss_how_often = 100,
                  debug = False,
            )


      cgp.parse_expressions()
      training_data = cgp.gen_training_data()
      sgd = cgp.run_training_loop_one_neuron_model(training_data)



      cgp2 = myCGP_sgdp(
                  one_neuron_model = True,
                  expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
                  output_vars = ['xw'],
                  dataset_size = 5000,
                  learning_rate = learning_rate,
                  training_iterations = 40000,
                  batch_size = 8,
                  display_loss_how_often = 100,
                  debug = False,
            )

      cgp2.parse_expressions()    
      sgd_plus = cgp2.run_training_loop_one_neuron_model(training_data, momentum_coeff=0.95)



      cgp3 = myCGP_adam(
                  one_neuron_model = True,
                  expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
                  output_vars = ['xw'],
                  dataset_size = 5000,
                  learning_rate = learning_rate,
                  training_iterations = 40000,
                  batch_size = 8,
                  display_loss_how_often = 100,
                  debug = False,
            )

      cgp3.parse_expressions()
      adam = cgp3.run_training_loop_one_neuron_model(training_data, b1=0.9, b2=0.99, eps=1e-8)


      plt.figure()
      plt.plot(sgd, 'b', label="SGD")
      plt.plot(sgd_plus, 'r', label="SGD+")
      plt.plot(adam, 'g', label="adam")
      plt.legend()
      plt.xlabel("iteration")
      plt.ylabel("Loss")
      plt.title(f"Performance analysis for Learning rate = {learning_rate}")
      plt.savefig("one/single_out_" + str(learning_rate)[2:] + ".jpg")

      print(f"\n\nLR={learning_rate}\nMin error for sgd: ", min(sgd))
      print("Min error for sgd+: ", min(sgd_plus))
      print("Min error for adam: ", min(adam))

      print("\nFinal error for sgd: ", sgd[-1])
      print("Final error for sgd+: ", sgd_plus[-1])
      print("Final error for adam: ", adam[-1])


plt.figure()
for b1 in [0.85, 0.9, 0.99]:
        for b2 in [0.89, 0.95, 0.99]:
                  start = time.time()
                  cgp3 = myCGP_adam(
                        one_neuron_model = True,
                        expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
                        output_vars = ['xw'],
                        dataset_size = 5000,
                        learning_rate = 0.001,
                        training_iterations = 40000,
                        batch_size = 8,
                        display_loss_how_often = 100,
                        debug = False,
                        )

                  cgp3.parse_expressions()
                  training_data = cgp3.gen_training_data()
                  adam = cgp3.run_training_loop_one_neuron_model( training_data, b1=b1, b2=b2, eps=1e-8 )
                  end = time.time()

                  print(f"\n\nFor b1 = {b1} and b2 = {b2}\n")
                  print("Minimum Loss = ", min(adam))
                  print("Final Loss = ", adam[-1])
                  print("Time Taken: ", end-start)

                  plt.plot(adam, label=f"b1:{b1}, b2:{b2}")

plt.legend()
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.title("Beta value comparisons for Single Neuron")
plt.savefig(f"one/beta_test_single.jpg")