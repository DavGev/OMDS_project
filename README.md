# Final Project
### Optimization Methods for Data Science, 2022-2023
*Sapienza Unversity of Rome, MSc Data Science*

*Professor: Veronica Piccialli*
____

In this project, we implement training methods for classification problems.

## Part 1. Multy-Layer Preceptron (MLP)

We wrote a program implementing an MLP network trained by minimizing the regularized binary cross-entropy error function
$$E(ω;π) = - \frac{1}{P} ∑_{i=1}^P {[y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i)]} + ρ \|ω\|^2$$
where the hyper parameter  $ρ = 10^{−4}$ stays fixed, $y_i ∈ \{0,1\}$ is the target value, and $p_i ∈ [0,1]$ is the Softmax probability for the $i$-th class.

The activation function $g(t, σ) := tanh(σt)$ is the hyperbolic tangent, with the spred of $σ$.

### Hyperparameters
- the number of hidden layers $H$ (max. 4) (only for question 1)
- the number of neurons $N$ of the hidden layers
- the spread $σ > 0$ in the activation function $g$ ($g$ is available in Python with $σ = 1$: `numpy.tanh`)

### Tasks
- **Question 1. (grade up to 20)** Use an optimization algorithm from `scipy.optimize` that uses the gradient to determine the parameters $v_j ,w_{ji}, b_j$ which minimize the error.
- **Question 2. (grade up to 10)** Develop an RBF neural network trained by implementing the decomposition method studied in class.

| Ex          | H | N | $σ$ | $ρ$   | Optimization | Message | Init train error | Final train error | Final  test error | f\grad evaluations | Time |
| -           | - | - | -   | -     | -            | -       | -                | -                 | -                 | -                  | -    |
| Q1 Full MLP | 2 | 8 | 1   | 0.001 | trust-constr |
| Q2 RBF      |


\* optimization: with parameters (optimality accuracy, max number of iterations etc)

\* message: in output (successful optimization or others, number of iterations, number of function/gradient evaluations, starting/final value of the objective function, starting/final accuracy etc)


## Dataset

We build a classifier that distinguishes between scan images of capital letters of the English alphabet. It is a good database for trying learning techniques on real-world data while spending minimal effort on preprocessing and formatting. The database is made of a large number of black-and-white rectangular pixels referring to the 26 capital letters in the English alphabet. The character images were based on 20 different fonts, and each letter within these 20 fonts was randomly distorted to produce a file of 20,000 unique stimuli. Each stimulus was converted into 16 primitive numerical attributes (statistical moments and edge counts), which were then scaled to fit into a range of integer values from 0 through 15. The columns of the dataset are referred to the following attributes:

* Y capital letter. This is the column of the output y.
* A1 horizontal position of box (integer)
* A2 vertical position of box (integer)
* A3 width of box (integer)
* A4 height of box (integer)
* A5 total # on pixels (integer)
* A6 mean x of on pixels in box (integer)
* A7 mean y of on pixels in box (integer)
* A8 mean x variance (integer)
* A9 mean y variance (integer)
* A10 mean x y correlation (integer)
* A11 mean of x * x * y (integer)
* A12 mean of x * y * y (integer)
* A13 mean edge count left to right (integer)
* A14 correlation of x-ege with y (integer)
* A15 mean edge count bottom to top (integer)
* A16 correlation of y-ege with x (integer)

The two/three classification tasks have the objective of discriminating between two/three characters in the Y column of the training set. The characters
used for this project are the first letters of our surnames, which are *A* and *G*. For the additional three-class classification problem we used added *R*

There is no test set, so we obtain it by randomly splitting the target set into a training set and a test set with a percentage of 80 %, 20%. A fixed number as a seed is used for the reproducability and a k-fold cross-validation is used to set the hyperparameters.
