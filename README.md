# math4ai-notes

This repository provides supplementary materials for the course **Mathematical Foundations for AI**.

It includes lecture notes, companion materials, and practice code based on the textbook **Mathematical Foundations for AI**.

The official PDF versions of *Mathematical Foundations for AI* are available from the Releases page.

- [Version 1.0.0, 2026-04-30](https://github.com/DrJaewookLee/math4ai-notes/releases/tag/v1.0.0)
- First Official Release (Amazon Paperback English Edition)
- ISBN: 9798195035235

---

## 📌 Contents

### Part I: Matrix Computations

#### Chapter 1: Linear Equations and Matrix Factorization
- Methods for solving systems of linear equations and the matrix representation of Gaussian elimination
- Properties of positive definite matrices and their role in the foundations of numerical optimization
- Cholesky factorization: an efficient decomposition method for symmetric positive definite matrices

#### Chapter 2: Vector Spaces, Geometry, and Norms
- Axiomatic definition of vector spaces and the structure of subspaces
- Inner products and norms: mathematical tools for measuring size, distance, and similarity in data
- Proof and geometric interpretation of the Cauchy-Schwarz inequality
- The Fundamental Theorem of Linear Algebra and the four fundamental subspaces

#### Chapter 3: Least Squares and Orthogonal Decomposition
- Geometric interpretation of linear transformations and orthogonal projections
- Least squares: deriving the optimal approximate solution that minimizes the sum of squared errors
- QR factorization: orthogonal decomposition of matrices through Gram-Schmidt orthogonalization and Householder transformations

#### Chapter 4: Fourier Analysis and Representations
- Fourier series: decomposing periodic functions into sums of trigonometric functions
- Discrete Fourier Transform (DFT) and the Fast Fourier Transform (FFT) algorithm for computational efficiency
- Characteristic functions in probability theory and applications of the inverse Fourier transform

---

### Part II: Spectral Methods and Differential Equations

#### Chapter 5: Singular Value Decomposition (SVD)
- Core principles of eigenvalues, eigenvectors, and the Spectral Theorem
- Geometric interpretation of SVD and the mathematical foundation of data compression through matrix structure analysis
- Generalized solutions using the Moore-Penrose pseudoinverse

#### Chapter 6: Spectral Methods for Dimension Reduction
- Principal Component Analysis (PCA): low-dimensional projection and visualization by maximizing data variance
- Multidimensional Scaling (MDS) and manifold learning methods that preserve geometric structure in high-dimensional data
- Graph-based dimension reduction methods and feature extraction strategies for nonlinear data

#### Chapter 7: Differential Equations and Dynamical Systems
- Solutions of linear differential equations and system analysis using the matrix exponential
- Phase portraits and dynamic stability analysis of systems
- Modeling and analysis of nonlinear systems through logistic growth and predator-prey models

---

### Part III: Optimization

#### Chapter 8: Matrix Calculus and Convex Analysis
- Topology of Euclidean spaces and continuity and differentiability of multivariable functions
- Gradient and Hessian matrix: analysis of local curvature and rates of change
- Matrix extensions of the chain rule, Taylor’s theorem, and convexity theory

#### Chapter 9: Optimization Algorithms for Machine Learning
- Optimality conditions for unconstrained optimization and line search methods
- Quasi-Newton methods: algorithms that approximate second-order derivative information to improve convergence speed
- Convergence analysis of momentum and adaptive learning-rate methods

#### Chapter 10: Constrained Optimization and Duality
- KKT conditions under equality and inequality constraints
- Lagrangian duality: the relationship between primal and dual problems and saddle-point theory

---

### Part IV: Statistical Learning Theory

#### Chapter 11: Probability, Information, and Estimation
- Quantifying information through entropy and the maximum entropy principle
- Properties of multivariate normal distributions and analysis of estimator bias and variance
- Statistical mechanisms of maximum likelihood estimation (MLE) and Bayesian estimation

#### Chapter 12: Linear Models and Generalization
- PAC learning theory and VC dimension for evaluating model complexity and generalization performance
- Advanced analysis of linear classification and regression models, including ridge regression and logistic regression

#### Chapter 13: Kernels and Support Vector Machine
- Mathematical definition of reproducing kernel Hilbert spaces (RKHS) and Mercer kernels
- Support Vector Machine (SVM): margin maximization and nonlinear data separation through the kernel trick

---

### Part V: Modern AI and Dynamics

#### Chapter 14: How Neural Networks Learn
- Matrix-calculus derivation of the backpropagation algorithm
- Training recurrent neural networks (RNNs), backpropagation through time (BPTT), and methods for addressing the vanishing gradient problem, including LSTM and GRU

#### Chapter 15: Transformers and Attention Mechanisms
- Development from Seq2Seq models to self-attention and an in-depth analysis of the Transformer architecture
- Mathematical roles of multi-head attention and positional encoding
- Extensions to modern model-efficiency methods, including LoRA, and to state space models (SSMs)

#### Chapter 16: Optimal Transport and Distributional Geometry
- Fenchel duality, convex conjugacy, and the Wasserstein distance
- Geometric optimization of generative models, including VAE and WAE, through optimal transport theory

#### Chapter 17: Geometric and Stochastic Dynamics
- Lyapunov stability analysis and gradient flows
- Neural ODEs: continuous-time neural network modeling using differential equations
- Stochastic differential equations (SDEs) and the mathematical principles of diffusion models, a core technique in modern generative AI

---

## 🛠️ How to Use

Clone this repository:

```bash
git clone https://github.com/DrJaewookLee/math4ai-notes.git
cd math4ai-notes  

#### Chapter 13: Kernels and Support Vector Machine
- Mathematical definition of reproducing kernel Hilbert spaces (RKHS) and Mercer kernels
- Support Vector Machine (SVM): margin maximization and nonlinear data separation through the kernel trick
