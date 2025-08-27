<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
## 1A. 1장 부록

### 1A.1 보충 설명: 1.1.2절  

- 행렬곱의 여러가지 표현방법 MATLAB 구현 코드


```matlab
function C = MatMat(A,B,method)

 [m,p] = size(A); [p,n] = size(B);
 C = zeros(m,n);
 switch lower(method)

 %(1) C(i,j) = A(i,1)B(1,j)+...+A(i,p)B(p,j).
 case {'prod'}
 for j=1:n
   for i=1:m
       for k=1:p
          C(i,j) = C(i,j)+A(i,k)*B(k,j);
      end
   end
 end

 %(2) C(i,j) = row i of A times column j of B.
 case {'dot'}
 for j=1:n
   for i=1:m
      C(i,j) = A(i,:)*B(:,j);
   end
 end

 %(3) column $j$ of C = A times column j of B.
 case {'col'}
 for j=1:n
   C(:,j) = C(:,j) + A*B(:,j);
 end

 %(4) row i of C = row i of A times B.
 case {'row'}
 for i=1:m
   C(i,:) = C(i,:) + A(i,:)*B;
 end

 %(5) AB = A(:,1)B(1,:)+...+A(:,p)B(p,:)
 case {'outer'}
 for k=1:p
   C = C + A(:,k)*B(k,:);
 end

 %(6) AB = A*B
 otherwise % case {'direct'}
 C=A*B;
 end

% Compare different matrix multiplication methods.
 n     Prod      Dot     Col      Row      Outer    Direct
 ----------------------------------------------------------
 100    0.0166   0.0110  0.0009   0.0008   0.0027   0.0001
 200    0.1298   0.0495  0.0015   0.0013   0.0157   0.0003
 500    2.2069   0.8798  0.0130   0.0157   0.2858   0.0069
1000   22.5814   9.3867  0.1738   0.1371   3.1449   0.0466
2000  251.9683  81.7906  3.4044   3.3608  32.6324   0.2459
```


---




### 1A.2 보충 설명: 1.2.1절 




---



> **정리 1A.1. [LU 분해(LU decomposition) $A=LU$]** 

행 교환이 필요하지 않은 경우, 모든 $m \times m$ 행렬 $A$는 유일한  $A=LU$ , 즉 **LU 분해(LU decomposition)**를 갖는다. 여기서 $L$ 행렬은 하삼각행렬이며 대각성분은 1, 대각 아래 성분은 소거 과정에서 얻은 승수이다. $U$는 상삼각행렬로, 전진 소거(forward elimination) 이후와 후진 대입(back-substitution) 이전에 나타나며, 대각 성분들이 피벗(pivot)이다.


---


**(증명)** 예를 들어, $4\times 4$ 행렬을 생각하자. (자세한 증명은 [^Trefethen and Bau. (1997)] 또는 [^Golub et al. (1995)] 을 참조하라.)
$$
\underbrace{\left[\begin{array}{rrrr}\times & \times & \times & \times\\
\times & \times & \times & \times\\
\times & \times & \times & \times\\
\times & \times & \times & \times\end{array} \right]}_{A}
~~{\underrightarrow{M_1}}~~
\underbrace{\left[\begin{array}{rrrr}\times & \times & \times & \times\\
0 & \times & \times & \times\\
0 & \times & \times & \times\\
0 & \times & \times & \times\end{array} \right]}_{M_1 A}
~~{\underrightarrow{M_2}}~~
\underbrace{\left[\begin{array}{rrrr}\times & \times & \times & \times\\
  & \times & \times & \times\\
  & 0 & \times & \times\\
  & 0 & \times & \times\end{array} \right]}_{M_2 M_1 A} 
~~{\underrightarrow{M_3}}~~
\underbrace{\left[\begin{array}{rrrr}\times & \times & \times & \times\\
  & \times & \times & \times\\
  &   & \times & \times\\
  &   & 0 & \times\end{array} \right]}_{M_3 M_2 M_1
A}
$$

$m-1$ steps 이후의 행렬은 상삼각행렬 $U$가 된다.

$$
\underbrace{M_{m-1}\cdots M_2M_1}_{L^{-1}}A=U \label{eq:ge1}
$$
$k$ step의 시작에서, $\boldsymbol{x}_k =\tilde{A}_k(:,k)$라 하자.
이는 행렬의  $k$번째 열을 나타낸다.
$$
\tilde{A}_k=M_{k-1}\cdots M_2 M_1 A=\left[\begin{array}{cc}
 \tilde{A}_k(1:k-1,1:k-1) & \tilde{A}_k(1:k-1,k:m) \\ & \\
 \boldsymbol{0} & \tilde{A}_k(k:m,k:m)\end{array} \right]
$$

이 때,  $\tilde{A}_k(1:k-1,1:k-1)$ 는 상삼각행렬이다. 만약,
$$
M_k=I-\boldsymbol{l}_k \boldsymbol{e}_k^T  \qquad\mbox{where}\qquad
\boldsymbol{e}_k=
\left[\begin{array}{c} 0 \\ \vdots \\ 1 \\ 0 \\ \vdots\\
0\end{array} \right],\;\boldsymbol{l}_k=
\left[\begin{array}{c} 0 \\ \vdots \\ 0 \\ l_{k+1,k} \\ \vdots\\
l_{m,k}\end{array} \right] ,\; l_{jk}=\frac{x_{jk}}{x_{kk}}, \;
j=k+1,...,m
$$

이라면 $M_k^{-1}=I+\boldsymbol{l}_k \boldsymbol{e}_k^T$이다. 참고로 $\boldsymbol{l}_k(k+1:m)=\tilde{A}_k(k+1:m,k)/\tilde{A}_k(k,k)$이다.
 $k$ step 이후에, 다음을 얻을 수 있다.
$$
\tilde{A}_k(:,k)=\boldsymbol{x}_k=\left[\begin{array}{c}x_{1k}\\ \vdots \\x_{kk}\\x_{k+1,k} \\ \vdots\\
x_{m,k}\end{array} \right] ~~\Longrightarrow~~
\tilde{A}_{k+1}(:,k)=M_k \boldsymbol{x}_k=\boldsymbol{x}_k-{\boldsymbol{l}}_k
(\boldsymbol{e}_k^T \boldsymbol{x}_k)
=\left[\begin{array}{c}x_{1k}\\
\vdots \\x_{kk}\\0
\\ \vdots \\ 0\end{array} \right]
$$

그리고 $j=k+1,...,m$,에 대해서,
$$
\tilde{A}_{k+1}(:,j)=M_k \boldsymbol{x}_j=\boldsymbol{x}_j-{\boldsymbol{l}}_k
(\boldsymbol{e}_k^T \boldsymbol{x}_j) =\boldsymbol{x}_j-\boldsymbol{l}_k x_{kj}=\left[\begin{array}{c}x_{1j}\\
\vdots \\x_{kj}\\\times
\\ \vdots \\ \times\end{array} \right]\\
\Longrightarrow
\tilde{A}_{k+1}(k+1:m,j)=\tilde{A}_k(k+1:m,j)-\boldsymbol{l}_k(k+1:m)*\tilde{A}_k(k,j)
$$

왜냐하면 $M_k^{-1}M_{k+1}^{-1}=I+l_k e_k^T+l_{k+1} e_{k+1}^T$이며, 다음을 얻는다.
$$
L=M_1^{-1}\cdots M_{m-1}^{-1}= \left[\begin{array}{ccccc} 1 & & &
& \\ l_{21} & 1 & & & \\ l_{31} & l_{32} & 1 & &\\
\vdots & \vdots & \vdots & \ddots & \\ l_{m1} & l_{m2} & \cdots &
l_{m,m-1}& 1
\end{array} \right]
$$

삼각 행렬분해(triangular factorization)은 $L$과 $U$가 대각에 1을 가지고 $D$가 피벗의 대각 행렬일 때 종종 쓰인다. $\P$

[**Algorithm 1A.1**. 피벗팅을 제외한 가우시안 소거법 (Gaussian Elimination without Pivoting)]

```matlab
function [L,U] = slu(A)

​     [m,m] = size(A);
​     for k=1:m-1
​       A(k+1:m,k) = A(k+1:m,k)/A(k,k);
​       A(k+1:m,k+1:m) = A(k+1:m,k+1:m) - A(k+1:m,k)*A(k,k+1:m);
​     end
​     L = eye(m,m) + tril(A,-1);
​     U = triu(A);
```

이 알고리즘에서,  $L$​과 $U$​는 $A$​와 똑같은 행렬에 덮어 씌우며 컴퓨터의 메모리 사용을 최소화 할 수 있다. 

가우시안 소거법의 경우 $LU$​분해 계산 비용은 $\sim \frac{2}{3} m^3$​​ flops이다. 



#### 피벗팅 가우시안 소거법 (Gaussian Elimination with Pivoting)

$LU$ 분해는 가우시안 소거법을 통해 가장 잘 이해할 수 있다. 가우시안 소거법에서, 행렬은 행 연산을 통해 수정되어 상삼각행렬  $U$를 만든다.이 행 연산들을 따라가면, $L$ 행렬을 찾을 수 있다. 행 연산(row operation)은 *한 행을 다른 행들과 선형 결합(linear combination)한 결과로 대체* 하는 것이다. $U$의 대각성분들은 **피벗(pivots)**라고 하며, $A$가 diagnoally dominant하다면 피벗팅은 피해야 한다.  


---



> **정리 1A.2 [$PA=LU$]** 

모든 $m \times m$ 행렬 $A$에 대해  $PA=LU$ 를 만족하는 치환행렬(permutation matrix) $P$, 단위 대각행렬을 갖는 하삼각행렬 $L$,  $m \times m$ 상삼각행렬 $U$가 있으며, 이를 <span>**피벗팅 LU 분해(LU decomposition with pivoting)**</span> 라 한다.


---


**(증명)** 예를 들어, 행렬 $A$를 생각하자. (자세한 증명은 [^Trefethen and Bau. (1997)] 또는 [^Golub et al. (1995)] 을 참조하라.)
$$
\underbrace{\left[\begin{array}{ccccc}\times & \times & \times & \times & \times\\
& \times & \times & \times & \times\\
& \times & \times & \times & \times\\
& x_{ik} & \times & \times & \times\\
& \times & \times & \times & \times\end{array} \right]}_{\rm
Pivot\; selection} ~~{\underrightarrow{P_1}}~~
\underbrace{\left[\begin{array}{ccccc}\times & \times & \times & \times & \times\\
& x_{ik} & \times & \times & \times\\
& \times & \times & \times & \times\\
& \times & \times & \times & \times\\
& \times & \times & \times & \times\end{array} \right]}_{\rm Row\;
interchange} ~~{\underrightarrow{M_1}}~~
\underbrace{\left[\begin{array}{ccccc}\times & \times & \times & \times & \times\\
& x_{ik} & \times & \times & \times\\
& 0 & \times & \times & \times\\
& 0 & \times & \times & \times\\
& 0 & \times & \times & \times\end{array} \right]}_{\rm
Elimination}
$$

 $m-1$ steps 후에, 행렬은 상삼각행렬  $U$ 가 된다.

$$
M_{m-1}P_{m-1}\cdots M_2P_2M_1P_1A=U \label{eq:gep}
$$

이 때, $P_2 M_1=(P_2 M_1 P_2^{-1})P_2$ 이며,  $P_2 M_1 P_2^{-1}$ 는 다음과 같다.
$$
\left[\begin{array}{ccccc}1 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 0\\
0 & 0 & 1 & 0 & 0\\
0 & 1 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 1\end{array} \right]
\left[\begin{array}{ccccc}1 & 0 & 0 & 0 & 0\\
\boldsymbol{\tau_1} & 1 & 0 & 0 & 0\\
{\tau_2} & 0 & 1 & 0 & 0\\
\boldsymbol{\tau_3} & 0 & 0 & 1 & 0\\
{\tau_4} & 0 & 0 & 0 & 1\end{array} \right]
\left[\begin{array}{ccccc}1 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 0\\
0 & 0 & 1 & 0 & 0\\
0 & 1 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 1\end{array} \right]=\left[\begin{array}{ccccc}1 & 0 & 0 & 0 & 0\\
\boldsymbol{\tau_3} & 1 & 0 & 0 & 0\\
{\tau_2} & 0 & 1 & 0 & 0\\
\boldsymbol{\tau_1} & 0 & 0 & 1 & 0\\
{\tau_4} & 0 & 0 & 0 & 1\end{array} \right]
$$

$$
M_k'=P_{m-1}\cdots P_{k+1}M_kP_{k+1}^{-1}\cdots P_{m-1}^{-1}
$$

라 하자.

그렇다면 $M_k'$는 단위 하삼각행렬이며 쉽게 역연산이 가능하다. (피벗팅을 하지 않는 가우시안 소거법처럼 negating the sub-diagonal entries을 통해)

따라서, 
$$
(M_{m-1}'\cdots M_{2}'M_{1}')(P_{m-1}\cdots P_{2}P_{1})A=U
$$

$L=(M_{m-1}'\cdots M_{2}'M_{1}')^{-1}$ , $P=P_{m-1}\cdots P_{2}P_{1}$ 라 하면, 
$$
PA=LU
$$
이다.

이 때, $P^{-1}=P^T$이다. $\P$

[**Algorithm 1A.2**. 피벗팅 가우시안 소거법 (Gaussian Elimination with Pivoting)]

```matlab
 function [L,U,P] = plu(A)

​     [m,n] = size(A); P = eye(m, m);
​     for k=1:m-1
​       [maxv,r] = max(abs(A(k:m,k)));
​       q = r+k-1;
​       P([k q],:) = P([q k],:);
​       A([k q],:) = A([q k],:);
​       if A(k,k) ~= 0
​          A(k+1:m,k) = A(k+1:m,k)/A(k,k);
​          A(k+1:m,k+1:n) = A(k+1:m,k+1:n) - A(k+1:m,k)*A(k,k+1:n);
​       end
​     end
​     L = eye(m,m) + tril(A,-1);
​     U = triu(A);
```

피벗팅의 목적은 가우시안 소거법을 모든 행렬에 대해 적용 가능하고 안정되게 만드는 것이다.  안정성의 측면에서, 피벗팅은 일반적으로 $\|L\|$ 을 order 1로 보장하며,  $\|U\|$ 를 order of  $\|A\|$ 로 보장한다. 하지만, 특정 행렬 $A$에 대해서는  $\|U\|/\|A\|$ 가 매우 클 수도 있다. 예를 들어 피벗팅이 일어나지 않는 $PA=LU$ 분해에서,
$$
A=\left[\begin{array}{ccccc}1 & 0 & 0 & 0 & 1\\
-1 & 1 & 0 & 0 & 1\\
-1 & -1 & 1 & 0 & 1\\
-1 & -1 & -1 & 1 & 1\\
-1 & -1 & -1 & -1 & 1\end{array} \right]=\left[\begin{array}{ccccc}1 & 0 & 0 & 0 & 0\\
-1 & 1 & 0 & 0 & 0\\
-1 & -1 & 1 & 0 & 0\\
-1 & -1 & -1 & 1 & 0\\
-1 & -1 & -1 & -1 & 1\end{array} \right]\left[\begin{array}{ccccc}1 & 0 & 0 & 0 & 1\\
0 & 1 & 0 & 0 & 2\\
0 & 0 & 1 & 0 & 4\\
0 & 0 & 0 & 1 & 8\\
0 & 0 & 0 & 0 & 16\end{array} \right]
$$

이 패턴은 다음과 같이 임의 차원의 $u_{mm}=2^{m-1}$을 갖는 임의의  $m$차원의 행렬로 계속 이어질 수 있다.

그러나 이러한 예에도 불구하고 부분 피벗팅은 실제로 매우 안정적이다. 만약 무작위로 수억 개의 행렬들중에서  $A$를 임의로 고른다면, 대부분의 행렬에서 이런 현상을 보지 못할 것이다.



### 1A.3 보충 설명: 1.2.3절 

[그림1-2.1. 희소행렬] 의 MATLAB 구현 예시

```matlab
      % For a Tridiagonal matrix
​       n=50;
​       S = randn(n,n)+10*eye(n,n);
​       S = triu(tril(S,1),-1);
​       figure(1);
​       [L,U]=lu(S);
​       Sinv=inv(S);
​       subplot(2,2,1);spy(S);title('S')
​       subplot(2,2,2);spy(L);title('L')
​       subplot(2,2,3);spy(U);title('U')
​       subplot(2,2,4);spy(Sinv);title('S^{-1}')

​       % For a Tridiagonal matrix with nonzeros in the first column and row

​       S(:,1) = .3; S(1,:) = .3;
​       figure(2);
​       [L,U]=lu(S);
​       Sinv=inv(S);
​       subplot(2,2,1);spy(S);title('S')
​       subplot(2,2,2);spy(L);title('L')
​       subplot(2,2,3);spy(U);title('U')
​       subplot(2,2,4);spy(Sinv);title('S^{-1}')
```


---



> **예제 1A.1**

$A$ 가 3중 대각 행렬 (Tridiagonal matrix)이지만 첫번 째 행과 열은 0이 아닌 행렬이라고 가정하자.

```matlab
% Script File: ShowSparseSolve
​      % Illustrates how the \ operator can exploit sparsity

​      clc; disp('        n     Time Full    Time Sparse ')
​      disp('------------------------------------------')
​      for n=[1000 2000 5000 10000 20000]
​        T = randn(n,n)+1000*eye(n,n);
​        T = triu(tril(T,1),-1); T(:,1) = .1; T(1,:) = .1;
​        b = randn(n,1);
​        tic; x = T\b; fullTime   = toc;
​        T_sparse = sparse(T);
​        tic; x = T_sparse\b; sparseTime = toc;
​        disp(sprintf('%10d  %10f  %10f ',n,fullTime,sparseTime))
​      end
```

이 결과는 $A$가 희소 행렬인 경우 필요한 컴퓨팅 시간은 $O(n)$ 으로 선형적으로 증가하고, 그렇지 않은 경우 $O(n^3)$ 으로 증가한다는 것을 보여준다.

            n     Time Full    Time Sparse
         -------------------------------------

​          1000    0.034818    0.002963
​          2000    0.140362    0.005990
​          5000    1.640370    0.017338
​         10000   12.390639    0.053636
​         20000   75.685054    0.186542

------




### 1A.4 보충 설명: 1.3절 

행렬식은 다음의 세 가지 가장 기본적인 속성으로 정의할 수 있다.


---



> **정의 1A.2. [행렬식; Determinant]**

-   행렬식은 첫 번째 행에 선형적으로 의존한다.
-   행렬식은 두 행이 교환될 때 부호가 바뀐다.
-   단위 행렬의 행렬식은  $1$이다.


---


위 행렬식 정의로부터, 다음이 성립함을 쉽게 보일수 있다.

- $A$의 두 행이 같으면, $\mathrm{det}(A)=0$.
- 한 행의 n배를 다른 행에서 빼는 연산은 행렬식을 변화시키지 않는다.
- $A$ 가 0인 행이나 열을 가지면, $\mathrm{det}(A)=0$.
- $A$ 의 전치 행렬의 행렬식은 $A$  자신의 행렬식과 같다
  $\mathrm{det}(A^T ) = \mathrm{det}(A)$.


---



> **정리 1A.3 [행렬식의 공식]** 정방행렬(square matrix) $A=(a_{ij})\in \Re^{n\times n}$ 에 대해

1)  행렬식은 피벗에 대한 수식을 제공한다. 즉, $A$ 가 가역 행렬이면, $A=P^{-1} LU$로 부터 

$$
\mathrm{det}(A)=\mathrm{det}(P^{-1} LU)
      =\pm \mbox{(products of the pivots)}
$$

가 성립하여, 소거(elimination) 순서와는 상관 없이 모든 피벗의 곱은 부호를 제외하면 일정하게 유지된다. 이 때 , $P$의 행렬식은 행이 짝수번 또는 홀수번 교환되었는지에 따라 $+1$ 또는 $-1$을 갖고, $\mathrm{det}(L)=1$ 이고$\mathrm{det}(U)=u_{11} ... u_{nn}$를 만족한다. 

2) $\mathrm{det}(A)= \sum_{\sigma} a_{1\sigma_{1}} a_{2\sigma_{2}}\cdots a_{n \sigma_{n}}
   \mathrm{det} P_{\sigma}$ 

이 때, 총합은  $(1,...,n)$까지의 모든  $n!$ permutations $\sigma$ 이다.

3) $A$의 행렬식은 $i$ 행과 i행의 여인수(cofactor)의 조합이다.

$$
\mathrm{det} (A) = a_{i1} A_{i1} +a_{i2} A_{i2} +\cdots+a_{{i n}}A_{{i n}}
$$

이 때, 여인수  $A_{ij} $ 는  $M_{ij }$의 부호를 갖는 행렬식이며, 
$$
A_{ij} = (-1)^{i+j} \mathrm{det}(M_{ij })
$$
위의 $M_{ij}$ 는 $A$에서 $i$행과 $j$열을 지운 행렬이다.


---


​    

정방행렬(square matrix) $A=(a_{ij})\in \Re^{n\times n}$ 의 수반행렬(Adjoint matrix) $\mathrm{adj}(A)=(A_{ji})\in \Re^{n\times n}$는 다음과 같이 정의된다.
$$
\mathrm{adj} (A) :=
\left[ \begin{array}{rrrr}A_{11} &A_{21} &\cdots &A_{n1}
\\A_{12} &A_{22} &\cdots &A_{n2}\\ \vdots & \vdots & \ddots& \vdots  \\
     A_{1n} &A_{2n} &\cdots &A_{nn} \end{array}
\right]
$$

[**Algorithm 1A.3** PA=LU의 행렬식]

```matlab
function d = determ(A)

​     [L, U, P] = lu(A);
​     d = det(P) * prod(diag(U));
```

[**Algorithm 1A.4** $A$의 $i$행과 $j$열의 여인수(cofactor) $A_{ij}$]

```matlab
function C = cofactor(A, i, j)

​     M = A;
​     M(i,:) = [];
​     M(:,j) = [];
​     C = (-1)^(i+j) * det(M);
```



#### 행렬식의 활용(Applications of determinants)

$A^{-1}$의 계산, $Ax=b$의 해 , 평행다면체의 부피, 피벗의 공식은 모두 다음 정리에 바탕을 둔다.


---



> **정리 1A.4. [행렬식의 활용]** 정방행렬(square matrix) $A=(a_{ij})\in \Re^{n\times n}$ 에 대해

(1) $A$가 가역이면, $\mathrm{det}(A)\neq 0$ 이고 
$$
A^{-1} = \frac{\mathrm{adj}( A)}{ \mathrm{det}(A)}
$$
이다.

(2) 크래머 공식(Cramer’s rule): 각각의 원소 $b$에 대한 $A^{-1} b $의 의존도를 측정한다. 만약 한 변수가 실험에서 바뀌거나 관찰값이 수정되면,  $x=A^{-1} b$ 의 "영향 계수(influence coefficient)"가 행렬식의 비율과 동일하다. 즉, 

$x=A^{-1} b$ 의 $j$번째 성분은,
$$
x_{j }= \frac{ \mathrm{det}(B_{j} )}{
\mathrm{det}(A)} ,\; \mbox{where} \; B_{j} = \left[
\begin{array}{rrrrr}a_{11} & \cdots & b_{1} &\cdots
&a_{1n}\\a_{21}
&\cdots &b_{2} &\cdots &a_{2n} \\   \vdots & & \vdots &  & \vdots  \\
      a_{n1} & \cdots & b_{n} &\cdots &a_{nn} \end{array}
\right]
$$

$B_{j}$는 벡터 $b$로 $A$ 행렬의 j번째 열의 값을 바꾼 행렬이다 . 

(3) 가우시안 소거법으로 $A$를 행간의 교환이나 치환 행렬 연산 없이 수행할 수 있다는 것은 선행되는 부분행렬 $A_{1} , ..., A_{n}$이 비특이행렬(nonsingular)이라는 것과 필요충분조건이다.


---


**(증명)** (1) 아래의 식을 보자.
$$
A \cdot \mathrm{adj} (A) =
\left[ \begin{array}{rrrr}a_{11} &a_{12} &\cdots
&a_{1n} \\a_{21} &a_{22} &\cdots &a_{2n} \\ \vdots & \vdots & \ddots& \vdots \\
        a_{n1} &a_{n2} &\cdots &a_{nn} \end{array}
\right] \left[ \begin{array}{rrrr}A_{11} &A_{21} &\cdots &A_{n1}
\\A_{12} &A_{22} &\cdots &A_{n2}\\ \vdots & \vdots & \ddots& \vdots  \\
     A_{1n} &A_{2n} &\cdots &A_{nn} \end{array}
\right] \\ =\left[ \begin{array}{rrrr}\mathrm{det}(A) &0 &\cdots &0  \\
0 &\mathrm{det}(A) &\cdots &0 \\  \vdots & \vdots & \ddots& \vdots  \\
    0 &0 &\cdots &\mathrm{det}(A) \end{array}
\right]= \mathrm{det}(A) I
$$

대각 성분을 제외한 모든 곳에서 성분이 0인 것을 증명하기 위하여 $j\neq i$이며 $i$번째 행이 $B$의 $j$ 번째 행에 복사되는 행만을 제외하고 $B$를 $A$와 동일하게 설정하자. 그러면,
$$
\mathrm{det}(B)=0= a_{i1} A_{j1} +a_{i2} A_{j2} +\cdots+a_{i n}
A_{j n},\; \forall i\neq j
$$
(2) $\mathrm{det}(B_{j})=(\mathrm{adj}(A)b)_j$.
(3) $A$가 LDU로 분해된다면, 좌상단 코너(upper left corners)에서는 다음을 만족한다.
$$
A_{k }= L_{k} D_{k} U_{k}
$$

모든 k에 대해, 부분행렬 $A_{k}$는 가우시안 소거법을 거친다. 특히 피벗 $d_{k}$는 행렬식의 비율로 표현할 수 있다. 즉,
$$
d_{k} = \frac{\mathrm{det}(A_{k})}{ \mathrm{det}(A_{k-1} )}
$$

[**Algorithm 1A.5** 크레머 공식]

```matlab
function x = cramer(A, b)

​     [m, n] = size(A);
​     for j = 1:n
​       B = A;
​       B(:, j) = b;
​       x(j) = det(B)/det(A);
​     end
```



#### 행렬식과 역행렬 공식(Matrix Inversion and Determinant Formula)

주로 사용되는 대각합(traces)와 행렬식(determinants)의 곱의 법칙과 항등식은 다음과 같다. 


---


**보조정리 1A.5 [행렬식 공식]**

(1) **대각합(trace)와 행렬식(determinant)의 곱의 법칙:**
$$
\label{td}
 \mathrm{tr}(\boldsymbol{A}\boldsymbol{B})=\mathrm{tr}(\boldsymbol{B}\boldsymbol{A}),\qquad \boldsymbol{A}\in \Re^{n\times p},\; \boldsymbol{B}\in \Re^{p\times n}\\
\mathrm{det}(\boldsymbol{A}\boldsymbol{B})=\mathrm{det}(\boldsymbol{A})\mathrm{det}(\boldsymbol{B})=\mathrm{det}(\boldsymbol{B}\boldsymbol{A}),\qquad \boldsymbol{A}\in \Re^{n\times n},\; \boldsymbol{B}\in \Re^{n\times n}
$$

(2) **실베스터 행렬항등식 (Sylvester’s determinant identity):**
$$
\label{sdi}
\mathrm{det}(\boldsymbol{I}_m+\boldsymbol{A}\boldsymbol{B}^T)=\mathrm{det}(\boldsymbol{I}_n+\boldsymbol{B}^T\boldsymbol{A})\qquad
\mbox{where}\quad \boldsymbol{A},\boldsymbol{B}\in \Re^{m\times n}
$$


---


**(증명)** (1) 증명하기 쉽다.
(2) $P$ 와$Q$ 를 다음과 같은 네 개의 블록으로 구성된 행렬이라 하면,
$$
P=\begin{bmatrix}
  {\boldsymbol{I}_m} & {-\boldsymbol{A}} \\
  {\boldsymbol{B}^T} & {\boldsymbol{I}_n}
\end{bmatrix},\qquad Q=\begin{bmatrix}
  {\boldsymbol{I}_n} & {\boldsymbol{B}^T} \\
  {-\boldsymbol{A}} & {\boldsymbol{I}_m}
\end{bmatrix}
$$

블록 행렬 $P$의 행렬식은 다음과 같다. 
$$
\mathrm{det}(P)=\mathrm{det}\begin{bmatrix}
  {\boldsymbol{I}_m} & {-\boldsymbol{A}} \\
  {\boldsymbol{B}^T} & {\boldsymbol{I}_n}
\end{bmatrix}=\mathrm{det}\begin{bmatrix}
  {\boldsymbol{I}_m} & {-\boldsymbol{A}} \\
  {\boldsymbol{0}} & {\boldsymbol{I}_n-\boldsymbol{B}^T(-\boldsymbol{A})}
\end{bmatrix}\\
=\mathrm{det}(\boldsymbol{I}_m)\mathrm{det}(\boldsymbol{I}_n+\boldsymbol{B}^T\boldsymbol{A})
=\mathrm{det}(\boldsymbol{I}_n+\boldsymbol{B}^T\boldsymbol{A})
$$

이와 비슷하게 $Q$도 구할 수 있다.
$$
\mathrm{det}(\boldsymbol{I}_n+\boldsymbol{B}^T\boldsymbol{A})=\mathrm{det}(P)=\mathrm{det}(Q)
=\mathrm{det}(\boldsymbol{I}_m+\boldsymbol{A\boldsymbol{B}^T})
$$

 $\P$


---



> **예제 1A.2** 

슈어 보수행렬 보조정리를 통해
(i) 모든 대칭 행렬 $A\succ 0$ ,$C\succeq 0$에 대해:
$$
\label{shur2}
C\succeq B^TA^{-1}B \quad \Longleftrightarrow \quad  \begin{bmatrix}
  {A} & {B} \\
  {B^T} & {C}
\end{bmatrix}\succeq 0
$$
(ii) 모든 대칭 행렬$A\succ 0$ , $C\succeq 0$에 대해:
$$
\label{shur3}
\left. \begin{array}{c}
        \mathcal{R}(B)\perp \mathcal{N}(A) \\
        C\succeq B^TA^{-1}B
        \end{array} \right\}
 \quad \Longleftrightarrow \quad  \begin{bmatrix}
  {A} & {B} \\
  {B^T} & {C}
\end{bmatrix}\succeq 0
$$

*Hint:* Use
$$
\begin{bmatrix}
  {A} & {B} \\
  {B^T} & {C}
\end{bmatrix}=  \left[ \begin{array}{cc}
        R^T & 0 \\
        B^TR^{-1} & I
        \end{array} \right]
        \left[ \begin{array}{cc}
        I & 0 \\
        0 & C-B^TA^{-1}B
        \end{array} \right]
        \left[ \begin{array}{cc}
        R & R^{-T}B \\
        0 & I
        \end{array} \right]
$$


---


### 1A.5 보충 설명: 1.4절 

[**Algorithm 1A.5** 콜레스키 분해] ​

```matlab
function R= cholesky(A)

​     [m,m] = size(A);
​     for k=1:m
​      if(A(k,k) < 0)
​        error('Matrix should be positive definite');
​      end
​      A(k,k) = sqrt(A(k,k));
​      A(k+1:m,k) = A(k+1:m,k)/A(k,k);
​      for j=k+1:m
​        A(j:m,j) = A(j:m,j) - A(j:m,k)*A(j,k);
​      end
​      A(k,k+1:m) = zeros(1,m-k);
​     end
​     R = A';
```


