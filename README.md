# math4ai-notes
AI 수학 교과서 (Mathematical Foundations for AI) 수업 보조 자료 저장소 입니다.

**AI 수학 교과서 (Mathematical Foundations for AI)** 내용을 기반으로 한 강의 노트와 실습 코드를 제공합니다.

---
## 💬 의견을 남겨주세요!

이 챕터에 대한 질문이나 의견은 [**여기(Discussions)**](https://github.com/suebinhan/math4ai-notes/discussions/new?category=general)에 자유롭게 남겨주세요. 함께 공부하며 성장하는 공간이 되었으면 합니다! ✨

---

##  내용 (Contents)

**1부: 행렬 계산 (Matrix Computations)**

- **제1장: 선형 방정식과 LU 분해**
  - 연립선형방정식의 해법과 가우스 소거법의 행렬 표현
  - 양의 정치 행렬(Positive Definite Matrix)의 성질과 수치적 최적화의 기초
  - 숄레스키 분해(Cholesky Factorization): 대칭 양의 정치 행렬의 효율적 분해 기법
- **제2장: 벡터 공간과 노름 (Vector Spaces and Norms)**
  - 벡터 공간(Vector Spaces)의 공리적 정의와 부분 공간의 구조
  - 내적(Inner Products)과 노름(Norms): 데이터의 크기, 거리 및 유사도 측정 이론
  - 코시-슈바르츠 부등식(Cauchy-Schwarz Inequality)의 증명과 기하학적 의미
  - 선형대수학의 기본 정리와 네 가지 주요 부분 공간(Four Fundamental Subspaces) 분석
- **제3장: 최소자승법과 QR 분해 (Least Squares and QR Factorization)**
  - 선형 변환의 기하학적 해석과 직교 투영(Orthogonal Projection)
  - 최소자승법(Least Squares): 오차 제곱합을 최소화하는 최적 근사해 도출
  - QR 분해: 그람-슈미트 직교화 및 하우스홀더 변환을 통한 행렬의 직교 분해
- **제4장: 푸리에 변환 (Fourier Transform)**
  - 푸리에 급수(Fourier Series): 주기 함수를 삼각함수의 합으로 분해하는 조화 해석
  - 이산 푸리에 변환(DFT)과 연산 효율을 극대화한 고속 푸리에 변환(FFT) 알고리즘
  - 확률론에서의 특성 함수(Characteristic Functions) 및 푸리에 역변환의 응용



**2부: 스펙트럴 방법과 미분방정식 (Spectral Methods and Differential Equations)**

- **제5장: 특이값 분해 (Singular Value Decomposition, SVD)**
  - 고윳값과 고유벡터 및 스펙트럴 정리(Spectral Theorem)의 핵심 원리
  - SVD의 기하학적 해석과 행렬 구조 분석을 통한 데이터 압축의 수학적 토대
  - 무어-펜로즈 유사 역행렬(Pseudoinverse)을 이용한 일반화된 해법
- **제6장: 차원 축소를 위한 스펙트럴 방법 (Spectral Methods for Dimension Reduction)**
  - 주성분 분석(PCA): 데이터 분산을 최대화하는 저차원 사영 및 시각화
  - 다차원 척도법(MDS)과 고차원 데이터의 기하적 구조를 보존하는 매니폴드 학습(Manifold Learning)
  - 그래프 기반의 차원 축소 기법과 비선형 데이터의 특징 추출 전략
- **제7장: 미분 방정식 (Differential Equations)**
  - 선형 미분 방정식의 해법과 행렬 지수 함수(Matrix Exponential)를 이용한 시스템 해석
  - 상태 평면 분석(Phase Portraits)과 시스템의 동적 안정성 판별
  - 로지스틱 성장 및 포식자-피식자 모델을 통한 비선형 시스템의 모델링 및 분석



**3부: 최적화 (Optimization)**

- **제8장: 행렬 미적분학 (Matrix Calculus)**
  - 유클리드 공간의 위상과 다변수 함수의 연속성 및 미분 가능성
  - 그래디언트(Gradient)와 헤세 행렬(Hessian Matrix): 함수의 국소적 곡률 및 변화율 분석
  - 연쇄 법칙(Chain Rule)의 행렬 확장과 테일러 정리 및 볼록성(Convexity) 이론
- **제9장: 최적화 알고리즘 (Optimization Algorithms)**
  - 무제약 최적화의 최적성 조건과 라인 서치(Line Search) 기법
  - 준-뉴턴법(Quasi-Newton): 2차 미분 정보를 근사하여 수렴 속도를 높이는 알고리즘
  - 모멘텀(Momentum) 및 적응적 학습률(Adaptive Learning Rate) 기법의 수렴성 분석
- **제10장: 제약조건하의 최적화와 쌍대성 (Constrained Optimization and Duality)**
  - 등식 및 부등식 제약조건 하에서의 KKT 조건(Karush-Kuhn-Tucker conditions)
  - 라그랑주 쌍대성(Lagrangian Duality): 원 문제와 쌍대 문제의 관계 및 안장점(Saddle point) 이론



**4부: 통계적 학습 이론 (Statistical Learning Theory)**

- **제11장: 확률, 정보 및 추정 (Probability, Information, and Estimation)**
  - 엔트로피와 최대 엔트로피 원리를 통한 정보의 정량화
  - 다변량 정규 분포의 성질과 추정량의 편향(Bias), 분산(Variance) 분석
  - 최대 우도 추정(MLE)과 베이지안 추정(Bayesian Estimation)의 통계적 메커니즘
- **제12장: 선형 모형과 일반화 이론 (Linear Models and Generalization)**
  - PAC 학습이론과 VC 차원을 통한 학습 모델의 복잡도 및 일반화 성능 평가
  - 릿지 회귀(Ridge) 및 로지스틱 회귀를 포함한 선형 분류/회귀 모델의 심화 해석
- **제13장: 커널과 서포트 벡터 머신 (Kernels and Support Vector Machine)**
  - 재생 커널 힐베르트 공간(RKHS)과 머서 커널(Mercer Kernels)의 수학적 정의
  - SVM(Support Vector Machine): 마진 최대화와 커널 트릭을 통한 비선형 데이터 분리 기법



**5부: 현대 인공지능과 동역학 (Modern AI and Dynamics)**

- **제14장: 신경망 훈련 (Training Neural Networks)**
  - 오차 역전파(Backpropagation) 알고리즘의 행렬 미적분학적 유도
  - 순환 신경망(RNN)의 훈련과 시간 기반 역전파(BPTT) 및 기울기 소실 문제 해결(LSTM, GRU)
- **제15장: 어텐션 메커니즘과 트랜스포머 (Attention Mechanisms and Transformers)**
  - Seq2Seq 모델에서 Self-Attention으로의 발전 과정과 트랜스포머 아키텍처 심층 분석
  - Multi-Head Attention과 포지셔널 인코딩(Positional Encoding)의 수학적 역할
  - 최신 모델 효율화 기법인 LoRA 및 상태 공간 모델(SSM)로의 확장
- **제16장: 최적 운송과 분포 기하 (Optimal Transport and Distributional Geometry)**
  - 펜첼 쌍대성(Fenchel Duality)과 볼록 공액성 및 와세르슈타인 거리(Wasserstein Distance)
  - 최적 운송(Optimal Transport) 이론을 이용한 생성 모델(VAE, WAE)의 기하학적 최적화
- **제17장: 기하학적·확률적 동역학 (Geometric and Stochastic Dynamics)**
  - 리야푸노프 안정성(Lyapunov Stability) 분석과 그래디언트 플로우(Gradient Flows)
  - Neural ODE: 미분 방정식을 이용한 연속 시간 신경망 모델링
  - 확률 미분 방정식(SDE)과 최신 생성 AI의 핵심인 확산 모델(Diffusion Models)의 수학적 원리

---

## 🛠️ 사용 방법 (How to Use)

1. 저장소 클론:
   ```bash
   git clone https://github.com/DrJaewookLee/math4ai-notes.git
   cd math4ai-notes

<script src="https://giscus.app/client.js"
        data-repo="suebinhan/math4ai-notes"
        data-repo-id="R_kgDORoILJw"
        data-category="Announcements"
        data-category-id="DIC_kwDORoILJ84C4frI"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="preferred_color_scheme"
        data-lang="ko"
        crossorigin="anonymous"
        async>
</script>
