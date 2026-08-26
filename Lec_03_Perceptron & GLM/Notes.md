# Generalized Linear Models

Linear regression assumed $y\mid x;\theta \sim N(\mu,\sigma^2)$, and logistic regression assumed $y\mid x;\theta \sim \text{Bernoulli}(\phi)$. This chapter shows both are **special cases of one broader family**: **Generalized Linear Models (GLMs)**.

```text
Linear Regression  (y | x ~ Gaussian)
Logistic Regression (y | x ~ Bernoulli)
             ↓
     Both are special cases of GLMs
```

---

## 1. The Exponential Family

### 1.1 What is it?

A class of distributions is in the **exponential family** if it can be written as:

$$
p(y;\eta) = b(y)\exp\big(\eta^T T(y) - a(\eta)\big)
$$

- $\eta$ → **natural parameter** (also called the canonical parameter)
- $T(y)$ → **sufficient statistic** (in most cases we consider, $T(y) = y$)
- $a(\eta)$ → **log partition function**
- $e^{-a(\eta)}$ → acts as a normalization constant, ensuring $p(y;\eta)$ sums/integrates to 1 over $y$

A fixed choice of $T$, $a$, and $b$ defines a **family** of distributions. Varying $\eta$ moves you through different distributions within that family.

> 💡 **Intuition:** Many familiar distributions (Bernoulli, Gaussian, Poisson, multinomial, gamma, exponential, beta, Dirichlet, ...) are just different "settings" of this same general template. Once you can write a distribution in this form, you can plug it into the GLM recipe below.

---

### 1.2 Bernoulli as an Exponential Family Distribution

The Bernoulli distribution over $y \in \{0,1\}$ with mean $\phi$:

$$
p(y=1;\phi) = \phi, \qquad p(y=0;\phi) = 1-\phi
$$

Rewritten:

$$
p(y;\phi) = \phi^y(1-\phi)^{1-y} = \exp\left(y\log\phi + (1-y)\log(1-\phi)\right) = \exp\left(\log\frac{\phi}{1-\phi}\,y + \log(1-\phi)\right)
$$

Matching this to the exponential family form gives:

- **Natural parameter:** $\eta = \log\dfrac{\phi}{1-\phi}$
- $T(y) = y$
- $a(\eta) = -\log(1-\phi) = \log(1+e^\eta)$
- $b(y) = 1$

Inverting $\eta = \log(\phi/(1-\phi))$ for $\phi$:

$$
\phi = \frac{1}{1+e^{-\eta}}
$$

> ⚠️ **Important:** This is exactly the **sigmoid function**. It is not a coincidence — this is where the sigmoid form used in logistic regression actually comes from, as shown formally in Section 3 below.

---

### 1.3 Gaussian as an Exponential Family Distribution

For linear regression, the value of $\sigma^2$ never affected the choice of $\theta$ or $h_\theta(x)$, so we simplify by setting $\sigma^2=1$.

$$
p(y;\mu) = \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}(y-\mu)^2\right) = \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}y^2\right)\cdot \exp\left(\mu y - \frac{1}{2}\mu^2\right)
$$

Matching to the exponential family form:

- **Natural parameter:** $\eta = \mu$
- $T(y) = y$
- $a(\eta) = \mu^2/2 = \eta^2/2$
- $b(y) = \dfrac{1}{\sqrt{2\pi}}\exp(-y^2/2)$

> ⚠️ **Important:** If $\sigma^2$ is left as a free variable, the Gaussian can still be written in the exponential family, but $\eta$ becomes a 2-dimensional vector depending on both $\mu$ and $\sigma$. This requires a more general form $p(y;\eta,\tau) = b(y,\tau)\exp\big((\eta^TT(y)-a(\eta))/c(\tau)\big)$, where $\tau$ is the **dispersion parameter** ($c(\tau)=\sigma^2$ for the Gaussian). Fixing $\sigma^2=1$ avoids needing this generalization.

| Distribution | η (natural parameter) | T(y) | a(η) | b(y) |
|---|---|---|---|---|
| Bernoulli($\phi$) | $\log\frac{\phi}{1-\phi}$ | $y$ | $\log(1+e^\eta)$ | $1$ |
| Gaussian($\mu,1$) | $\mu$ | $y$ | $\eta^2/2$ | $\frac{1}{\sqrt{2\pi}}e^{-y^2/2}$ |

Other exponential family members: multinomial, Poisson (count data), gamma and exponential (continuous non-negative variables, e.g. time intervals), beta and Dirichlet (distributions over probabilities).

---

## 2. Constructing GLMs

### 2.1 Motivating Example

Suppose you want to predict $y$ = number of customers arriving at a store per hour, from features $x$ (promotions, advertising, weather, day-of-week). Counts are well modeled by the **Poisson distribution**. Since Poisson is an exponential family distribution, GLMs give a systematic way to build this model.

### 2.2 The Three GLM Assumptions/Design Choices

To derive a GLM for predicting $y$ from $x$:

1. **$y\mid x;\theta \sim \text{ExponentialFamily}(\eta)$** — given $x$ and $\theta$, $y$'s distribution is some exponential family distribution with parameter $\eta$.
2. **$h(x) = E[T(y)\mid x]$** — the hypothesis should predict the expected value of $T(y)$ given $x$. Since $T(y)=y$ in our examples, this means $h(x) = E[y\mid x]$.
3. **$\eta = \theta^Tx$** — the natural parameter and the inputs are related **linearly**. (If $\eta$ is vector-valued, then $\eta_i = \theta_i^Tx$.)

> ⚠️ **Important:** Assumption 3 is the least theoretically justified of the three — it's better understood as a **design choice** in the GLM "recipe" rather than a strict assumption. Together, these three choices produce an elegant class of learning algorithms with desirable properties like ease of learning.

```text
y | x; θ ~ ExponentialFamily(η)      (Assumption 1)
        ↓
h(x) = E[y | x]                      (Assumption 2)
        ↓
η = θᵀx                              (Assumption 3, linear design choice)
        ↓
   hθ(x) derived from these three
```

---

### 2.3 Ordinary Least Squares as a GLM

Model $y\mid x \sim N(\mu,\sigma^2)$ (so ExponentialFamily = Gaussian). Recall from Section 1.3 that $\mu = \eta$ for the Gaussian.

$$
h_\theta(x) = E[y\mid x;\theta] = \mu = \eta = \theta^Tx
$$

- 1st equality: Assumption 2
- 2nd equality: $y\mid x;\theta\sim N(\mu,\sigma^2)$, so its mean is $\mu$
- 3rd equality: from the Gaussian's exponential-family form, $\mu=\eta$ (Assumption 1)
- 4th equality: Assumption 3

This recovers exactly $h_\theta(x) = \theta^Tx$ — ordinary least squares.

---

### 2.4 Logistic Regression as a GLM

Model $y\mid x \sim \text{Bernoulli}(\phi)$, since $y\in\{0,1\}$. Recall from Section 1.2 that $\phi = 1/(1+e^{-\eta})$, and for a Bernoulli, $E[y\mid x;\theta] = \phi$.

$$
h_\theta(x) = E[y\mid x;\theta] = \phi = \frac{1}{1+e^{-\eta}} = \frac{1}{1+e^{-\theta^Tx}}
$$

This is exactly the logistic/sigmoid hypothesis used in logistic regression.

> 💡 **Intuition:** This answers the question "why the sigmoid function, specifically?" — it isn't an arbitrary S-shaped choice. Once you assume $y\mid x$ is Bernoulli and apply the GLM recipe, the sigmoid **falls out automatically** as a mathematical consequence.

```text
Assume y | x ~ Bernoulli(φ)
         ↓
   φ = 1/(1+e^-η)   (from exponential family form)
         ↓
   η = θᵀx           (GLM linearity assumption)
         ↓
hθ(x) = 1/(1+e^-θᵀx)   ← the sigmoid, derived, not assumed
```

---

### 2.5 Canonical Response and Link Functions

- **Canonical response function:** $g(\eta) = E[T(y);\eta]$ — gives the distribution's mean as a function of the natural parameter.
- **Canonical link function:** $g^{-1}$ — the inverse of the response function.

| Distribution | Canonical response function $g(\eta)$ |
|---|---|
| Gaussian | Identity function |
| Bernoulli | Logistic (sigmoid) function |

---

## Key Takeaways

- **Exponential family form:** $p(y;\eta) = b(y)\exp(\eta^TT(y)-a(\eta))$, with $\eta$ = natural parameter, $T(y)$ = sufficient statistic (usually $T(y)=y$), $a(\eta)$ = log partition function (normalizes the distribution).
- **Bernoulli** and **Gaussian** (with $\sigma^2=1$) are both exponential family distributions:
  - Bernoulli: $\eta=\log\frac{\phi}{1-\phi}$, and inverting gives $\phi = 1/(1+e^{-\eta})$ — the sigmoid.
  - Gaussian: $\eta=\mu$.
- **GLMs** are built from three assumptions/design choices: (1) $y\mid x \sim \text{ExponentialFamily}(\eta)$, (2) $h(x)=E[y\mid x]$, (3) $\eta = \theta^Tx$ (linear relationship).
- **Ordinary least squares** = GLM with a Gaussian response, giving $h_\theta(x) = \theta^Tx$.
- **Logistic regression** = GLM with a Bernoulli response, giving $h_\theta(x) = 1/(1+e^{-\theta^Tx})$ — this derivation is *why* the sigmoid function is the natural choice for binary classification, rather than an arbitrary pick.
- The **canonical response function** $g(\eta)=E[T(y);\eta]$ maps the natural parameter to the distribution's mean; its inverse is the **canonical link function**. Identity for Gaussian, logistic/sigmoid for Bernoulli.
- **Big picture:** exponential family → GLM recipe → both linear regression and logistic regression emerge as special cases, unifying two previously separate-seeming algorithms under one framework.
