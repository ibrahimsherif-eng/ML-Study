# Classification and Logistic Regression

## 1. The Classification Problem

Like regression, but the target $y$ takes only a small number of discrete values. We focus on **binary classification**: $y \in \{0, 1\}$.

- $y = 0$: **negative class** (also "-")
- $y = 1$: **positive class** (also "+")
- $y^{(i)}$ is called the **label** for example $x^{(i)}$

**Example:** Spam classifier — $x^{(i)}$ = email features, $y^{(i)} = 1$ if spam, $0$ otherwise.

> ⚠️ **Important:** Using linear regression directly on a discrete $y$ works poorly, and $h_\theta(x)$ can output values outside $[0,1]$, which makes no sense for a probability of class membership.

---

## 2. Logistic Regression

### 2.1 The Hypothesis

To keep predictions bounded in $[0,1]$, logistic regression uses the **sigmoid (logistic) function**:

$$
h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}, \qquad g(z) = \frac{1}{1+e^{-z}}
$$

- $g(z)$ → sigmoid function
- $\theta^T x = \theta_0 + \sum_{j=1}^{d}\theta_j x_j$ (with convention $x_0 = 1$)

**Behavior of $g(z)$:**
- $z \to \infty \Rightarrow g(z) \to 1$
- $z \to -\infty \Rightarrow g(z) \to 0$
- Always bounded between 0 and 1

> 💡 **Intuition:** The sigmoid squashes any real number into a valid probability, so $h_\theta(x)$ can be interpreted as $P(y=1\mid x)$.

Other S-shaped functions could work too, but the sigmoid is the natural choice — this becomes clear later with GLMs and generative models.

**Useful derivative property:**

$$
g'(z) = g(z)(1-g(z))
$$

This identity is used repeatedly in the gradient derivation below.

---

### 2.2 Probabilistic Model

Assume:

$$
P(y=1\mid x;\theta) = h_\theta(x), \qquad P(y=0\mid x;\theta) = 1-h_\theta(x)
$$

Compactly:

$$
p(y\mid x;\theta) = (h_\theta(x))^y (1-h_\theta(x))^{1-y}
$$

- This single formula reduces to $h_\theta(x)$ when $y=1$ and $1-h_\theta(x)$ when $y=0$.

### 2.3 Likelihood and Log-Likelihood

Assuming the $n$ training examples are independent, the **likelihood** of the parameters is:

$$
L(\theta) = \prod_{i=1}^{n} h_\theta(x^{(i)})^{y^{(i)}} \left(1-h_\theta(x^{(i)})\right)^{1-y^{(i)}}
$$

Taking logs (easier to maximize) gives the **log-likelihood**:

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} y^{(i)} \log h(x^{(i)}) + (1-y^{(i)}) \log(1-h(x^{(i)}))
$$

Goal: find $\theta$ that **maximizes** $\ell(\theta)$ (Maximum Likelihood Estimation).

---

### 2.4 Fitting θ: Gradient Ascent

Since we're maximizing (not minimizing), the update uses a **positive** sign:

$$
\theta := \theta + \alpha \nabla_\theta \ell(\theta)
$$

**Derivation (single example $(x,y)$):**

$$
\frac{\partial}{\partial \theta_j}\ell(\theta) = (y - h_\theta(x))\, x_j
$$

This uses the sigmoid derivative identity $g'(z)=g(z)(1-g(z))$ to simplify the chain rule through $g(\theta^Tx)$.

This gives the **stochastic gradient ascent rule**:

$$
\theta_j := \theta_j + \alpha\left(y^{(i)} - h_\theta(x^{(i)})\right) x_j^{(i)}
$$

> ⚠️ **Important:** This update rule *looks identical* to the LMS rule from linear regression, but it is **not the same algorithm** — here $h_\theta(x)$ is a non-linear (sigmoid) function of $\theta^Tx$, whereas in linear regression it's linear. This apparent coincidence is explained later by the theory of **Generalized Linear Models (GLMs)**.

```text
Linear Regression  →  same-looking update rule  ←  Logistic Regression
        (hθ linear)                                    (hθ = sigmoid)
                     both explained by GLMs
```

---

### 2.5 Alternative View: Logistic Loss

Define the **logistic loss**:

$$
\ell_{\text{logistic}}(t,y) = y\log(1+e^{-t}) + (1-y)\log(1+e^{t})
$$

where $t = \theta^Tx$ is called the **logit**.

Substituting $h_\theta(x) = 1/(1+e^{-\theta^Tx})$ shows:

$$
-\ell(\theta) = \ell_{\text{logistic}}(\theta^Tx, y)
$$

i.e., the negative log-likelihood equals the logistic loss evaluated at the logit. Differentiating:

$$
\frac{\partial \ell_{\text{logistic}}(t,y)}{\partial t} = \frac{1}{1+e^{-t}} - y
$$

Applying the chain rule through $t = \theta^Tx$ gives the same gradient as before:

$$
\frac{\partial}{\partial \theta_j}\ell(\theta) = (y-h_\theta(x))\,x_j
$$

> 💡 **Intuition:** This "logit" framing is useful later for nonlinear models (e.g., neural networks), where the loss is expressed directly as a function of the logit rather than through the sigmoid explicitly.

---

## 3. Digression: The Perceptron Algorithm

Modify logistic regression to force **hard** outputs (exactly 0 or 1) by replacing the sigmoid with a **threshold function**:

$$
g(z) = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}
$$

Using $h_\theta(x) = g(\theta^Tx)$ with this new $g$, and the same-looking update:

$$
\theta_j := \theta_j + \alpha\left(y^{(i)} - h_\theta(x^{(i)})\right) x_j^{(i)}
$$

gives the **perceptron learning algorithm**.

- Historically viewed (1960s) as a rough model of a biological neuron.
- Important for learning theory as a simple starting point.

> ⚠️ **Important:** Despite the similar-looking update rule, the perceptron is fundamentally different from logistic regression and linear regression. Its predictions cannot be given a meaningful probabilistic interpretation, and it cannot be derived as a maximum likelihood estimator.

| Algorithm | Output type | Probabilistic interpretation | Derived from MLE? |
|---|---|---|---|
| Linear Regression | Continuous | Yes (Gaussian noise) | Yes |
| Logistic Regression | Probability in $[0,1]$ | Yes (Bernoulli) | Yes |
| Perceptron | Hard 0/1 | No | No |

---

## 4. Multi-Class Classification

Now $y \in \{1, \dots, k\}$ (e.g., spam / personal / work email). Modeled with a **multinomial distribution** over $k$ outcomes $\phi_1,\dots,\phi_k$, subject to $\sum_{i=1}^k \phi_i = 1$.

### 4.1 Why Not Direct Linear Outputs?

We introduce $k$ parameter vectors $\theta_1,\dots,\theta_k \in \mathbb{R}^d$ and want $\theta_1^Tx,\dots,\theta_k^Tx$ to represent probabilities. Two problems:

1. $\theta_j^Tx$ is not necessarily in $[0,1]$.
2. The values don't necessarily sum to 1.

### 4.2 The Softmax Function

The **softmax function** converts raw scores ("logits") $t_1,\dots,t_k$ into a valid probability vector:

$$
\text{softmax}(t_1,\dots,t_k) = \left[\frac{e^{t_1}}{\sum_{j=1}^k e^{t_j}}, \dots, \frac{e^{t_k}}{\sum_{j=1}^k e^{t_j}}\right]
$$

Setting $t_i = \theta_i^Tx$, the model becomes:

$$
P(y=i\mid x;\theta) = \phi_i = \frac{e^{\theta_i^Tx}}{\sum_{j=1}^k e^{\theta_j^Tx}}
$$

> 💡 **Intuition:** Exponentiating makes all values positive; dividing by the sum normalizes them to add up to 1 — this is the natural multi-class generalization of the sigmoid.

### 4.3 Cross-Entropy Loss

Negative log-likelihood of one example:

$$
-\log p(y\mid x,\theta) = -\log\frac{e^{t_y}}{\sum_{j=1}^k e^{t_j}}
$$

Define the **cross-entropy loss**:

$$
\ell_{ce}(t, y) = -\log\frac{e^{t_y}}{\sum_{j=1}^k e^{t_j}}
$$

Total loss over the dataset:

$$
\ell(\theta) = \sum_{i=1}^{n} \ell_{ce}\big((\theta_1^Tx^{(i)},\dots,\theta_k^Tx^{(i)}),\, y^{(i)}\big)
$$

### 4.4 Gradient of Cross-Entropy

With respect to logit $t_i$:

$$
\frac{\partial \ell_{ce}(t,y)}{\partial t_i} = \phi_i - \mathbb{1}\{y=i\}
$$

where $\mathbb{1}\{\cdot\}$ is the indicator function. In vector form:

$$
\frac{\partial \ell_{ce}(t,y)}{\partial t} = \phi - e_y
$$

($e_y$ = one-hot vector with 1 at position $y$.)

By the chain rule, gradient with respect to parameter vector $\theta_i$:

$$
\frac{\partial \ell(\theta)}{\partial \theta_i} = \sum_{j=1}^{n} \left(\phi_i^{(j)} - \mathbb{1}\{y^{(j)}=i\}\right) x^{(j)}
$$

where $\phi_i^{(j)}$ is the model's predicted probability of class $i$ for example $x^{(j)}$.

> 💡 **Intuition:** The gradient is simply "predicted probability minus actual (one-hot) label," times the input — the multi-class analog of $(y-h_\theta(x))x_j$ in binary logistic regression.

This gradient enables (stochastic) gradient descent to minimize $\ell(\theta)$.

---

## 5. Newton's Method for Maximizing ℓ(θ)

### 5.1 Newton's Method (Scalar Case)

For finding a root of $f(\theta)=0$:

$$
\theta := \theta - \frac{f(\theta)}{f'(\theta)}
$$

> 💡 **Intuition:** Approximate $f$ by the tangent line at the current guess, solve for where that line crosses zero, and move there. Repeat.

### 5.2 Applying to Maximize ℓ(θ)

Maxima of $\ell$ occur where $\ell'(\theta)=0$. Set $f(\theta)=\ell'(\theta)$:

$$
\theta := \theta - \frac{\ell'(\theta)}{\ell''(\theta)}
$$

### 5.3 Multidimensional Generalization (Newton-Raphson)

Since $\theta$ is vector-valued in logistic regression:

$$
\theta := \theta - H^{-1}\nabla_\theta \ell(\theta)
$$

- $\nabla_\theta \ell(\theta)$ → vector of partial derivatives
- $H$ → the **Hessian** matrix ($d{+}1 \times d{+}1$, including intercept), with entries:

$$
H_{ij} = \frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}
$$

When applied to logistic regression's log-likelihood, this method is also called **Fisher scoring**.

| Method | Iterations needed | Cost per iteration | Notes |
|---|---|---|---|
| (Batch) Gradient Descent | More | Cheap | Simple, scales well |
| Newton's Method | Fewer | Expensive (invert $d\times d$ Hessian) | Faster overall if $d$ is small |

---

## Key Takeaways

- **Logistic regression** models $P(y=1\mid x)$ with the sigmoid $h_\theta(x) = 1/(1+e^{-\theta^Tx})$, keeping predictions bounded in $[0,1]$.
- The sigmoid's derivative identity $g'(z) = g(z)(1-g(z))$ is the key trick that simplifies the log-likelihood gradient.
- Parameters are fit via **Maximum Likelihood**: maximize $\ell(\theta) = \sum_i y^{(i)}\log h(x^{(i)}) + (1-y^{(i)})\log(1-h(x^{(i)}))$.
- The resulting **stochastic gradient ascent** rule, $\theta_j := \theta_j + \alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}$, looks identical to linear regression's LMS rule — but the models are different (sigmoid vs. linear $h_\theta$). GLM theory later explains why.
- The **logistic loss** view ($\ell_{\text{logistic}}(t,y)$, with $t=\theta^Tx$ the "logit") is equivalent to negative log-likelihood and generalizes better to nonlinear models.
- The **perceptron** replaces the sigmoid with a hard threshold. Same-looking update rule, but no probabilistic/MLE interpretation — a fundamentally different algorithm.
- **Multi-class classification** ($y\in\{1,\dots,k\}$) uses the **softmax** function to turn $k$ logits into valid probabilities, and is trained by minimizing **cross-entropy loss**.
  - Gradient of cross-entropy w.r.t. logits: $\phi - e_y$ (predicted probability minus one-hot label) — the natural multi-class extension of the binary $(y-h_\theta(x))$ term.
- **Newton's method** (Newton–Raphson) maximizes $\ell(\theta)$ via $\theta := \theta - H^{-1}\nabla_\theta\ell(\theta)$, converging in fewer iterations than gradient descent but at higher per-iteration cost (Hessian inversion). Also called **Fisher scoring** in this context.
