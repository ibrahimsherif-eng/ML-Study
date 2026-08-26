# Generative Learning Algorithms

## 1. Discriminative vs. Generative Learning

So far, algorithms like logistic regression modeled $p(y\mid x;\theta)$ directly (e.g. $h_\theta(x)=g(\theta^Tx)$) — these are **discriminative** algorithms. They (or algorithms like the perceptron that map $x$ straight to a label) essentially learn a **decision boundary** separating the classes.

**Generative** algorithms instead model $p(x\mid y)$ (how each class generates its features) and $p(y)$ (the **class priors**), then use **Bayes' rule** to get $p(y\mid x)$:

$$
p(y\mid x) = \frac{p(x\mid y)\,p(y)}{p(x)}
$$

- $p(x) = p(x\mid y{=}1)p(y{=}1) + p(x\mid y{=}0)p(y{=}0)$
- For prediction we only need to compare classes, so the denominator can be dropped:

$$
\arg\max_y p(y\mid x) = \arg\max_y \; p(x\mid y)\,p(y)
$$

> 💡 **Intuition:** Instead of learning one line that separates elephants from dogs (discriminative), a generative approach builds a separate model of "what elephants look like" and "what dogs look like," then classifies a new animal by asking which model it resembles more.

```text
Discriminative:  learn p(y|x) directly, or x -> label directly
Generative:      learn p(x|y) and p(y),  then get p(y|x) via Bayes' rule
```

---

## 2. Gaussian Discriminant Analysis (GDA)

### 2.1 What is it?

GDA is a generative model for **continuous-valued** input features $x$. It assumes $p(x\mid y)$ follows a **multivariate normal distribution**.

### 2.2 The Multivariate Normal Distribution

Parameterized by mean vector $\mu \in \mathbb{R}^d$ and covariance matrix $\Sigma \in \mathbb{R}^{d\times d}$ (symmetric, positive semi-definite). Written $N(\mu,\Sigma)$:

$$
p(x;\mu,\Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)
$$

- $|\Sigma|$ → determinant of $\Sigma$
- $E[X] = \mu$
- $\text{Cov}(Z) = E[(Z-E[Z])(Z-E[Z])^T] = E[ZZ^T]-E[Z]E[Z]^T$; for $X\sim N(\mu,\Sigma)$, $\text{Cov}(X)=\Sigma$

> 💡 **Intuition:** $\mu$ shifts the distribution's center; $\Sigma$ controls its spread and orientation. A larger $\Sigma$ (e.g. $2I$) spreads the density out; a smaller one (e.g. $0.6I$) compresses it. Off-diagonal entries in $\Sigma$ tilt/compress the contours along diagonal directions, turning circular contours into ellipses.

### 2.3 The GDA Model

For a binary classification problem with continuous $x$:

$$
y \sim \text{Bernoulli}(\phi), \qquad x\mid y{=}0 \sim N(\mu_0,\Sigma), \qquad x\mid y{=}1 \sim N(\mu_1,\Sigma)
$$

$$
p(y) = \phi^y(1-\phi)^{1-y}
$$
$$
p(x\mid y{=}0) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0)\right)
$$
$$
p(x\mid y{=}1) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)\right)
$$

**Parameters:** $\phi, \Sigma, \mu_0, \mu_1$.

> ⚠️ **Important:** Although there are two mean vectors $\mu_0,\mu_1$ (one per class), GDA is normally applied with a **single shared covariance matrix** $\Sigma$ across both classes.

### 2.4 Maximum Likelihood Estimation

Log-likelihood of the data:

$$
\ell(\phi,\mu_0,\mu_1,\Sigma) = \log\prod_{i=1}^n p(x^{(i)}\mid y^{(i)};\mu_0,\mu_1,\Sigma)\,p(y^{(i)};\phi)
$$

Maximizing gives the MLE estimates:

$$
\phi = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{y^{(i)}=1\}
$$
$$
\mu_0 = \frac{\sum_{i=1}^n \mathbb{1}\{y^{(i)}=0\}\,x^{(i)}}{\sum_{i=1}^n \mathbb{1}\{y^{(i)}=0\}}, \qquad
\mu_1 = \frac{\sum_{i=1}^n \mathbb{1}\{y^{(i)}=1\}\,x^{(i)}}{\sum_{i=1}^n \mathbb{1}\{y^{(i)}=1\}}
$$
$$
\Sigma = \frac{1}{n}\sum_{i=1}^n (x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T
$$

- $\phi$ → fraction of positive examples
- $\mu_0,\mu_1$ → per-class feature means (average feature vector within each class)
- $\Sigma$ → shared covariance, averaged over both classes' deviations from their own class mean

### 2.5 What GDA Looks Like

The fitted model produces two Gaussian contours (one per class) with the **same shape and orientation** (shared $\Sigma$) but **different centers** ($\mu_0,\mu_1$). The decision boundary — where $p(y{=}1\mid x)=0.5$ — is the straight line separating the two contour sets.

```text
Class 0 data → fit N(mu_0, Sigma)
Class 1 data → fit N(mu_1, Sigma)     (same Sigma for both)
        ↓
Decision boundary: straight line where p(y=1|x) = 0.5
```

---

### 2.6 GDA vs. Logistic Regression

If we express $p(y{=}1\mid x;\phi,\Sigma,\mu_0,\mu_1)$ as a function of $x$, it simplifies to:

$$
p(y{=}1\mid x;\phi,\Sigma,\mu_0,\mu_1) = \frac{1}{1+\exp(-\theta^Tx)}
$$

for some $\theta$ that is a function of $\phi,\Sigma,\mu_0,\mu_1$ — **exactly the logistic regression form**.

> ⚠️ **Important:** The implication only goes one way. If $p(x\mid y)$ is multivariate Gaussian with shared $\Sigma$, then $p(y\mid x)$ is necessarily logistic. The **converse is false**: $p(y\mid x)$ being logistic does **not** imply $p(x\mid y)$ is Gaussian. So GDA makes **stronger** assumptions than logistic regression.

**When each is better:**

| | GDA | Logistic Regression |
|---|---|---|
| Assumptions | Stronger ($x\mid y$ Gaussian, shared $\Sigma$) | Weaker (only assumes the logistic form for $p(y\mid x)$) |
| When assumptions hold | Asymptotically efficient — no algorithm beats it for large $n$ | Still works, but not optimal |
| When assumptions are wrong | May or may not do well (unpredictable) | Still tends to work well (e.g. on Poisson-distributed $x\mid y$, which is also logistic) |
| Data efficiency | More data-efficient when roughly correct | Needs more data but is more robust |
| Robustness | Less robust to incorrect modeling assumptions | More robust to deviations from assumptions |
| Practical usage | Less commonly used | Used more often in practice |

> 💡 **Intuition:** GDA is a stronger bet — it pays off big if you're right about the Gaussian assumption but can fail unpredictably if you're wrong. Logistic regression is a safer, more general-purpose bet.

---

## 3. Naive Bayes (Optional Reading)

### 3.1 What is it?

Naive Bayes is a generative algorithm for **discrete-valued** features $x_j$. Motivating example: spam classification, where an email is represented as a binary vector over a **vocabulary** — $x_j=1$ if word $j$ of the dictionary appears in the email, else $0$.

> ⚠️ **Important:** With a vocabulary of 50,000 words, $x\in\{0,1\}^{50000}$. Modeling $p(x\mid y)$ as a full multinomial over all $2^{50000}$ outcomes would need $2^{50000}-1$ parameters — completely infeasible.

### 3.2 The Naive Bayes (NB) Assumption

To make this tractable, assume the $x_j$'s are **conditionally independent given $y$**:

$$
p(x_1,\dots,x_{50000}\mid y) = \prod_{j=1}^d p(x_j\mid y)
$$

> 💡 **Intuition:** Knowing an email is spam ($y{=}1$), knowing whether "buy" appears tells you *nothing extra* about whether "price" appears — even though "buy" and "price" are clearly *not* unconditionally independent (spam emails tend to contain both together). This is a strong, often technically false, assumption — but it works well in practice.

### 3.3 Parameters and MLE

Parameters: $\phi_{j\mid y=1}=p(x_j{=}1\mid y{=}1)$, $\phi_{j\mid y=0}=p(x_j{=}1\mid y{=}0)$, $\phi_y=p(y{=}1)$.

Maximum likelihood estimates:

$$
\phi_{j\mid y=1} = \frac{\sum_{i=1}^n \mathbb{1}\{x_j^{(i)}=1 \wedge y^{(i)}=1\}}{\sum_{i=1}^n \mathbb{1}\{y^{(i)}=1\}}, \qquad
\phi_{j\mid y=0} = \frac{\sum_{i=1}^n \mathbb{1}\{x_j^{(i)}=1 \wedge y^{(i)}=0\}}{\sum_{i=1}^n \mathbb{1}\{y^{(i)}=0\}}
$$
$$
\phi_y = \frac{\sum_{i=1}^n \mathbb{1}\{y^{(i)}=1\}}{n}
$$

$\phi_{j\mid y=1}$ is simply the fraction of spam emails containing word $j$.

### 3.4 Prediction

$$
p(y{=}1\mid x) = \frac{\left(\prod_{j=1}^d p(x_j\mid y{=}1)\right)p(y{=}1)}{\left(\prod_{j=1}^d p(x_j\mid y{=}1)\right)p(y{=}1) + \left(\prod_{j=1}^d p(x_j\mid y{=}0)\right)p(y{=}0)}
$$

Pick whichever class has the higher posterior probability.

### 3.5 Beyond Binary Features

Naive Bayes generalizes easily to $x_j \in \{1,\dots,k_j\}$ by modeling $p(x_j\mid y)$ as **multinomial** instead of Bernoulli. Continuous features not well modeled by a Gaussian (unlike GDA's assumption) can be **discretized** into bins and handled the same way.

**Example:** discretizing living area into buckets:

| Living area (sq ft) | <400 | 400–800 | 800–1200 | 1200–1600 | >1600 |
|---|---|---|---|---|---|
| $x_j$ | 1 | 2 | 3 | 4 | 5 |

A house with 890 sq ft → $x_j=3$.

> 💡 **Intuition:** When continuous features clearly don't follow a Gaussian, discretizing and using Naive Bayes can outperform forcing a (possibly wrong) Gaussian assumption in GDA.

---

## 4. Laplace Smoothing

### 4.1 The Problem: Zero Probabilities

If a word (e.g. "neurips") never appears in the training set at all, its MLE estimate is $\phi_{j\mid y=1}=\phi_{j\mid y=0}=0$. Then for any new email containing that word:

$$
p(y{=}1\mid x) = \frac{0}{0}
$$

— undefined, since every class-conditional product is multiplied by this zero term.

> ⚠️ **Important:** It's statistically unsound to estimate a probability as exactly zero just because an event never appeared in a *finite* training set.

### 4.2 The Fix

For a multinomial random variable $z \in \{1,\dots,k\}$ with MLE $\phi_j = \frac{1}{n}\sum_{i=1}^n\mathbb{1}\{z^{(i)}=j\}$, **Laplace smoothing** replaces it with:

$$
\phi_j = \frac{1+\sum_{i=1}^n \mathbb{1}\{z^{(i)}=j\}}{k+n}
$$

- Adds 1 to the numerator, $k$ to the denominator.
- Still satisfies $\sum_{j=1}^k \phi_j = 1$.
- Guarantees $\phi_j > 0$ for every $j$, eliminating the zero-probability problem.
- Under fairly strong conditions, this can be shown to be the *optimal* estimator.

Applied to Naive Bayes:

$$
\phi_{j\mid y=1} = \frac{1+\sum_{i=1}^n \mathbb{1}\{x_j^{(i)}=1 \wedge y^{(i)}=1\}}{2+\sum_{i=1}^n \mathbb{1}\{y^{(i)}=1\}}, \qquad
\phi_{j\mid y=0} = \frac{1+\sum_{i=1}^n \mathbb{1}\{x_j^{(i)}=1 \wedge y^{(i)}=0\}}{2+\sum_{i=1}^n \mathbb{1}\{y^{(i)}=0\}}
$$

> ⚠️ **Important:** In practice, smoothing $\phi_y$ usually doesn't matter much, since we typically have a healthy fraction of both classes, so its MLE is already far from 0.

---

## 5. Event Models for Text Classification

Two ways to generate an email under Naive Bayes, differing in **how words are represented**:

| | Bernoulli Event Model | Multinomial Event Model |
|---|---|---|
| Feature representation | $x_j=1$ if word $j$ of the vocabulary appears (fixed-length vector = vocab size) | $x_j$ = identity of the $j$-th word in the email (variable length = email length $d$) |
| Distribution of $x_j\mid y$ | Bernoulli | Multinomial over the vocabulary |
| Generation story | Decide independently, for each vocab word, whether to include it | Generate each word position independently from the same multinomial over the vocabulary |
| Message probability | $p(y)\prod_{j=1}^{d} p(x_j\mid y)$ (product over *vocabulary* words) | $p(y)\prod_{j=1}^{d} p(x_j\mid y)$ (product over *word positions* in the email) |

> ⚠️ **Important:** The two formulas look similar but mean different things — in the Bernoulli model, $x_j\mid y$ is Bernoulli and $d$ = vocabulary size; in the multinomial model, $x_j\mid y$ is multinomial and $d$ = number of words in that particular email (varies per document).

**Multinomial event model parameters:** $\phi_y=p(y)$, $\phi_{k\mid y=1}=p(x_j{=}k\mid y{=}1)$, $\phi_{k\mid y=0}=p(x_j{=}k\mid y{=}0)$ (assumed the same for every position $j$).

MLE estimates (with Laplace smoothing, needed in practice for good performance — note the denominator now adds $|V|$, the vocabulary size, instead of 2):

$$
\phi_{k\mid y=1} = \frac{1+\sum_{i=1}^n\sum_{j=1}^{d_i} \mathbb{1}\{x_j^{(i)}=k \wedge y^{(i)}=1\}}{|V|+\sum_{i=1}^n \mathbb{1}\{y^{(i)}=1\}\,d_i}
$$
$$
\phi_{k\mid y=0} = \frac{1+\sum_{i=1}^n\sum_{j=1}^{d_i} \mathbb{1}\{x_j^{(i)}=k \wedge y^{(i)}=0\}}{|V|+\sum_{i=1}^n \mathbb{1}\{y^{(i)}=0\}\,d_i}
$$

The multinomial event model tends to perform even better than the Bernoulli event model for text classification specifically.

---

## Key Takeaways

- **Discriminative** algorithms (logistic regression) learn $p(y\mid x)$ or a direct $x\to y$ mapping. **Generative** algorithms learn $p(x\mid y)$ and $p(y)$, then combine via Bayes' rule: $p(y\mid x)\propto p(x\mid y)p(y)$.
- **GDA** assumes continuous $x\mid y \sim N(\mu_0 \text{ or }\mu_1, \Sigma)$ with a **shared** covariance matrix, and $y\sim\text{Bernoulli}(\phi)$. MLE gives simple closed-form estimates: $\phi$ = fraction positive, $\mu_0,\mu_1$ = per-class feature means, $\Sigma$ = averaged within-class scatter.
- **GDA implies logistic regression's form** for $p(y\mid x)$, but not vice versa — GDA is a *stronger* assumption. When the Gaussian/shared-$\Sigma$ assumption is (approximately) correct, GDA is more data-efficient and asymptotically optimal; when it's wrong, logistic regression's weaker assumptions make it more robust. This is why logistic regression is used more often in practice.
- **Naive Bayes** assumes features $x_j$ are conditionally independent given $y$ — a strong but often effective simplification that avoids exponential blowup in parameters for high-dimensional discrete data (like word-presence vectors).
- Naive Bayes generalizes from binary $x_j$ to multi-valued $x_j\in\{1,\dots,k_j\}$ (multinomial $p(x_j\mid y)$), and continuous features can be handled by **discretizing** them into bins.
- **Laplace smoothing** ($\phi_j = \frac{1+\text{count}}{k+n}$) fixes the zero-probability problem caused by unseen feature values in a finite training set, and is essential in practice, especially for text classification.
- Two **event models** for text: the **Bernoulli event model** (word presence/absence, fixed-length vocabulary vector) and the **Multinomial event model** (sequence of word identities, variable email length) — the multinomial version generally performs better for text classification.
