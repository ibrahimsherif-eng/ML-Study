# CS229 — Support Vector Machines

## 1. Margins: Intuition

### 1.1 What is it?
SVMs are built around the idea of separating classes with the largest possible **gap** (margin). Before formalizing this, we build intuition using logistic regression.

### 1.2 Intuition
In logistic regression, $h_\theta(x) = g(\theta^Tx)$ predicts $y=1$ iff $\theta^Tx \geq 0$. The larger $\theta^Tx$ is (for a positive example), the more **confident** the prediction.

> 💡 **Intuition:** A point far from the decision boundary (like point A) can be classified with high confidence. A point close to the boundary (like point C) could flip its predicted class from a tiny shift in the boundary — so we trust that prediction less.

This gives a goal: find parameters that make correct predictions **confidently**, i.e., points should lie far from the decision boundary. This idea is later formalized as the **geometric margin**.

---

## 2. Notation

To simplify the SVM derivation, notation changes from logistic regression:

| Item | Logistic Regression | SVM |
|---|---|---|
| Labels | $y \in \{0,1\}$ | $y \in \{-1,1\}$ |
| Parameters | $\theta$ (with $x_0=1$ trick) | $w, b$ separate |
| Classifier | $h_\theta(x)=g(\theta^Tx)$ | $h_{w,b}(x)=g(w^Tx+b)$ |
| $g(z)$ | Sigmoid, outputs probability | $g(z)=1$ if $z\geq0$, else $-1$ (hard decision) |

- $w$ plays the role of $[\theta_1 \dots \theta_d]^T$, $b$ plays the role of $\theta_0$.
- Unlike logistic regression, the SVM classifier directly outputs $\{-1,1\}$ — it does not estimate $p(y=1)$ first.

---

## 3. Functional and Geometric Margins

### 3.1 Functional Margin

$$
\hat\gamma^{(i)} = y^{(i)}(w^Tx^{(i)} + b)
$$

- $\hat\gamma^{(i)}$ → confidence/correctness measure for example $i$.
- If $y^{(i)}(w^Tx^{(i)}+b) > 0$, the prediction is **correct**.
- A **large** functional margin = confident **and** correct prediction.

> ⚠️ **Important:** The functional margin is *not* a reliable confidence measure by itself. Since $g$ only depends on the **sign** of $w^Tx+b$, scaling $(w,b) \to (2w,2b)$ doesn't change the classifier at all, but it doubles the functional margin. So the functional margin can be made arbitrarily large without any real improvement — a normalization (e.g. $\|w\|=1$) is needed to fix this.

For a training set, the functional margin is the **worst case** over all examples:

$$
\hat\gamma = \min_{i=1,\dots,n} \hat\gamma^{(i)}
$$

### 3.2 Geometric Margin

The geometric margin is the actual **Euclidean distance** from a point to the decision boundary.

Derivation idea: $w$ is orthogonal to the decision boundary. Moving from point $x^{(i)}$ toward the boundary by distance $\gamma^{(i)}$ along the unit vector $w/\|w\|$ lands exactly on the boundary (where $w^Tx+b=0$). Solving this equation gives:

$$
\gamma^{(i)} = y^{(i)}\left(\left(\frac{w}{\|w\|}\right)^Tx^{(i)} + \frac{b}{\|w\|}\right)
$$

- $\gamma^{(i)}$ → true distance from example $i$ to the separating hyperplane.
- The $y^{(i)}$ factor makes the margin positive for correct predictions, negative for wrong ones.

**Relationship between the two margins:**
$$
\gamma^{(i)} = \frac{\hat\gamma^{(i)}}{\|w\|}
$$

- If $\|w\|=1$, functional margin = geometric margin.
- Geometric margin is **invariant to rescaling** of $(w,b)$ — this is the key property that lets us impose arbitrary scaling constraints later without losing anything.

For a training set, the geometric margin is again the worst case:
$$
\gamma = \min_{i=1,\dots,n} \gamma^{(i)}
$$

| Margin Type | Formula | Scale-invariant? | Use |
|---|---|---|---|
| Functional | $y^{(i)}(w^Tx^{(i)}+b)$ | No | Raw confidence measure |
| Geometric | $y^{(i)}\left(\frac{w}{\|w\|}\right)^Tx^{(i)} + \frac{b}{\|w\|}$ | Yes | True distance; used for optimization |

---

## 4. The Optimal Margin Classifier

### 4.1 What is it?
Given a linearly separable training set, find the decision boundary that **maximizes the geometric margin** — the biggest possible "gap" between classes.

### 4.2 From Idea to Solvable Problem

**Step 1 — Natural formulation (non-convex, unusable directly):**
$$
\max_{\gamma,w,b} \gamma \quad \text{s.t.} \quad y^{(i)}(w^Tx^{(i)}+b)\geq \gamma,\ \ \|w\|=1
$$
The constraint $\|w\|=1$ is non-convex, so standard solvers can't handle it.

**Step 2 — Drop the norm constraint, use the margin relationship:**
$$
\max_{\hat\gamma,w,b} \frac{\hat\gamma}{\|w\|} \quad \text{s.t.} \quad y^{(i)}(w^Tx^{(i)}+b)\geq \hat\gamma
$$
Now the objective $\hat\gamma/\|w\|$ is itself non-convex — still not solvable directly.

**Step 3 — Exploit scale invariance:** Since $(w,b)$ can be rescaled freely without changing the classifier, fix the scale by setting $\hat\gamma = 1$. Then maximizing $1/\|w\|$ is the same as **minimizing $\|w\|^2$**:

$$
\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y^{(i)}(w^Tx^{(i)}+b)\geq 1,\ i=1,\dots,n
$$

This is a **convex quadratic objective with linear constraints** — solvable with standard Quadratic Programming (QP) software. Its solution is the **optimal margin classifier**.

> 💡 **Intuition:** Minimizing $\|w\|^2$ subject to every example having functional margin ≥ 1 is exactly equivalent to maximizing the geometric margin — we just re-parameterized the same problem into a convex, solvable form.

To go further (and eventually use kernels), we derive the **dual form** via Lagrange duality.

---

## 5. Lagrange Duality

### 5.1 What is it?
A general framework for solving constrained optimization problems, especially useful when constraints include inequalities.

### 5.2 Equality-Constrained Case (Lagrange Multipliers)
$$
\min_w f(w) \quad \text{s.t.} \quad h_i(w)=0,\ i=1,\dots,l
$$
Define the Lagrangian:
$$
L(w,\beta) = f(w) + \sum_{i=1}^l \beta_i h_i(w)
$$
Solve by setting $\partial L/\partial w_i = 0$ and $\partial L/\partial \beta_i = 0$.

### 5.3 General Case (Inequality + Equality Constraints)

**Primal problem:**
$$
\min_w f(w) \quad \text{s.t.} \quad g_i(w)\leq 0\ (i=1,\dots,k), \quad h_i(w)=0\ (i=1,\dots,l)
$$

**Generalized Lagrangian:**
$$
L(w,\alpha,\beta) = f(w) + \sum_{i=1}^k \alpha_i g_i(w) + \sum_{i=1}^l \beta_i h_i(w)
$$

Define $\theta_P(w) = \max_{\alpha,\beta:\alpha_i\geq0} L(w,\alpha,\beta)$.

- If $w$ **violates** a constraint → $\theta_P(w) = \infty$ (the max blows up).
- If $w$ **satisfies** all constraints → $\theta_P(w) = f(w)$.

So minimizing $\theta_P(w)$ over $w$ is equivalent to the original primal problem. Its optimal value is $p^* = \min_w \theta_P(w)$.

**Dual problem** — swap the order of min and max:
$$
\theta_D(\alpha,\beta) = \min_w L(w,\alpha,\beta), \qquad d^* = \max_{\alpha,\beta:\alpha_i\geq0}\theta_D(\alpha,\beta)
$$

**Weak duality (always true):**
$$
d^* \leq p^*
$$
(max-min ≤ min-max in general.)

### 5.4 Strong Duality and KKT Conditions

> ⚠️ **Important:** $d^*=p^*$ only holds under specific conditions: $f$ and all $g_i$ convex, all $h_i$ affine, and the constraints are strictly feasible (some $w$ exists with $g_i(w)<0$ for all $i$).

Under these conditions, there exist $w^*, \alpha^*, \beta^*$ solving both problems simultaneously, and they satisfy the **Karush-Kuhn-Tucker (KKT) conditions**:

$$
\frac{\partial}{\partial w_i}L(w^*,\alpha^*,\beta^*)=0,\quad \frac{\partial}{\partial \beta_i}L(w^*,\alpha^*,\beta^*)=0
$$
$$
\alpha_i^* g_i(w^*) = 0 \quad \text{(dual complementarity)}
$$
$$
g_i(w^*)\leq 0,\qquad \alpha_i^*\geq 0
$$

- Conversely, any $w^*,\alpha^*,\beta^*$ satisfying KKT **is** a solution to both problems.
- The **dual complementarity condition** ($\alpha_i^*g_i(w^*)=0$) is key: if $\alpha_i^*>0$, then $g_i(w^*)=0$ (the constraint is active/tight). This is what later identifies **support vectors** and gives the SMO convergence test.

---

## 6. Optimal Margin Classifier — Dual Form

> ⚠️ **Important — key takeaway of this section:** The equivalence between the primal problem (Eq. 6.8) and the dual problem (Eq. 6.12), plus the relation $w=\sum_i \alpha_i y^{(i)}x^{(i)}$, are the most important results here.

### 6.1 Setup
Rewrite the primal constraint as:
$$
g_i(w) = -y^{(i)}(w^Tx^{(i)}+b) + 1 \leq 0
$$

By KKT dual complementarity, $\alpha_i > 0$ **only** for training examples whose functional margin is exactly 1 (i.e., points that lie exactly on the margin boundary). These points are called **support vectors** — typically a small subset of the training set.

### 6.2 Building the Lagrangian
$$
L(w,b,\alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^n \alpha_i\big[y^{(i)}(w^Tx^{(i)}+b)-1\big]
$$
(Only $\alpha_i$ multipliers — no $\beta_i$, since there are no equality constraints.)

### 6.3 Deriving the Dual
Minimize $L$ over $w,b$ by setting gradients to zero:

$$
\nabla_w L = w - \sum_{i=1}^n \alpha_i y^{(i)}x^{(i)} = 0 \quad\Rightarrow\quad w = \sum_{i=1}^n \alpha_i y^{(i)}x^{(i)} \tag{key relation}
$$
$$
\frac{\partial L}{\partial b} = \sum_{i=1}^n \alpha_i y^{(i)} = 0
$$

Substituting $w$ back into $L$ and simplifying (the $b$ term vanishes because of the second equation) gives the **dual optimization problem**:

$$
\max_\alpha W(\alpha) = \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n y^{(i)}y^{(j)}\alpha_i\alpha_j \langle x^{(i)}, x^{(j)}\rangle
$$
$$
\text{s.t.}\quad \alpha_i \geq 0,\ i=1,\dots,n \qquad \sum_{i=1}^n \alpha_i y^{(i)} = 0
$$

- $\langle x^{(i)}, x^{(j)}\rangle$ → inner product between training points. **This is the crucial detail**: the entire algorithm is expressed only in terms of inner products between input vectors — this property enables the **kernel trick**.

### 6.4 Recovering the Classifier
Once $\alpha^*$ is found:
- $w^*$ is recovered via $w=\sum_i \alpha_i y^{(i)}x^{(i)}$.
- The intercept is recovered as:
$$
b^* = -\frac{\max_{i:y^{(i)}=-1} w^{*T}x^{(i)} + \min_{i:y^{(i)}=1} w^{*T}x^{(i)}}{2}
$$

**Prediction on a new point $x$:**
$$
w^Tx+b = \sum_{i=1}^n \alpha_i y^{(i)} \langle x^{(i)}, x\rangle + b
$$

> 💡 **Intuition:** Since $\alpha_i=0$ for all non-support-vectors, this sum only needs the inner products between $x$ and the (few) support vectors — making prediction efficient even when the number of training points is huge.

```mermaid
flowchart LR
    A[Primal: minimize 1/2||w||^2] --> B[Lagrangian L w,b,alpha]
    B --> C[Set gradients wrt w,b to zero]
    C --> D[Dual: maximize Wa in terms of inner products]
    D --> E[Solve for alpha]
    E --> F[Recover w*, b*]
    F --> G[Predict using inner products with support vectors]
```

---

## 7. Regularization and the Non-Separable Case

### 7.1 Why Needed
Real data is often **not linearly separable**, and even when it is, a hard-margin classifier can be very sensitive to outliers — a single mislabeled/extreme point can drastically shrink the margin or make separation impossible.

### 7.2 Soft-Margin Formulation ($\ell_1$ regularization)

$$
\min_{\gamma,w,b} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i
$$
$$
\text{s.t.}\quad y^{(i)}(w^Tx^{(i)}+b)\geq 1-\xi_i,\ \ \xi_i \geq 0,\ i=1,\dots,n
$$

- $\xi_i$ → "slack" allowing example $i$ to have functional margin less than 1 (or even be misclassified), at a cost of $C\xi_i$ added to the objective.
- $C$ → trade-off parameter between maximizing the margin (small $\|w\|^2$) and keeping most examples correctly classified with margin ≥ 1.

### 7.3 Dual Form with Slack
Following the same Lagrangian procedure (with extra multipliers $r_i\geq0$ for $\xi_i\geq0$) leads to a dual problem **almost identical** to the separable case:

$$
\max_\alpha W(\alpha) = \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n y^{(i)}y^{(j)}\alpha_i\alpha_j\langle x^{(i)},x^{(j)}\rangle
$$
$$
\text{s.t.}\quad 0\leq \alpha_i \leq C,\ i=1,\dots,n \qquad \sum_{i=1}^n \alpha_i y^{(i)}=0
$$

> ⚠️ **Important:** The *only* change from the separable case is that the constraint becomes $0\leq\alpha_i\leq C$ (previously just $\alpha_i\geq0$). Equation (6.10) for $w$ still holds, so predictions still use Eq. (6.15). The formula for $b^*$ from the separable case (6.13) is **no longer valid** here.

**KKT dual complementarity conditions (soft margin):**
$$
\alpha_i = 0 \Rightarrow y^{(i)}(w^Tx^{(i)}+b) \geq 1
$$
$$
\alpha_i = C \Rightarrow y^{(i)}(w^Tx^{(i)}+b) \leq 1
$$
$$
0 < \alpha_i < C \Rightarrow y^{(i)}(w^Tx^{(i)}+b) = 1
$$

These conditions are used later as the **convergence test** for the SMO algorithm.

---

## 8. The SMO Algorithm

### 8.1 What is it?
**Sequential Minimal Optimization (SMO)**, by John Platt, is an efficient algorithm for solving the SVM dual problem.

### 8.2 Building Block: Coordinate Ascent
For an unconstrained problem $\max_\alpha W(\alpha_1,\dots,\alpha_n)$:

```text
Loop until convergence:
    For i = 1 to n:
        alpha_i := argmax over alpha_i of W(alpha_1,...,alpha_i,...,alpha_n)
                    (holding all other alpha_j fixed)
```

> 💡 **Intuition:** Optimize one variable at a time while freezing the rest — each step moves parallel to one coordinate axis. This is efficient whenever the inner "argmax" step can be solved in closed form.

### 8.3 Why Plain Coordinate Ascent Fails for SVM
The dual has the constraint $\sum_{i=1}^n \alpha_i y^{(i)} = 0$. If we tried to update only $\alpha_1$ while holding $\alpha_2,\dots,\alpha_n$ fixed:
$$
\alpha_1 = -y^{(1)}\sum_{i=2}^n \alpha_i y^{(i)}
$$
$\alpha_1$ is **completely determined** by the others — there's no freedom left to optimize it alone.

> ⚠️ **Important:** Because of the linear equality constraint, at least **two** $\alpha_i$'s must be updated simultaneously to preserve feasibility.

### 8.4 The SMO Algorithm
```text
Repeat until convergence:
    1. Select a pair (alpha_i, alpha_j) to update
       (heuristic: pick the pair expected to give the most progress).
    2. Re-optimize W(alpha) with respect to alpha_i, alpha_j only,
       holding all other alpha_k fixed.
```

**Convergence test:** check whether the KKT conditions (7.3's Eqs.) are satisfied within a tolerance `tol` (typically 0.01–0.001).

### 8.5 Deriving the Efficient Two-Variable Update
Fix $\alpha_3,\dots,\alpha_n$; update $\alpha_1,\alpha_2$. The equality constraint becomes:
$$
\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = \zeta \quad \text{(constant, since the rest are fixed)}
$$

This confines $(\alpha_1,\alpha_2)$ to a line segment inside the box $[0,C]\times[0,C]$. This gives bounds $L \leq \alpha_2 \leq H$.

Express $\alpha_1$ in terms of $\alpha_2$:
$$
\alpha_1 = (\zeta - \alpha_2 y^{(2)}) y^{(1)}
$$

Substituting into $W(\alpha)$ gives a **quadratic function of $\alpha_2$ alone** (of the form $a\alpha_2^2+b\alpha_2+c$), which can be maximized in closed form by setting its derivative to zero — call this unconstrained optimum $\alpha_2^{\text{new,unclipped}}$.

**Clip to satisfy the box constraint:**
$$
\alpha_2^{\text{new}} =
\begin{cases}
H & \text{if } \alpha_2^{\text{new,unclipped}} > H \\
\alpha_2^{\text{new,unclipped}} & \text{if } L \leq \alpha_2^{\text{new,unclipped}} \leq H \\
L & \text{if } \alpha_2^{\text{new,unclipped}} < L
\end{cases}
$$

Finally, recover $\alpha_1^{\text{new}}$ using the linear relationship $\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = \zeta$.

> 💡 **Intuition:** Because updating two variables under a linear constraint reduces to a 1-D quadratic optimization (in $\alpha_2$) with a closed-form solution, each SMO update step is very cheap — this is what makes SMO efficient compared to generic QP solvers.

---

# Key Takeaways

**Big picture flow:**
```text
Margins (functional & geometric)
        ↓
Optimal margin classifier (primal QP: min 1/2||w||^2)
        ↓
Lagrange duality (KKT conditions)
        ↓
Dual form (expressed via inner products ⟨x(i),x(j)⟩)
        ↓
Kernels (implicit in the inner-product form; enables high-dim features)
        ↓
Soft margin (regularization for non-separable data / outliers)
        ↓
SMO algorithm (efficiently solves the dual)
```

- **Functional margin** $\hat\gamma^{(i)}=y^{(i)}(w^Tx^{(i)}+b)$: confidence measure, but scale-dependent (can be inflated by scaling $w,b$).
- **Geometric margin** $\gamma^{(i)}=\hat\gamma^{(i)}/\|w\|$: true distance to boundary, scale-invariant — the quantity we actually want to maximize.
- **Optimal margin classifier**: reformulating "maximize geometric margin" as $\min \frac12\|w\|^2$ s.t. $y^{(i)}(w^Tx^{(i)}+b)\geq1$ turns a non-convex problem into a solvable convex QP.
- **Lagrange duality**: for convex problems with affine equality constraints and feasible inequality constraints, $p^*=d^*$, and KKT conditions characterize the solution. **Dual complementarity** ($\alpha_i^*g_i(w^*)=0$) is central to identifying support vectors.
- **Support vectors**: training points with $\alpha_i>0$; these are the points exactly on the margin boundary. Only a few points are usually support vectors — this makes prediction efficient.
- **Dual form of SVM**: both training ($W(\alpha)$) and prediction ($w^Tx+b$) depend on data **only through inner products** $\langle x^{(i)},x^{(j)}\rangle$ — the foundation for the kernel trick.
- **Soft margin (regularized) SVM**: introduces slack variables $\xi_i$ and parameter $C$ to allow margin violations, handling non-separable data and reducing outlier sensitivity. Dual form is identical except $\alpha_i$ is now bounded: $0\leq\alpha_i\leq C$.
- **SMO algorithm**: solves the dual efficiently by repeatedly optimizing two $\alpha_i$'s at a time (minimum needed to respect the equality constraint $\sum\alpha_iy^{(i)}=0$), with each pairwise update solvable in closed form via a 1-D quadratic optimization plus clipping to $[L,H]$.
