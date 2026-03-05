# Implementing Linear Regression from Scratch: A Deep Dive into Gradient Descent

* Published: March 5, 2026

## Introduction

I was working my way through Andrew Ng's Machine Learning course when the gradient descent update rule appeared on the screen. The formula told me to transpose my feature matrix and multiply it by the error vector. Most tutorials just say "trust the math," hand-wave the matrix dimensions, and move on to importing `scikit-learn`.

I couldn't do that. I needed to know *why* that transpose was there. So, I grabbed a marker, walked over to my whiteboard, and derived the matrix calculus myself. That one afternoon of manual derivation completely changed how I understand machine learning. I didn't just want to use a black-box model; I wanted to build it from the ground up using nothing but NumPy. Here is what I learned.

---

## 1. The Math: Fitting the Line

Linear Regression aims to fit a straight line through data. We define a **Hypothesis ($h_\theta$)** to represent our prediction:

$$h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n$$

To find the best parameters ($\theta$), we need a way to measure our "wrongness." We use the **Cost Function** (Mean Squared Error):

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

### The "Aha!" Moment: Why $X^T$?

When I first saw the gradient update code `(1/m) * X.T @ err`, I initially just accepted the transpose as a shape requirement. Then I worked through the matrix calculus on my whiteboard and realized it's not a trick—it's exactly what the chain rule produces.

To minimize the cost, we use **Gradient Descent**. Imagine standing on a mountain in thick fog; you feel the slope beneath your feet and take a step in the steepest downward direction. We update our parameters simultaneously until we reach the bottom:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

Applying the chain rule to the cost function results in $X^T(X\theta - y)$. The transpose ($X^T$) is mathematically necessary because it aligns each feature with its corresponding error across all $m$ training examples, giving us a perfectly shaped gradient vector.

---

## 2. My Implementation

By moving from scalar math to **Vectorization**, I replaced slow Python loops with high-performance NumPy matrix operations.

### Computing Cost

This function calculates the Mean Squared Error. We square the errors to mathematically penalize larger outliers more heavily than small ones.

```python
    def _compute_cost(self, X, y, theta):
        """Compute the cost function for linear regression""" 
        m = len(y) 
        predictions = X @ theta
        err = predictions - y
        
        # Calculate MSE. Note the 1/(2m) to make the derivative cleaner later.
        cost = (1 / (2 * m)) * np.sum(err ** 2)
        return cost

```

### Gradient Descent Loop

This is the engine of the algorithm. Instead of looping through features, the `@` operator handles the dot product instantly, updating all weights simultaneously.

```python
    def _gradient_descent(self, X, y):
        """Perform Gradient descent to update theta"""
        m, n = X.shape
        theta = np.zeros(n)

        for i in range(self.num_iterations):
            prediction = X @ theta
            err = prediction - y
            
            # The Gradient: Where the X-transpose magic happens
            gradient = (1/m) * X.T @ err
            
            # Update theta
            theta = theta - self.alpha * gradient

            cost = self._compute_cost(X, y, theta)
            self.cost_history.append(cost)

        return theta

```

---

## 3. Testing on Real Data: California Housing

The model was trained on the California Housing dataset using a custom gradient descent implementation. After 1000 iterations the cost converged and the model was benchmarked against scikit-learn.

### Model Performance

| Metric | Value | Interpretation |
| --- | --- | --- |
| **Training $R^2$** | 0.6126 | Model explains 61% of variance in training data

| **Testing $R^2$** | 0.5757 | Model explains 58% of variance on unseen data


| **sklearn $R^2$** | 0.5758 | Difference of 0.000082 — essentially identical



The tiny gap between our model and sklearn confirms the gradient descent implementation is mathematically correct.

### The Divergence Trap

During experimentation, I discovered that choosing a learning rate ($\alpha$) is a delicate balancing act. When I set $\alpha = 0.99$, the algorithm failed completely. The cost function skyrocketed from 0.33 to 6.49 in just 900 iterations, resulting in a disastrous $R^2$ of -29.24. The steps were so large that the model kept "overshooting" the minimum, climbing the opposite side of the cost valley until it diverged entirely.

---

## 4. Visualizing Multicollinearity: The 3D Cost Surface

The moderate $R^2$ score (~0.58) is a data and feature problem, not a code problem. Most tutorials stop at the correlation matrix to identify redundant features, but I wanted to see exactly how multicollinearity physically alters the optimization path.

Gradient descent minimises the cost function by iteratively updating theta in the direction of the steepest descent. The 3D cost surface and contour plots reveal how this happens for any pair of features.


**Circular contours:** Features are independent, gradient descent flows smoothly to the minimum.


**Elongated/tilted ellipse:** Features are correlated, gradient descent has to slide along a narrow valley.

The theta 7 vs theta 8 plot showed a very stretched diagonal ellipse — direct visual confirmation of the Latitude/Longitude multicollinearity. Latitude and Longitude are -0.92 correlated because they jointly encode geographic location.

When two features are highly correlated with each other, the model cannot distinguish their individual contributions and splits/distributes the weight between them arbitrarily. The predictions stay roughly correct, but the individual weights become unstable and unreliable. This exact same distortion was happening with `AveRooms` and `AveBedrms` (0.85 correlated), causing their weight signs to conflict with their actual correlations.

### Capped Data Limits

Beyond multicollinearity, the predictions vs actual plot shows a vertical line at Price = 5.0, meaning prices were artificially capped in the dataset. The model learns a false ceiling, corrupting predictions at the high end.

---

## 5. Feature Engineering Plan

Every problem identified above is a concrete, solvable step toward a significantly better model. Each problem identified has a direct feature engineering solution.


**Step 1 — Remove Capped Prices (Data Quality Fix):** These corrupted samples should be removed before any other changes.


**Step 2 — Fix AveRooms + AveBedrms (Multicollinearity Fix):** Replace both correlated features with a single meaningful ratio.


**Step 3 — Fix Latitude + Longitude (Interaction Effect):** An interaction term captures the combined effect that neither variable can express alone.


**Step 4 — Capture Non-Linearity in MedInc:** Adding a squared term allows the model to capture this.

---

## 6. Challenges I Faced

1. **Understanding the Matrix Calculus:** Deriving the partial derivatives by hand on a whiteboard was the only way I could intuitively understand how matrix multiplication shapes the gradient.
2. **Feature Scaling (Normalization):** I quickly learned that without Z-score normalization, the iterations go haywire. Features with larger ranges dominate the gradients.
3. **Choosing the Learning Rate:** Seeing the model diverge with $\alpha = 0.99$ was a harsh lesson in not getting greedy. A smaller, steady learning rate ($\alpha = 0.1$) proved much more effective.

---

## Key Takeaways

1. **Trust but Verify:** Deriving the math yourself turns a "magic" library function into a tool you intimately understand.
2. **$R^2$ Tells a Data Story:** A moderate score is rarely a failure of the algorithm; it usually highlights multicollinearity or non-linear relationships hiding in the data.
3. **Visuals are Diagnostic Tools:** Cost surface plots and correlation matrices are essential for diagnosing optimization struggles and unstable weights.

**Code:** Full code available on my GitHub: <https://github.com/mrinChanSIT/ML-Learning-Journey>
