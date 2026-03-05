# %%
import numpy as np    # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt    # pyright: ignore[reportMissingImports]

# %%
class LinearRegression:
    """
    Linear Regressopm implemented from scratch

    Supports:
    - Gradient Descent optimization
    - Normal equation (closed form solution)
    - Feature Normalization
    - Polynomial Features
    - Learning curve visualization
    """

    def __init__(self, alpha = 0.01, num_iterations = 1000, method = 'gradient_descent', normalize = True):
        """
        Initializing the Model 

        Parameters:
        -----------

        alpha : float
            Learning rate for gradient descent
        num_iterations : int
            Number of iterations for gradient descent
        method : str
            Optimization method to use ('gradient_descent' or 'normal_equation')
        normalize  : bool
            Whether to normalize features
        """

        self.alpha = alpha
        self.num_iterations = num_iterations
        self.method = method
        self.normalize = normalize

        # Will be set during training
        self.theta = None
        self.mu = None
        self.sigma = None
        self.cost_history = []
        self.theta_history = []
    
    def _add_bias(self, X):
        """ 
        Add a biad term (column of ones) to X - one bias ter for each example (i = 1...m)
        """

        X = np.c_[np.ones(X.shape[0]), X]
        return X
    
    def _feature_normalize(self, X):
        """ 
        Normalize features - using Z-score normalization
        axis = 0 means - operations moved down the columns across all rows at once. (i.e. for each feature)
                        X (m, n) -> mu (1, n)
                        X (m, n) -> sigma (1, n)
        """

        X_norm = X.copy()
        mu = np.mean(X, axis = 0)
        sigma = np.std(X, axis = 0)

        sigma[sigma == 0] = 1 # avoid division by zero  : if all values are the same, std is 0 : boolean masking : Find all 0s in sigma and replace them with 1s
        X_norm[:,] = (X  - mu) / sigma # element-wise operation for each feature (column)
        # X_norm (m, n) -> mu (1, n) -> sigma (1, n)
        #X_norm = (X  - mu) / sigma

        return X_norm, mu, sigma
    
    def _compute_cost(self, X, y, theta):
        """ 
        Compute the cost function for linear regression
        """ 

        m = len(y) # number of training examples
        
        #Prediction
        predictions = X @ theta

        #Error
        err = predictions - y
        

        #Cost

        cost = (1/ (2*m)) * np.sum(err ** 2)

        return cost

    def _gradient_descent(self, X, y):
        """ 
        Perform Gradient descent to update theta for set num_iterations
        """

        m,n = X.shape

        theta = np.zeros(n)

        for i in range(self.num_iterations):

            #prediction
            prediction = X @ theta

            #error
            err = prediction - y
            # print("err.shape", err.shape)

            #Gradient
            gradient = (1/m) * X.T @ err
            # print('gradient.shape', gradient.shape)

            #Update theta
            theta = theta - self.alpha * gradient

            #Compute cost
            cost = self._compute_cost(X, y, theta)
            self.theta_history.append(theta.copy())
            self.cost_history.append(cost)

            #Print Prgress
            if i % 100 == 0:
                print(f"Iteration {i} : Cost = {cost: .4f}")

        return theta

    def _normal_equation(self, X, y):
        """ 
        Solve using the normal equation : Theta =(X^T X)^-1 X^T y 

        """

        theta = np.linalg.pinv(X.T @ X) @ X.T @ y

        return theta
    
    def fit(self, X, y):
        """ 
        Train the model

        parameters:
        -----------
        
        X : numpy array (m,n) 
            Training features (m examples, n features)
        y : numpy array (m,)
            Training labels
        """

        if self.normalize:
            X_norm, self.mu, self.sigma = self._feature_normalize(X)
        else:
            X_norm = X
        
        #Add biar term to X
        X_with_bias = self._add_bias(X_norm)

        if self.method == 'gradient_descent':
            self.theta = self._gradient_descent(X_with_bias, y)
        elif self.method == 'normal_equation':
            self.theta = self._normal_equation(X_with_bias, y)
        else:
            raise ValueError(f"Invalid method: {self.method}")
        
        return self
    
    def predict(self, X):
        """ 
        Make predictions for new data

        parameters : 
        -----------

        X : numpy array (m,n)
            New Features (m examples, n features)

        Returns:
        --------

        predictions : numpy array of shape (m,) 
        """ 

        if self.theta is None:
            raise ValueError("Model not trained yet. Please call fit() first.")

        if self.normalize:
            X_norm = (X - self.mu) / self.sigma
        else:
            X_norm = X

        #Add bias term to X
        X_with_bias = self._add_bias(X_norm)
        
        predictions = X_with_bias @ self.theta
        
        return predictions
    
    def score(self, X, y):
        """ 
        Calculate R^2 score for the model

        Returns : r2_score : float
        """ 
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2) # residual sum of squares
        ss_total = np.sum((y - np.mean(y)) ** 2) # total sum of squares

        r2_score = 1 - (ss_res / ss_total)

        return r2_score
    
    def plot_cost_history(self):
        """ 
        Plot the cost history
        """

        if not self.cost_history:
            print("No cost history to plot. Cost history is not available. Use fit() to train the model first.")
            return
        
        plt.figure(figsize = (10, 6))
        plt.plot(self.cost_history, label = 'Cost')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost History during training')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_predictions(self, X, y):
        """Plot predictions vs actual (only for 1D features)"""
        if X.shape[1] != 1:
            print("Can only plot for single feature. Use plot_predictions_vs_actual instead.")
            return
        
        predictions = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.5, label='Actual')
        
        # Sort for line plot
        sort_idx = np.argsort(X[:, 0]) # tell numpy to sort the array X[:, 0] : 0th column of X
        plt.plot(X[sort_idx], predictions[sort_idx], 'r-', linewidth=2, label='Predicted')
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Linear Regression Fit')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_predictions_vs_actual(self, X, y):
        """Plot predicted vs actual values"""
        predictions = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y.min(), predictions.min())
        max_val = max(y.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect predictions')
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predictions vs Actual Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def plot_cost_surface(self, X, y, theta_i = 0, theta_j = 1):
        """ Plot the cost Surface for 2D features. Call adter fit() for more than 1 feature """
        X_norm = (X - self.mu) / self.sigma
        if not self.theta_history:
            raise RuntimeError("No theta history to plot. Use fit() to train the model first.")
        
        theta_final = self.theta_history[-1]

        m = len(y)

        print(f"theta_final: {theta_final}")
        t_i = np.linspace(theta_final[theta_i] - 10, theta_final[theta_i] + 10, 100)
        t_j = np.linspace(theta_final[theta_j] - 10, theta_final[theta_j] + 10, 100)

        T_i, T_j = np.meshgrid(t_i, t_j)
        
        
        # Compute cost over grid
        Z = np.zeros(T_i.shape)
        for a in range(T_i.shape[0]):
            for b in range(T_i.shape[1]):
                theta_temp = theta_final.copy()
                theta_temp[theta_i] = T_i[a, b]
                theta_temp[theta_j] = T_j[a, b]
                # print(f"theta_temp: {theta_temp}")
                pred = X_norm @ theta_temp[1:] + theta_temp[0]
                Z[a, b] = (1 / (2 * m)) * np.sum((pred - y) ** 2)

        # Extract GD trace for the two chosen thetas
        trace_i = [th[theta_i] for th in self.theta_history]
        trace_j = [th[theta_j] for th in self.theta_history]

        # Plot
        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(T_i, T_j, Z, cmap='viridis', alpha=0.7)

        # ax1.plot_surface(T_i, T_j, Z, cmap='viridis', alpha=0.7)
        ax1.plot(trace_i, trace_j, self.cost_history,
                 color='red', linewidth=2, marker='o', markersize=2, label='GD Trace')
        ax1.set_xlabel(f'θ{theta_i}')
        ax1.set_ylabel(f'θ{theta_j}')
        ax1.set_zlabel('Cost')
        ax1.set_title('Cost Surface + Gradient Descent Trace')
        ax1.legend()

        ax2 = fig.add_subplot(122)
        cp = ax2.contourf(T_i, T_j, Z, levels=40, cmap='viridis') #what are levels? Ans: The number of contour lines to draw
        ax2.contour(T_i, T_j, Z, levels=40, colors='white', linewidths=0.3, alpha=0.4)
        plt.colorbar(cp, ax=ax2, label='Cost', shrink=0.5)
        ax2.plot(trace_i, trace_j, color='red', linewidth=2,
                 marker='o', markersize=2, label='GD Trace')
        ax2.set_xlabel(f'θ{theta_i}')
        ax2.set_ylabel(f'θ{theta_j}')
        ax2.set_title('Contour View')
        ax2.legend()

        plt.tight_layout()
        plt.show()
        
    


    


# %%
# ===== COMPREHENSIVE TESTING =====

# print("="*50)
# print("TEST 1: Simple 1D Linear Regression")
# print("="*50)

# # Generate simple data
# np.random.seed(42)
# X = np.random.randn(100,1)
# # print("X", X)
# y = 2 + 3 * X[:,0] + np.random.randn(100)*0.5 

# # Train with gradient descent
# model_gd = LinearRegression(alpha = 0.1, num_iterations=1000, method = 'gradient_descent')
# model_gd.fit(X, y)

# print(f"\nGradient Descent theta: {model_gd.theta}")
# print(f"R^2 score: {model_gd.score(X, y):.4f}")

# # %%
# # Train with normal equation
# model_ne = LinearRegression(method='normal_equation')
# model_ne.fit(X, y)

# print(f"\nNormal Equation theta: {model_ne.theta}")
# print(f"R^2 score: {model_ne.score(X, y):.4f}")

# # %%
# # Visualize
# model_gd.plot_cost_history()
# model_gd.plot_predictions(X, y)

# # %%
# print("\n" + "="*50)
# print("TEST 2: Multivariable Regression")
# print("="*50)

# # Generate Data with multiple features
# X_multi = np.random.randn(100, 3)
# y_multi = 1 + 2 * X_multi[:, 0] - 1.5 * X_multi[:,1] + 0.5 * X_multi[:,2] + np.random.randn(100)*0.5

# model_gd_multi = LinearRegression(alpha=0.1, num_iterations=1000)
# model_gd_multi.fit(X_multi, y_multi)

# print(f"\nGradient Descent theta: {model_gd_multi.theta}")
# print(f"R^2 score: {model_gd_multi.score(X_multi, y_multi):.4f}")


# # %%
# model_gd_multi.plot_cost_history()
# model_gd_multi.plot_predictions_vs_actual(X_multi, y_multi)

# # %%
# print("\n" + "="*50)
# print("TEST 3.1: Without Normalization (harder to converge)")
# print("="*50)

# # Generate data with different scales
# X_scaled = np.c_[np.random.randn(100)*1000, np.random.randn(100)*0.01]
# y_scaled = 5 + 0.5 * X_scaled[:,0] -200 * X_scaled[:,1] + np.random.randn(100)*10 

# model_no_norm = LinearRegression(alpha = 0.0001, num_iterations=1000, normalize = False)
# model_no_norm.fit(X_scaled, y_scaled)
# print(f"\nWithout normalization R^2: {model_no_norm.score(X_scaled, y_scaled):.4f}")
# print("\n" + "="*50)
# print("TEST 3.2: With Normalization (Can to converge)")
# print("="*50)

# model_with_norm = LinearRegression(alpha=0.1, num_iterations=1000, normalize=True)
# model_with_norm.fit(X_scaled, y_scaled)


# print(f"With normalization R^2: {model_with_norm.score(X_scaled, y_scaled):.4f}")


# %%



