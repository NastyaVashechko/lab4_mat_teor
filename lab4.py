import numpy as np
import matplotlib.pyplot as plt

expected_returns = np.array([0.2, 0.4, 0.6])  
std_devs = np.array([0.1, 0.18, 0.3])         
correlations = np.array([[1, 1, -1],         
                         [1, 1, -1],
                         [-1, -1, 1]])

cov_matrix = np.outer(std_devs, std_devs) * correlations

def portfolio_return(weights):
    return np.dot(weights, expected_returns)

def portfolio_risk(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

risks = []
returns = []
efficient_weights = []

for w1 in np.arange(0, 1.1, 0.1):  
    for w2 in np.arange(0, 1.1 - w1, 0.1): 
        w3 = 1 - w1 - w2  
        if w3 < 0:
            continue  
        
        weights = np.array([w1, w2, w3])
        risk = portfolio_risk(weights)
        ret = portfolio_return(weights)

        risks.append(risk)
        returns.append(ret)
        efficient_weights.append(weights)

optimal_index = np.argmin(risks) 
optimal_weights = efficient_weights[optimal_index]  
optimal_return = returns[optimal_index] 
optimal_risk = risks[optimal_index]

print("Оптимальна структура портфеля:")
print(f"Вага акцій A1: {optimal_weights[0]:.4f}")
print(f"Вага акцій A2: {optimal_weights[1]:.4f}")
print(f"Вага акцій A3: {optimal_weights[2]:.4f}")
print(f"Очікувана норма прибутку: {optimal_return:.2%}")
print(f"Мінімальний ризик (середньоквадратичне відхилення): {optimal_risk:.2%}")

plt.plot(risks, returns, 'g-', label='Efficient Frontier')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier for Portfolio')
plt.legend()
plt.grid()
plt.show()
