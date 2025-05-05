from sklearn.linear_model import Perceptron

X = [[0,0], [0,1], [1,0], [1,1]]
y_and = [0, 0, 0, 1]

clf_and = Perceptron(max_iter=10, eta0=0.1, tol=1e-3)
clf_and.fit(X, y_and)
print("Resultados AND con sklearn:")
for i in X:
    print(f"{i} => {clf_and.predict([i])[0]}")

y_or = [0, 1, 1, 1]
clf_or = Perceptron(max_iter=10, eta0=0.1, tol=1e-3)
clf_or.fit(X, y_or)
print("\nResultados OR con sklearn:")
for i in X:
    print(f"{i} => {clf_or.predict([i])[0]}")
