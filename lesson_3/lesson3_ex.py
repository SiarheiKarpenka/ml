# coding: utf-8
# Задание 3
#
#
# Цель:
#
# - Поработать с массивами в numpy
# - Нучиться считывать данные из csv и визуализировать их
# - Получить навыки решения задачи линейной регрессии (без обучения, с использованием нормального уравнения)
#
#
# Для выполнения задания используете набора данных http://people.sc.fsu.edu/~jburkardt/datasets/regression/x28.txt
#
# Можете попробовать ещё и другие наборы данных: http://people.sc.fsu.edu/~jburkardt/datasets/regression/regression.html
#
# **Дополнительно**
#
# Если вы хотите чего-то более сложно, дополнительно можно попробовать решить задачу https://www.kaggle.com/c/house-prices-advanced-regression-techniques и отправить результат в kaggle. Для решения этой задачи мы ещё не разбирали много чего, но можете почитать обучающие материалы https://www.kaggle.com/c/house-prices-advanced-regression-techniques#tutorials
#
# ## Загрузите данные
#
#
# Необходимо считать данные в numpy массив, и разделить их на X (атрибуты) и y (результат, последний столбец).
#
# **Все данные нужно будет разделить в соотношении 80/20** Большую часть данных мы будем использовать для обучения. Оставшуюся часть для тестирования алгоритма. Мы пока не разибрали техники валидации, это тема для следующих заняний. Пока мы будем просто сравнивать ошибку на обучающей выборке и на тестовой. **Сделайте так, чтобы разделение на тестовую и обучающую выборку было случайным, но воспроизводимым (т.е. при каждом запуске получали одно и тоже разделение)** (с.м. numpy.random.seed)


# In[6]:


import numpy as np 

from numpy import genfromtxt
# PUT YOUR CODE HERE
raw_data = genfromtxt('x28.csv', delimiter=',', skip_header=True)
print(raw_data.shape)
print(raw_data.dtype)

# In[25]:
X_all = raw_data[:,0:16]
y_all = raw_data[:,16:]

print(X_all.shape)
print(y_all.shape)


# In[28]:
indexes = np.arange(0, len(raw_data), 1)
print(indexes)

np.random.seed(31)
np.random.shuffle(indexes)
print(indexes)


# PUT YOUR CODE HERE
X_train, X_test = X_all[:48,:], X_all[48:,:]
y_train, y_test = y_all[:48,:], y_all[48:,:]


# In[29]:

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


## 2 Попробуйте построить несколько графиков, зависимость у от какого-то одного из аттрибутов

# In[32]:

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))
plt.scatter(X_train[:,0], y_train)
plt.show()

plt.figure(figsize=(16, 8))
plt.scatter(X_train[:,1], y_train)
plt.show()

plt.figure(figsize=(16, 8))
plt.scatter(X_train[:,2], y_train)
plt.show()

# Используйте нормальное уравление, чтобы найти коэффициенты
#
# $$
# # \omega = (X^TX)^{-1}X^Ty
# # $$
#
# # In[38]:
#
#
from numpy.linalg import pinv

print(X_train.shape)
print(y_train.shape)

# PUT YOUR CODE HERE
w = pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

## Проверьте результат
print(w)

# Постройте модель линейной регрессии. Вычислите y_pred. И посчтитайте MSE. Проделайте тоже самое на тестовой выборке и сравните резултаты.
# In[ ]:
def mse(y_pred, y):
    return (y_pred-y).T.dot(y_pred-y)/len(y_pred)

# In[46]:
# PUT YOUR CODE HERE
y_pred = w.T.dot(X_train.T)


# In[47]:


print(y_pred.shape, y_train.shape)
print(mse(y_pred, y_train))


# In[48]:


# PUT YOUR CODE HERE
y_test_pred =  w.T.dot(X_test.T)


# In[50]:


print(mse(y_test_pred, y_test))


# Визуализируйте данные, ответ в зависимости от первого аттрибута. На одно графике покажите предсказанные и настоящие значения.

# In[52]:


import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))
plt.scatter(X_train[:,0], y_train,c='b')
plt.scatter(X_train[:,0], y_pred, c='r')
plt.show()


plt.figure(figsize=(16, 8))
plt.scatter(X_test[:,0], y_test,c='b')
plt.scatter(X_test[:,0], y_test_pred, c='r')
plt.show()


# В результате вы увидете что синие и красные (предсказанные) точки находятся рядом но не совпадают.