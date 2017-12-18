import pandas as pd

import numpy as np

dataset = pd.read_csv("titanic.csv", index_col='PassengerId')
dataset.head()

dataset = dataset[['Fare', 'Pclass', 'Age', 'Sex', 'Survived']]
dataset.loc[dataset['Sex'] == 'male', 'Sex'] = 0
dataset.loc[dataset['Sex'] == 'female', 'Sex'] = 1
dataset.head()

mean_age = dataset['Age'].mean()

dataset['Age'] = dataset[['Age']].fillna(value=mean_age)

dataset.head()


X = np.array( dataset[['Fare', 'Pclass', 'Age', 'Sex']] )
y = np.array( dataset[['Survived']] )[:,0]


# ### Обучите логистическую регрессию
#
# Убедитесь что вы завершили работу в файле ```linear_model.py```.

# In[127]:


from lesson_5.exercise.linear_model import LogisticRegression


# In[121]:


clf = LogisticRegression(epsilon=0.000001)


# In[122]:


clf.fit(X,y, verbose=False)


# ### Посмотрите значение целевой функции
#

# In[11]:


clf.cost(X, y)


# ### Предсказание на тестовых данных
#
# Чтобы завершить задание скачайте тестовую выборку с kaggle. И загрузите датасет

# In[82]:


testset = pd.read_csv("lesson_5/exercise/test.csv", index_col='PassengerId')
testset.head()


# ### Проделайте такие же преобразования
#
# - возьмите только столбцы ['Fare', 'Pclass', 'Age', 'Sex']
# - замените текстовое представление пола на числовое
# - замените пропуски в возрасте средним значением по столбцу Age

# In[83]:


testset = testset[['Fare', 'Pclass', 'Age', 'Sex']]
testset.loc[testset['Sex'] == 'male', 'Sex'] = 0
testset.loc[testset['Sex'] == 'female', 'Sex'] = 1
mean_age_test = testset['Age'].mean()
testset['Age'] = testset[['Age']].fillna(value=mean_age_test)

# В датасете есть ещё пропуск в одном месте, заполним его просто нулём
testset = testset.fillna(0)
testset.head(5)


# In[84]:


# Преобразуем DataFrame в массив
X_test =np.array( testset[['Fare', 'Pclass', 'Age', 'Sex']] )


# In[85]:


# Получим результат в виде вероятностей
result = clf.predict(X_test)


# In[86]:


# заменим вероятности метками классов
result[[result>0.5]]=1
result[[result<0.5]]=0


# ### Сформируйте результат в заданном формате и отправьте результат в kaggle
#
# Ожидаемый результат около 74% точности
#

# In[22]:


testset.insert(0,"Survived",result)
testset['Survived'] = testset['Survived'].astype(int)
testset[['Survived']].to_csv('submition.csv')


# ### Отправьте выполненное задание на проверку, с указанием имени пользователя, для проверки в kaggle
