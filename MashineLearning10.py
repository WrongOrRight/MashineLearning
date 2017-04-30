import sklearn
import numpy
import pandas
import sklearn.feature_extraction.text
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
import scipy
#1.Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла salary-train.csv (либо его заархивированную версию salary-train.zip).
data_train = pandas.read_csv('salary-train.csv')
data_test = pandas.read_csv('salary-test-mini.csv')

#2.Проведите предобработку:
#.Приведите тексты к нижнему регистру (text.lower()).

for clf in data_train.columns[0:1]:
    print(clf)
    data_train[clf]=data_train[clf].str.lower()
#Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова. Для такой замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text). Также можно воспользоваться методом replace у DataFrame, чтобы сразу преобразовать все тексты:
#train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', 
#  regex = True)

    data_train[clf]=data_train[clf].str.replace('[^a-zA-Z0-9]', ' ')
#Примените TfidfVectorizer для преобразования текстов в векторы признаков. Оставьте только те слова, которые встречаются хотя бы в 5 объектах (параметр min_df у TfidfVectorizer).
    TfidfVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=5,max_df=100000000)
    X=( TfidfVectorizer.fit_transform(data_train[clf]))
    
#Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'. Код для этого был приведен выше.

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

#Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
print(data_train['LocationNormalized'])
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

#Объедините все полученные признаки в одну матрицу "объекты-признаки". Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными. Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack.
X_train = scipy.sparse.hstack([X, X_train_categ])

#3. Обучите гребневую регрессию с параметрами alpha=1 и random_state=241. Целевая переменная записана в столбце SalaryNormalized.
y_train = data_train['SalaryNormalized']
model = Ridge(alpha=1)
model.fit(X_train, y_train)
#4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv. Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.
data_test['FullDescription']=data_test['FullDescription'].str.lower()
data_test['FullDescription']=data_test['FullDescription'].str.replace('[^a-zA-Z0-9]', ' ')
X_test_text = TfidfVectorizer.transform(data_test['FullDescription'])
X_test_cat = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = scipy.sparse.hstack([X_test_text, X_test_cat])

y_test = model.predict(X_test)
print(y_test)