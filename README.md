# python-data
 --------------------------
Вам необходимо сгенерировать свой датасет или выбрать любой публичный датасет имеющий
не менее 100 млн записей. Если не получается сделать расчеты на таком объеме - можно уменьшить размер.
имеющий колонки различных типов данных - numeric, datetime (range в несколько лет), string.
достаточно минимального набора колонок, чтобы датафайл был не сильно большой.
имеющий не менее 10% дублей (допускается добавить их самостоятельно)
хранящийся в csv / json формате в файле на диске или по url

------------------------------------

Считываем файл или тянем данные по ссылке и далее процессим данные
удалить пустые / na строки
удалить дубли
строки в которых нет цифр превратить в пустые
удалить записи в промежутке от 1 до 3 часов ночи
Для ускорения выполнения распараллеливайте выполнение этих шагов

------------------------------------

Расчет метрик
Агрегация по времени, для каждого часа рассчитать
кол-во уникальных string
среднее и медиану для numeric
Так же напишите SQL запрос для выполнения подобных расчетов напрямую в базе данных. Можно его вставить в код в виде комментария.

------------------------------------

Мерж с метриками
К каждой строке в исходном датасете примержить метрики ближайшего часа рассчитанные в предыдущем шаге

------------------------------------

Аналитические метрики
Для колонки numeric по полному датасету построить
Гистограмму
95% доверительный интервал, с комментарием как выбирали методику расчета

------------------------------------

Визуализация
Отрисовать график среднего значения numeric колонки (y) по месяцам (x).
Heatmap по частотности символов в колонке string


------------------------------------
Пример сгенерированного датасета
![image](https://github.com/sweddde/python-data/assets/115980523/ec83fbbb-63ea-4956-affb-4e5ac1ee90d3)
------------------------------------
Пример отфильтрованного датасета
![image](https://github.com/sweddde/python-data/assets/115980523/261b5d53-bf48-4296-9b81-58efb6c73d90)
------------------------------------
Пример метрик
------------------------------------
![image](https://github.com/sweddde/python-data/assets/115980523/57cb1d87-b4ce-4cf3-86cb-3c989ef90a6e)
------------------------------------
Пример соединения
![image](https://github.com/sweddde/python-data/assets/115980523/f68e085c-f490-4279-bdef-5ee0879237f3)
------------------------------------
Пример гистограммы
![image](https://github.com/sweddde/python-data/assets/115980523/d1d029f0-5e1f-4fd1-9c35-133097ab55aa)
------------------------------------
Доверительный интервал
95% Confidence Interval: (499.60171024882465, 500.73266431927806)
------------------------------------
Пример графика среднего значения
![image](https://github.com/sweddde/python-data/assets/115980523/23478fb6-1b15-4044-8649-59497909eba2)
------------------------------------
Хитмап по частоте символов
![image](https://github.com/sweddde/python-data/assets/115980523/a8c06b57-882c-4dc1-8e9b-c13053174142)
------------------------------------


