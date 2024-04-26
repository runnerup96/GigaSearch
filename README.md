# Compas+ Knowledge Navigator

Это код решения на хакатон "GigaChat". 

1) Сначала получить токены, для работы с Google Search и GigaChatAPI:

Для `GOOGLE_API_KEY` , получить отсюда https://console.cloud.google.com/apis/api/customsearch.googleapis.com/credentials

Для `GOOGLE_CSE_ID` , получить отсюда https://programmablesearchengine.google.com/

Для `GIGACHAT_API_KEY`, получить отсюда https://developers.sber.ru/portal/products/gigachat-api


## Настройка
Настройка окружения:
1) Установить окружение conda:
    ```conda create --name <my-env>```
2) Активировать его:
```conda activtate <my-env>```
3) Установить библиотеки:
```pip install -r requirements.txt```


## Запуск решения

Решение запускается локально одной строчкой
```
streamlit run web_explorer.py
```
Для развертывания онлайн - https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app