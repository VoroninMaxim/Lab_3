#!/bin/bash

# Проверяем запущен ли скрипт посредственн из каталога Lab_3
if [[ "$(basename "$(pwd)")" != "Lab_3" ]]; then
  echo "Скрипт должен быть запущен из каталога 'Lab_3'"
  exit 1
fi

# Создание образа
docker build -t iris_app_image .

# Создаем и запускаем контейнер
docker run -d --restart always --name iris_app_container -p 8000:8000 iris_app_image