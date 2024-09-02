FROM python:3.10
# переключаем на root
USER root
# устаналиваем рабочую директорию
WORKDIR /app
# копируем файлы приложения в контейнер
COPY . /app/
# устанавливаем зависимости
RUN pip install -r requirements.txt
# запускаем приложение
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--post", "8000"]