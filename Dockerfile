FROM mysql/mysql-server
ADD schema.sql /docker-entrypoint-initdb.d
EXPOSE 3306
