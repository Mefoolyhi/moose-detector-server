DROP DATABASE IF EXISTS `DB`;
CREATE DATABASE `DB`;
USE `DB`;

DROP TABLE IF EXISTS `Predictions`;
CREATE TABLE `Predictions` (
  `OrderTime` datetime,
  `Item` varchar(100) NOT NULL
);