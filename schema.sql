DROP DATABASE IF EXISTS `DB`;
CREATE DATABASE `DB`;
USE `DB`;

DROP TABLE IF EXISTS `predictions`;
CREATE TABLE `predictions` (
  `id` MEDIUMINT NOT NULL AUTO_INCREMENT,
  `prediction_time` datetime NOT NULL,
  `camera_id` INT NOT NULL DEFAULT 1,
  `photo` LONGBLOB NOT NULL,
  `moose_count` INT DEFAULT 0,
  `bear_count` INT DEFAULT 0,
  `hog_count` INT DEFAULT 0,
  `lynx_count` INT DEFAULT 0,
  PRIMARY KEY (`id`)
);