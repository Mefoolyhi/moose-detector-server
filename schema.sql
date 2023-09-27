DROP DATABASE IF EXISTS `DB`;
CREATE DATABASE `DB`;
USE `DB`;

DROP TABLE IF EXISTS `prediction`;
CREATE TABLE `prediction` (
  `id` MEDIUMINT NOT NULL AUTO_INCREMENT,
  `prediction_time` datetime NOT NULL,
  `camera_id` INT NOT NULL DEFAULT 1,
  `photo` LONGBLOB NOT NULL,
  `animal_type` TEXT NOT NULL,
  `animal_count` INT DEFAULT 0,
  PRIMARY KEY (`id`)
);

CREATE INDEX camera_index ON prediction (camera_id);