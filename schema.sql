DROP DATABASE IF EXISTS `db`;
CREATE DATABASE `db`;
USE `db`;

CREATE TABLE IF NOT EXISTS `prediction` (
  `id` MEDIUMINT NOT NULL AUTO_INCREMENT,
  `prediction_time` datetime NOT NULL,
  `camera_id` INT NOT NULL DEFAULT 1,
  `photo` LONGBLOB NOT NULL,
  `lynx_count` INT DEFAULT 0,
  `brown_bear_count` INT DEFAULT 0,
  `moose_count` INT DEFAULT 0,
  `wild_boar_count` INT DEFAULT 0,
  `other_count` INT DEFAULT 0,
  `area_type` TEXT NOT NULL,
  PRIMARY KEY (`id`)
);

CREATE INDEX camera_index ON prediction (camera_id);