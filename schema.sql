-- Create database
CREATE DATABASE IF NOT EXISTS distracted_driver;
USE distracted_driver;

-- Users table to store both owners and drivers
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100) NOT NULL,
    user_type ENUM('owner', 'driver') NOT NULL,
    reference_image LONGTEXT,
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    risk_level INT DEFAULT 0,
    suspended BOOLEAN DEFAULT FALSE,
    suspension_message TEXT
);

-- Driver-Owner relationships
CREATE TABLE IF NOT EXISTS driver_owner (
    id INT AUTO_INCREMENT PRIMARY KEY,
    driver_id INT NOT NULL,
    owner_id INT NOT NULL,
    FOREIGN KEY (driver_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Alerts table to store drowsiness detection events
CREATE TABLE IF NOT EXISTS alerts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Vehicle information (optional)
CREATE TABLE IF NOT EXISTS vehicles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    owner_id INT NOT NULL,
    driver_id INT,
    vehicle_name VARCHAR(100) NOT NULL,
    license_plate VARCHAR(20) NOT NULL,
    FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (driver_id) REFERENCES users(id) ON DELETE SET NULL
);

