USE lotbotsystem;

CREATE TABLE IF NOT EXISTS user (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    marketing_opt_in BOOLEAN NOT NULL DEFAULT FALSE,
    name VARCHAR(50) NOT NULL,
    birth_date DATE NOT NULL,
    phone_number VARCHAR(15) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    car_number VARCHAR(20) NOT NULL UNIQUE,
    user_role ENUM('user', 'admin') NOT NULL DEFAULT 'user',
    subscribe_membership BOOLEAN NOT NULL DEFAULT FALSE,
    kakao_auth VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- 전화번호 숫자 길이 제한 (예: 10~11자리)
    CONSTRAINT chk_phone CHECK (LENGTH(phone_number) BETWEEN 10 AND 11),
    -- 차량번호 형식 제한 (예: 최소 5자리 이상)
    CONSTRAINT chk_car_number CHECK (CHAR_LENGTH(car_number) >= 5)
);