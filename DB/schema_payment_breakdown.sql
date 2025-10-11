USE lotbotsystem;

CREATE TABLE payment_breakdown (
    payment_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '결제 내역 고유 ID',
    user_id INT NOT NULL COMMENT '결제한 사용자 ID',
    use_date DATE NOT NULL COMMENT '결제 사용 일자',
    amount INT NOT NULL COMMENT '결제 금액 (원)',
    type ENUM('정기권', '출차') NOT NULL COMMENT '결제 유형',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '내역 생성 시각',

    CONSTRAINT fk_payment_user FOREIGN KEY (user_id) 
        REFERENCES user(user_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);