USE lotbotsystem;

CREATE TABLE IF NOT EXISTS membership_user (
    membership_id INT AUTO_INCREMENT PRIMARY KEY,              -- PK
    user_id INT NOT NULL,                                      -- FK (user.user_id)
    membership_start DATE NOT NULL,                            -- 구매 개시일
    membership_end DATE NOT NULL,                              -- 기간 종료일
    CONSTRAINT fk_membership_user_user
        FOREIGN KEY (user_id) REFERENCES user(user_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);