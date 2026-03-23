
CREATE TABLE IF NOT EXISTS students (
    student_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    attendance_percentage REAL    NOT NULL CHECK(attendance_percentage BETWEEN 0  AND 100),
    assignment_score      REAL    NOT NULL CHECK(assignment_score      BETWEEN 0  AND 100),
    midsem_score          REAL    NOT NULL CHECK(midsem_score          BETWEEN 0  AND 100),
    semester              INTEGER NOT NULL CHECK(semester              BETWEEN 1  AND 8),
    course_difficulty     INTEGER NOT NULL CHECK(course_difficulty     BETWEEN 1  AND 5),
    dropout_risk          INTEGER NOT NULL CHECK(dropout_risk          IN (0, 1))
);
