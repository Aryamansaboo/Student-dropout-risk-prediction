CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attendance_percentage REAL,
    assignment_score REAL,
    midsem_score REAL,
    semester INTEGER,
    course_difficulty INTEGER,
    dropout_risk INTEGER
);
