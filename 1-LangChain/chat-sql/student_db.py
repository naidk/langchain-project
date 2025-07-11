import sqlite3

# Connect to the database
connection = sqlite3.connect('student.db')
cursor = connection.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS STUDENT(
    NAME VARCHAR(25), 
    CLASS VARCHAR(25), 
    SECTION VARCHAR(25), 
    MARKS INT
)""")

# Optional: Clear old records (for testing repeat runs)
cursor.execute("DELETE FROM STUDENT")

# Insert multiple records
students = [
    ('John', 'DataScience', 'A', 85),
    ('Jane', 'DataScience', 'B', 90),
    ('Mukesh', 'DataScience', 'B', 90),
    ('Jacob', 'DEVOPS', 'B', 90),
    ('Ravi', 'DEVOPS', 'A', 90)
]
cursor.executemany("INSERT INTO STUDENT VALUES (?, ?, ?, ?)", students)

# Display all records
print("All records in the STUDENT table:")
for row in cursor.execute("SELECT * FROM STUDENT"):
    print(row)

# Commit and close
connection.commit()
connection.close()
