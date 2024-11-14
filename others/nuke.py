#Written By Shankar to clear the User Authentication Database
import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('./sqlitedb/users.db')
c = conn.cursor()

# Delete all records from the users table
c.execute('DELETE FROM users')
conn.commit()

# Verify that the table is empty
c.execute('SELECT * FROM users')
print(c.fetchall())  # Should print an empty list

# Close the connection
conn.close()