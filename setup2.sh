#!/bin/bash

# If you're encountering this error: psql: FATAL: Peer authentication failed for user "postgres"
# See this answer: https://stackoverflow.com/a/21166595 and change authentication method to trust for postgres and *all* users
# If you're still not sure about what to do, send Sergio a message

# If you don't use virtualenv the modules will be installed for the current user
# Please consider installing virtualenv so as to not pollute your python environment (sudo apt install python3-virtualenv in Ubuntu)

# Check if user has PSQL installed first
if [ -x "$(command -v psql)" ]; then
	# NOTE: The minimum version is 9.6 since it introduces some syntax used
	# 	e.g. ALTER TABLE table_name ADD COLUMN IF EXISTS was introduced in 9.6
	# Check what version of PostgreSQL is installed
	[[ "$(psql --version)" =~ ([0-9]*[.][0-9]) ]] && version="$BASH_REMATCH[1]"
	
	 

	 


	psql -U postgres -c "CREATE ROLE dbadmin WITH LOGIN PASSWORD 'dbadmin';"
	psql -U postgres -c "ALTER ROLE dbadmin CREATEDB;"

	psql -U postgres -c "CREATE DATABASE userdb OWNER dbadmin;"
	psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE userdb TO dbadmin;"
	psql -U postgres -c 'ALTER DATABASE userdb SET datestyle TO "ISO, MDY";'

	psql -U postgres -c "CREATE DATABASE test_userdb OWNER dbadmin;"
	psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE test_userdb TO dbadmin;"
	psql -U postgres -c 'ALTER DATABASE test_userdb SET datestyle TO "ISO, MDY";'

	psql -U dbadmin -d userdb -f sql/tables.sql
	psql -U dbadmin -d test_userdb -f sql/tables.sql
else
	echo "PostgreSQL 9.6 is required (newer versions may work as well)"
	echo "Please install it using your package manager"
	echo "Aborting setup"
	exit
fi



echo "Finished installing dependencies"
echo 'To run the webapp, call "python3 run.py"'
