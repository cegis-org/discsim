# VALIData

VALIData is a simulation tool developed for the [Center for Effective Governance of Indian States (CEGIS)](https://www.cegis.org/), an organization dedicated to assisting state governments in India to achieve better development outcomes.

## Overview

An important goal of CEGIS is to improve the quality of administrative data collected by state governments. One approach is to re-sample a subset of the data and measure deviations from the original samples collected. These deviations are quantified as **discrepancy scores**, and significant scores are flagged for third-party intervention.

Often, it's unclear which re-sampling strategy yields the most accurate and reliable discrepancy scores. The goal of this project is to create a simulator that predicts discrepancy scores and assesses their statistical accuracy across different re-sampling strategies.

VALIData comprises a backend API built with FastAPI and a frontend interface developed using Streamlit. The project utilizes PostgreSQL for database management and is containerized with Docker for easy deployment.

## Getting Started

### Cloning the Repository

```bash
git clone https://github.com/cegis-org/discsim.git
cd discsim
```

### Setting Up a Virtual Environment

We recommend using a virtual environment to manage dependencies. You can use either `venv` or `conda`.

#### Using `venv`

1. **Create the virtual environment:**

   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

#### Using `conda`

1. **Create the environment:**

   ```bash
   conda create -n discsim-env python=3.11
   ```

2. **Activate the environment:**

   ```bash
   conda activate discsim-env
   ```

### Installing Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Setting Up the PostgreSQL Database

### Installing PostgreSQL

#### On Ubuntu/Linux

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

#### On macOS

Install PostgreSQL using Homebrew:

```bash
brew update
brew install postgresql
brew services start postgresql
```

#### On Windows

Download and install PostgreSQL from the [official website](https://www.postgresql.org/download/windows/).

### Configuring the Database

1. **Start the PostgreSQL service (if not already running):**

   - On Ubuntu/Linux:

     ```bash
     sudo service postgresql start
     ```

2. **Switch to the `postgres` user:**

   ```bash
   sudo -u postgres psql
   ```

3. **Create the database and user:**

   ```sql
   -- Create the database
   CREATE DATABASE discsim;

   -- Create the user with a password
   CREATE USER "user" WITH PASSWORD 'password';

   -- Grant privileges on the database
   GRANT ALL PRIVILEGES ON DATABASE discsim TO "user";
   ```

4. **Grant privileges on the `public` schema:**

   ```sql
   -- Change ownership of the public schema
   ALTER SCHEMA public OWNER TO "user";

   -- Grant all privileges on the schema
   GRANT ALL ON SCHEMA public TO "user";

   -- Grant privileges on all tables, sequences, and functions
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "user";
   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO "user";
   GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO "user";
   ```

5. **Verify the privileges (optional):**

   ```sql
   -- List all schemas and their owners
   \dn+

   -- Check privileges on the public schema
   SELECT nspname, nspowner, has_schema_privilege('user', nspname, 'CREATE, USAGE') AS has_privs
   FROM pg_namespace
   WHERE nspname = 'public';
   ```

6. **Exit the `psql` shell:**

   ```sql
   \q
   ```

## Configuring Environment Variables

Create a `.env` file in the project's root directory and add the following content:

```env
# API configuration
API_BASE_URL="http://localhost:8000"

# PostgreSQL configuration
POSTGRES_USER="user"
POSTGRES_PASSWORD="password"
POSTGRES_DB="discsim"

# Database URL
DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}"

# Log level
LOG_LEVEL=info
```

Ensure that the `DATABASE_URL` matches your local PostgreSQL configuration.

## Running the Application

### Running the Backend API and Frontend Locally

#### Starting the Backend API Server

1. **Activate your virtual environment if not already active:**

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

2. **Run the API server:**

   ```bash
   python api/run.py
   ```

   The API server will start on `http://localhost:8000`.

#### Starting the Frontend Streamlit App

1. **Open a new terminal window.**

2. **Activate your virtual environment.**

3. **Run the Streamlit app:**

   ```bash
   streamlit run dashboard/Home.py --server.port=8501
   ```

   The frontend will be accessible at `http://localhost:8501`.

#### Accessing the Application

- **Frontend Interface:** Open your web browser and navigate to `http://localhost:8501` to interact with the application.
- **API Documentation:** Access the API docs at `http://localhost:8000/docs`.

### Running the Application with Docker (Optional)

If you prefer to use Docker, you can run the entire application stack using Docker Compose.

#### Prerequisites

- **Docker**
- **Docker Compose**

#### Steps

1. **Build and start the containers:**

   ```bash
   docker-compose build
   docker-compose up
   ```

2. **Access the services:**

   - **API Server:** `http://localhost:8000`
   - **Frontend:** `http://localhost:8501`

   **Note:** The PostgreSQL database runs inside Docker and is accessible to the other containers.

## Contributing

We welcome contributions! If you'd like to contribute to DiscSim:

1. **Fork the repository.**
2. **Create a new branch for your feature or bug fix:**

   ```bash
   git checkout -b feature-name
   ```

3. **Commit your changes and push to your fork.**
4. **Submit a pull request.**

For major changes, please open an issue first to discuss your ideas.

## License

MIT License.

## Acknowledgments

Thank you for checking out VALIData! We hope this tool aids in enhancing the quality of administrative data and contributes to better governance and development outcomes.
