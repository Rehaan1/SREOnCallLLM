# Nebula Runbook LLM

## A Runbook Assistant for Site Reliability Engineers

### Steps to Run

1. Clone the repository

2. Install the required packages

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory of the project and add the following environment variables:

```bash
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

4. Run the application

```bash
streamlit run app.py
```

5. Open the browser and navigate to `http://localhost:8501`

### Features to Develop:

- [ ] **Runbook Upload Capability**

  - Implement functionality to upload runbooks directly into the system.

- [x] **Intelligent Context Selection**

  - Utilize agents to select the correct context, rather than relying on all available context.

- [x] **Get Context from PDF, Web Pages**

  - Initialize vector to get context from PDFs and web pages.

- [ ] **Get Context References for Agent**

  - Develop a system to provide context references used by agent.

- [ ] **Store a history of previous responses and contexts used given prompt**

  - Implement a system to store the history of previous responses and contexts used.

- [x] **Persistent Vector Database Storage**

  - Develop a system to create and store a vector database in permanent storage locally.

- [x] **Persistent Vector Database Storage On Cloud**

  - Develop a system to store the vector database on cloud storage.

- [ ] **Contextual Continuity**

  - Ensure that the context from previous responses is carried forward appropriately.

- [ ] **Optimize Vector Database Initialization**
  - Improve the latency associated with initializing the vector database.
