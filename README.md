# ✈️ Aviation Defect-Repair Hybrid Matching Engine

> **⚠️ DISCLAIMER:** *Code not fully shared due to data restriction policy and non-disclosure agreement (NDA). The repository contains a redacted/structural overview of the project for portfolio purposes.*

A Python-based NLP project designed to match newly reported aircraft cabin/system issues (*Defects*) with historical maintenance records (*Repairs*). This system operates as an advanced **Retrieval** engine utilizing a **Hybrid Matching** approach, combining **Semantic Vector Search** (for contextual understanding) and **Rule-based Regex** (for high-precision extraction of domain-specific entities such as seat numbers, lavatories, galleys, etc.).

This project serves as the foundational layer (the *Retrieval-Augmented* phase) for building an AI-powered aviation technician assistant.

## ✨ Key Features
* **Local & Hardware-Aware Execution:** Runs seamlessly in a local environment. Automatically detects and utilizes hardware acceleration (**CUDA** for NVIDIA, **MPS** for Apple Silicon, or **CPU**) for the embedding process.
* **Auto-Translation Pipeline:** Dynamically translates defect descriptions from any language into English using `deep-translator` to maximize the performance of the embedding model.
* **Semantic Vector Search:** Utilizes the `sentence-transformers` model (`all-MiniLM-L6-v2`) to compute the semantic similarity (*cosine similarity*) between current issues and thousands of historical repair logs.
* **Domain-Specific Regex Bonus:** Extracts physical aircraft entities (Seat Row/Letter, Lavatory, Galley, Door) from the original text. The system applies a *multiplier bonus score* if physical location matches are detected, preventing the AI from hallucinating on specific details.
* **Live Database Integration:** Retrieves real-time defect reporting data directly from Google Sheets via a Service Account, integrated with a local Excel-based maintenance database.


2. **Install dependencies:**
   It is highly recommended to use a Virtual Environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Credentials & Database:**
   Ensure the Google Service Account file (`credentials.json`) and the latest historical maintenance data are placed in the correct directories according to the internal engineering team's configuration.

## 🚀 Usage


**System Workflow:**
1. Fetches the latest *Defects* data from the designated operational Google Sheets tab.
2. Reads historical *Repairs* data from the local database.
3. Translates complaint descriptions, vectorizes the text, and finds the best matches (*Top N*).
4. Saves the matching results along with semantic and regex bonus scores into a JSON file as the final output.

## ⚙️ Parameter Configuration (Tuning)
Scoring parameters can be adjusted in the main configuration within the script:
* `TOP_N = 3`: The number of top repair recommendations to output.
* `SIM_THRESHOLD = 0.40`: The minimum *Cosine Similarity* threshold (0.0 - 1.0) for a candidate to be considered relevant.
* `SEAT_BONUS_MULT = 0.3`: The additional weight (*multiplier*) applied for regex matches on physical entities.

## 🔮 Future Roadmap: Towards Full RAG

Currently, the system operates as a highly accurate **Retrieval (R)** engine using a hybrid method. The next development phase aims to complete this pipeline into a full **RAG (Retrieval-Augmented Generation)** architecture by integrating the **Generation (G)** component.

* **LLM Integration:** Connect the JSON output from this system to an LLM API (such as Google Gemini, OpenAI, or Claude) as a Context Prompt.
* **Automated Action Plan:** Utilize the LLM to synthesize the search results and automatically generate a summary of 3-5 initial investigation/troubleshooting steps for the technicians.
* **Chatbot UI:** Build an interactive user interface using frameworks like Streamlit so mechanics can input defects and instantly receive data-driven repair guidelines based on historical data.
```
