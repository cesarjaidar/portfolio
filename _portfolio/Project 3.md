---
title: "Invoice Processing with GPT and PyMuPDF"
excerpt: "Extracting structured data from unstructured PDF invoices using PyMuPDF and OpenAI’s GPT model to generate valid JSON representations."
collection: portfolio
---

Description:

This project demonstrates an automated pipeline for processing invoice PDFs into structured JSON data. Key components include:

📄 Text Extraction: PyMuPDF extracts text from PDF invoices.

🤖 GPT Integration: OpenAI's GPT-4 converts the raw text into structured JSON using tailored prompts.

🛠️ Validation: Regex and json.loads() ensure the output is strictly valid JSON.

💾 File Output: The resulting JSON is saved for downstream processing.

The approach showcases practical AI applications in document automation, addressing challenges such as unstructured text, invalid JSON formats, and batch processing scalability.


🔗 [View Project on GitHub](https://github.com/cesarjaidar/portfolio/blob/master/files/OpenAI%20API.py)
