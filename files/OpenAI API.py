#!/usr/bin/env python
# coding: utf-8
Author: Cesar Jaidar
# # DSC670 Week 3 - Working with Language Models

# In[61]:


import openai
import fitz  # PyMuPDF
import json
import re


# ## We load the PDF to extract all text from it using PyMuPDF. This is crucial as the invoice data is stored as unstructured text.

# In[62]:


# Load the PDF file
pdf_path = 'dsc-670-exercise-invoice.pdf'
with fitz.open(pdf_path) as doc:
    pdf_text = ""  # Initialize an empty string
    for page in doc:  # Loop through pages to extract text
        pdf_text += page.get_text()


# ## This step is for debugging and ensuring that the text extraction works correctly.

# In[63]:


# Display extracted text for verification
print("Extracted Text from PDF:")
print(pdf_text)


# ## Setting up the OpenAI client with the API key to enable communication with the GPT model. The extracted text is sent to GPT with a specific prompt to generate structured JSON. I had to include the word (strictly) to the prompt, as OpenAI was returning non valid JSONs. 

# In[64]:


# Initialize OpenAI client with API key
client = openai.OpenAI(api_key="enter your key")

# Create an assistant
assistant = client.beta.assistants.create(
    name="Invoice Processor",
    instructions="You are an assistant that converts invoice text into a strictly valid JSON format.",
    tools=[],
    model="gpt-4",
)


# In[65]:


# Start a thread
thread = client.beta.threads.create()


# ## Same as above, I had to include additionl instructions (output only valid JSON) to work properly. 

# In[66]:


# Add the invoice text as a user message
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=f"Extract the structured JSON from the following invoice (output only valid JSON):\n{pdf_text}",
)


# In[67]:


# Run the assistant to process the request
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Process the invoice text into structured JSON."
)


# In[68]:


# Check the run status
print("Run completed with status: " + run.status)


# ## This section handles the JSON parsing and validation to ensure only valid JSON is processed.

# In[79]:


import json
import re

# Extract JSON block using regex
try:
    # Use regex to find the JSON block
    match = re.search(r'\{[\s\S]*\}', assistant_response)
    if match:
        json_content = match.group(0)  # Extract JSON string

        # Validate and load JSON
        structured_json = json.loads(json_content)

        # Print structured JSON
        print("\nExtracted Structured JSON:")
        print(json.dumps(structured_json, indent=4))

        # Save JSON to file
        output_path = "invoice_structured.json"
        with open(output_path, 'w') as json_file:
            json.dump(structured_json, json_file, indent=4)
        print(f"JSON saved successfully to {output_path}")
    else:
        print("No valid JSON block found in the response.")

except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    print("Content causing the error:")
    print(assistant_response)


# ## Results
1. Successful Extraction:
   - The extracted JSON includes all necessary fields such as address, date, invoice number, items, and totals.
   - Hierarchical organization ensures clarity and usability.

2. Validation:
   - The JSON response was validated using json.loads().
   - Regex was employed to isolate the JSON block from potential extraneous text in the API response.

3. File Output:
   - The structured JSON was saved to a file (invoice_structured.json) for easy access and reuse.- The integration of OpenAI's API with PyMuPDF successfully bridges the gap between unstructured text and structured data.
- The approach demonstrates robustness, handling both text extraction and JSON validation effectively.
- Future enhancements could include automating the process for batch processing multiple invoices or integrating additional validation steps.