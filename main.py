import uvicorn
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import datetime
import logging
from openai import OpenAI
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from retriever import Context_Retriever

# --- 1. Configuration and Setup ---
# Configure logging for better error visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
try:
    client = OpenAI()
except Exception as e:
    logging.error(f"Error initializing OpenAI client: {e}")
    client = None

# --- 2. FastAPI Web Application ---
app = FastAPI()

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

context_retriever = Context_Retriever()

class Question(BaseModel):
    question: str

# --- 3. OpenAI Answer Generation ---
def generate_answer_with_openai(context: str, question: str):
    if not client:
        return "Error: OpenAI client not initialized. Check your API key."

    system_prompt = "You are a helpful assistant. Use only the provided context to answer the user's question. If the context doesn't contain the answer, say that you don't have information on that topic."
    user_prompt = f'Context: "{context}"\n\nQuestion: "{question}"'

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return "Sorry, an error occurred while communicating with the OpenAI AI."

# --- 4. API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(question_data: Question):
    query = question_data.question.lower().strip()

    # --- Direct Answers (If-Else Statements) for common queries ---
    if query in ["hi", "hello", "hey"]:
        return JSONResponse({"answer": "Hello! How can I help you with space biology today?"})

    if "what is the international space station" in query or "what is the iss" in query:
        return JSONResponse({"answer": "The International Space Station (ISS) is a habitable artificial satellite in low Earth orbit, serving as a microgravity and space environment research laboratory."})

    if "how old is the iss" in query:
        current_year = datetime.datetime.now().year
        age = current_year - 1998
        return JSONResponse({"answer": f"The first component of the International Space Station was launched in 1998, making it {age} years old in 2025."})
    
    if "founder of nasa" in query or "who founded nasa" in query:
        return JSONResponse({"answer": "NASA was established by President Dwight D. Eisenhower in 1958."})

    if "temperature" in query and "iss" in query:
        return JSONResponse({"answer": "The internal temperature on the ISS is kept at a comfortable 22°C (72°F)."})

    # --- Semantic Search and OpenAI Generation for other queries ---
    relevant_context = context_retriever.retrieve_context(query)
    
    if relevant_context:
        final_answer = generate_answer_with_openai(relevant_context, query)
    else:
        final_answer = "I'm sorry, my knowledge base doesn't seem to contain a specific fact about that topic."
    
    return JSONResponse({"answer": final_answer})

# --- Plotting Endpoint ---
class PlotData(BaseModel):
    data: list[float]

@app.post("/plot")
async def plot_data_endpoint(plot_data: PlotData):
    data = plot_data.data
    
    # Create the plot
    fig, ax = plt.subplots()
    ax.bar(range(len(data)), data, color='cyan')
    ax.set_title("Data Plot from AstroIntel")
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Values")
    
    # Save the plot to a byte stream
    img_bytes_io = BytesIO()
    fig.savefig(img_bytes_io, format='png')
    img_bytes_io.seek(0)
    plt.close(fig)
    
    # Encode the image to base64 for embedding in HTML
    encoded_image = base64.b64encode(img_bytes_io.read()).decode('utf-8')
    
    return JSONResponse({"image": encoded_image})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)