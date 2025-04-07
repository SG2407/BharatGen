from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import requests
from mcp import MCPClient

app=FastAPI()
mcp=MCPClient(storage_backend='sqlite',storage_path='/app/mcp_data/chatbot.db')
class ChatRequest(BaseModel):
    patient_id:str
    query:str
def get_llm_response(prompt:str)->str:    
    response=requests.post(
        "http://llm-api:8080/generate",
        json={"inputs":prompt},
        headers={"Content-Type":"application/json"}
    )
    
    return response.json()[0]["generated_text"]
    
def get_patient_history(patient_id:str)->str:
    conn=sqlite3.connect("./mcp_data/chatbot.db")
    cursor=conn.cursor()
    cursor.execute(
        """SELECT prompt, response FROM mcp_interactions
            WHERE patient_id = ?
            ORDER BY timestamp DESC LIMIT 3
        """,(patient_id,))
    history="\n".join([f"Q: {p}\nA: {r}" for p,r in cursor.fetchall()])
    conn.close()
    return history
@app.post('/chat')
async def chat(request:ChatRequest):
    history=get_patient_history(request.patient_id)
    context=mcp.get_context(request.patient_id)

    prompt=f"""
    [Patient History]
    {history}

    [New Query]
    {request.query}

    [Patient Context]
    {context}

    [Instructions]
    Respond empathetically as a healthcare assistant.
"""

    response=get_llm_response(prompt)

    conn=sqlite3.connect("./mcp_data/chatbot.db")
    cursor=conn.cursor()
    cursor.execute("""
        INSERT INTO  mcp_interactions (patient_id, prompt, response)
                   VALUES (?,?,?)
""",(request.patient_id,request.query,response))
    
    conn.commit()
    conn.close()

    return {"response":response}
