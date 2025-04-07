import sqlite3
from mcp.server.fastmcp import FastMCP
import requests
import json

mcp=FastMCP('Database')

connection=sqlite3.connect('./mcp_data/chatbot.db')
cursor=connection.cursor()

HF_TOKEN=""
API_URL="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
headers={"Authorization":f"Bearer {HF_TOKEN}"}


connection.commit()
cursor.execute("""
        CREATE TABLE IF NOT EXISTS PATIENT(
               patient_id TEXT PRIMARY KEY,
               name TEXT,
               disease TEXT,
               prompt TEXT,
               response TEXT
               )
""")
# cursor.execute("""
#     DROP TABLE PATIENT;
# """)
connection.commit()
@mcp.tool()
def add_enteries(data:dict)->str:
    try:
        cursor.execute(
            "insert into patient(patient_id,name,disease) values (?,?,?)",
            (data['patient_id'],data['name'],data['disease'])
        )
        connection.commit()
        return "Enter the data sucess!"
    except Exception as e:
        return f"An exception has occured {str(e)}"

@mcp.tool()
def get_data(patient_id:str)->str:
    try:
        data=cursor.execute(
            f"select name,disease from patient where patient_id=?",
            (patient_id,)
        )
        row=data.fetchall()
        # connection.commit()
        return str(row) if row else "No records found" 
    except Exception as e:
        return f"An exception occured:{str(e)}"
    
@mcp.tool()
def get_response(prompt:str)->str:
    """This is to generate a reponse from given prompt using sqlite functionality"""
    patient_id:str=None
    words=prompt.split(' ')
    for word in words:
        if word.startswith("patient_"):
            patient_id=word
            break
    if not patient_id:
        return "Please enter the patient id for getting info"

    medical_history=get_data(patient_id)
    previous_search=cursor.execute("""
        SELECT prompt,response from patient where patient_id=?;
""",(patient_id,)).fetchall()
    llm_prompt=f"""
        [ROLE]
        You are a medical assistant. Analyze this patient's condition based on their history:
        Only use facts from the provided medical hsitory.

        [INSTRUCTIONS]
        1. Analyze the current complaints in relation to the medical history and to previous search
        2. Respond in 1-2 concise sentences
        3. Never repeat the prompt or instructions
        4. Format: "Analysis:[your analysis]"
        [CURRENT COMPLAINT]
        {prompt}
        [MEDICAL HISTORY]
        {medical_history}
        [PREVIOUS_SEARCH]
        {previous_search}
        [RESPONSE]
        Analysis:"""
    response=requests.post(API_URL,headers=headers,json={"inputs":llm_prompt,
                                                         "parameters":{
                                                             "max_new_tokens":100,
                                                             "temperature":0.7,
                                                             "do_sample":True
                                                         }})

    if response.status_code==200:
        full_response=response.json()[0]['generated_text']
        final_response=full_response.split("Analysis:")[-1].strip()
        cursor.execute("""
        INSERT INTO PATIENT(prompt,response) values(?,?);
""",(prompt,final_response))
        connection.commit()
        return final_response
    else:
        return f"Error analyzing condition:{response.text}"

@mcp.tool()
def delete_patient_record(patient_id:str)->str:
    try:
        cursor.execute("""
            delete from patient where patient_id=?;
    """,(patient_id,))
        return f"record patient with id:{patient_id} deleted"
    except Exception as e:
        return f"There is the error:{str(e)}"
if __name__=='__main__':
    mcp.run(transport='stdio')
