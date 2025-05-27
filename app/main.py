from fastapi import FastAPI,HTTPException,status
from langchain_core.documents import Document
from .schema import InputBase,QueryBase
from .rag import KnowRag

app = FastAPI()

@app.get("/")
def root():
  return {"message":"serra pa"}

rag = KnowRag()

@app.post("/input",status_code=status.HTTP_200_OK)
async def input(data : InputBase):
  msg = data.input
  rag.add_data(data=msg)
  return {"success":"data has been sucessfully added"}

@app.post("/retrieve",status_code=status.HTTP_200_OK)
async def retrieve_info(data :QueryBase):
  retrieval_chain = rag.create_chain()
  query = data.query
  try:
     result = retrieval_chain.invoke({"question":query})
  except:
    raise HTTPException(status_code=status.HTTP_409_CONFLICT)
  return {"message":result.split('</think>')[1]}
