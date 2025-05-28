from fastapi import FastAPI,HTTPException,status,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .schema import InputBase,QueryBase
from .rag import KnowRag
import easyocr as eocr
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.post("/input_image")
async def upload_image(image: UploadFile):
    allowed_extensions = {"jpg", "jpeg", "png"}
    extension = image.filename.split(".")[-1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    file_location = f"ocr.{extension}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    reader = eocr.Reader(['en'])
    result = reader.readtext(file_location)

    result_text = ""
    for (bbox, text, prob) in result:
        result_text += " " + text 
    rag.add_data(data=result_text)
    return {"message": result_text.strip()}
  