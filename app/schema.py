from pydantic import BaseModel



class InputBase(BaseModel):
  input : str


class QueryBase(BaseModel):
  query : str