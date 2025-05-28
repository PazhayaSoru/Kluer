import os
from dotenv import load_dotenv
from .llm import LLMModel
load_dotenv()

GROQ_API_KEY= os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT")
SERPER_API_KEY=os.getenv("SERPER_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
NEO4J_URI= os.getenv("NEO4J_URI")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")
AURA_INSTANCEID=os.getenv("AURA_INSTANCEID")
AURA_INSTANCENAME=os.getenv("AURA_INSTANCENAME")


os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY 
os.environ["SERPER_API_KEY"] = SERPER_API_KEY
os.environ["LANGCHAIN_API_KEY"] =  LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["GROQ_API_KEY"] = "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"
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

#imports 
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from typing import Tuple,List,Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_core.runnables import RunnableBranch,RunnableLambda,RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from .prompts import CONDENSE_QUESTION_PROMPT,base_prompt,extract_prompt

class Entities(BaseModel):
  """ Identifying information from """

  names: List[str] = Field(
    ...,
    description="all the person, organization or business entities that appear in the text"
  )

class KnowRag:
  def __init__(self):
    #LLM for RAG
    self.llm = LLMModel("deepseek-r1-distill-llama-70b").get_model()
    #LLM-Graph Transformer
    self.llm_transformer = LLMGraphTransformer(llm=self.llm)
    #Embeddings for Vector Indexing 
    self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #Vector Index
    self.vector_index = Neo4jVector.from_existing_graph(
    self.embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=['text'],
    embedding_node_property="embedding",
      )
    #Neo4j Database Interface
    self.graph=Neo4jGraph(
      url=NEO4J_URI,
      username=NEO4J_USERNAME,
      password=NEO4J_PASSWORD
      )
    
    self.entity_chain = extract_prompt | self.llm.with_structured_output(Entities)

    


  def add_data(self, data: str):
    documents = [Document(page_content=data)]
    print(documents)
    graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
    
    self.graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    



  def generate_full_text_query(self,input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f"{word}~2 AND" 
    full_text_query += f"{words[-1]}~2"
    return full_text_query.strip()
  
  def structured_retriever(self,question: str) -> str:
    result = ""
    
    entities = self.entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = self.graph.query(
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node, score
            CALL {
                WITH node
                MATCH (node)-[r:MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": self.generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result
  
  def retriever(self,question : str):
    print(f"Search query: {question}")
    structured_data = self.structured_retriever(question)
    unstructured_data = [el.page_content for el in self.vector_index.similarity_search(question)]
    final_data = f"""
      Structured Data:
      {structured_data}
  Unstructured data:
  {"Document ".join(unstructured_data)}
  """
    return final_data
  
  def format_chat_history(self,chat_history : List[Tuple[str,str]]) -> List:
    buffer = []
    for human, ai in chat_history:
      buffer.append(HumanMessage(content=human))
      buffer.append(AIMessage(content=ai))
    return buffer
  
  def create_chain(self):

    _search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    
    (
      RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
        run_name="HasChatHistoryCheck"
    ),
        # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: self.format_chat_history(x["chat_history"])
        ) |
        CONDENSE_QUESTION_PROMPT |
        ChatGroq(
          model="deepseek-r1-distill-llama-70b",
          temperature=0
        ) |
        StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x: x["question"]),
    )


    chain = (
  RunnableParallel(
    {
      "context":_search_query | self.retriever,
      "question":RunnablePassthrough(),
    }
  )
  |base_prompt
  |self.llm
  |StrOutputParser()
    )

    return chain
  
  


    

