import os
from azure.search.documents import SearchClient 
from azure.search.documents.models import VectorizedQuery   
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SearchCorpus:
    
    def __init__(self):
        
        # assign the Search variables for Azure Cogintive Search - use .env file and in the web app configure the application settings
        AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
        AZURE_SEARCH_ADMIN_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
        AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
        credential_search = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
        OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL")

        self.sc = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX, credential=credential_search)
        self.model = OPENAI_EMBED_MODEL
        self.openai_client = OpenAI()
        
        print(f"Init SearchShadow for index - {AZURE_SEARCH_INDEX}")
    
    def get_embedding(self, text, model):
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(input = [text], model=model).data[0].embedding
    
    def search_hybrid(self, query: str) -> str:
        vector_query = VectorizedQuery(vector=self.get_embedding(query, self.model), k_nearest_neighbors=5, fields="contentVector")
        #vector = Vector(value=self.get_embedding(query, self.model), k=3, fields="contentVector")
        results = []

        r = self.sc.search(  
            search_text=query,  # set this to engage a Hybrid Search
            vector_queries= [vector_query],  
            select=["title", "category", "sourcefile", "content"],
            top=3,
        )  
        for doc in r:
                results.append(f"[CATEGORY:  {doc['category']}]" + " " + f"[SOURCEFILE:  {doc['sourcefile']}]" + doc['content'])
                #print("\n".join(f"[CATEGORY:  {doc['category']}]"  + " " + f"[SOURCEFILE:  {doc['sourcefile']}]"))
                print(f"[TITLE:  {doc['title']}]")
        return ("\n".join(results))