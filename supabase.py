import os
import supabase
from supabase import create_client, Client
import logging
from dotenv import load_dotenv
logging.basicConfig(
    filename='app.log',        # File to write logs to
    level=logging.DEBUG,       # Set the minimum log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

load_dotenv()

class SuperBase:

  def __init__(self):
    self.connect()


  def connect(self):
    url: str = os.getenv("SUPABASE_URL")
    key: str = os.getenv("SUPABASE_KEY")
    supabase: Client = create_client(url, key)


  def fetch(self):
    response = supabase.table("documents").select("*").execute()


  def insert(self):
    try:
      response = supabase.table("documents").insert([
          { "id": 1, "name": "Nepal" },
          { "id": 1, "name": "Vietnam" },
        ]).execute()
      return response
    except Exception as exception:
      return exception
    

  def update(self):
    response = (
    supabase.table("documents")
    .update({"name": "Australia"})
    .eq("id", 1)
    .execute()
    )


  def delete(self):
    response = supabase.table("documents").delete().in_("id", [1, 2, 3]).execute()



