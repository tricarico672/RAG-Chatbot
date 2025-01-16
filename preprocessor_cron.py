import requests
from typing import List
from datetime import datetime, timedelta
import logging
import os
import shutil
# Warning control
import warnings
import requests
warnings.filterwarnings('ignore')
##
import os
import requests
import re
from datetime import datetime
import logging
import requests
import os
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
import logging
import requests
import os
from datetime import datetime, timedelta, timezone
from headers import run_pip_installations
import streamlit as st
from preprocess_and_run import PreProcessor
import msal
logging.basicConfig(
    filename='app.log',        # File to write logs to
    level=logging.DEBUG,       # Set the minimum log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)
from dotenv import load_dotenv
load_dotenv()

os.environ['USER_AGENT'] = os.getenv("USER_AGENT")
# Microsoft Graph API credentials

# Split SCOPES into a list
scopes = os.getenv("SCOPES", "").split(",")  # Split by comma
scopes = [scope.strip() for scope in scopes]  # Remove any extra whitespace

# Validate and use the scopes list
if not isinstance(scopes, list) or not all(isinstance(s, str) for s in scopes):
    raise ValueError("SCOPES must be a list of strings.")


logging.basicConfig(level=logging.INFO)

# Get OAuth2 token from Azure AD
def get_access_token():

    app = msal.PublicClientApplication(
        os.getenv("CLIENT_ID"),
        authority=os.getenv("AUTHORITY"),
    )

    result = app.acquire_token_interactive(scopes=scopes)
    print(result)
    logging.info(result)
    if "access_token" in result:
        access_token = result["access_token"]
    else:
        print(result.get("error"))
        logging.error(result.get("error"))

    return access_token
 

# Need to check if we need this or not
# Create a subscription
def create_subscription(access_token):
    url = f"{os.getenv("SITE_URL")}/subscriptions"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    data = {
        "changeType": "updated",
        "notificationUrl": "https://contoso.azurewebsites.net/api/webhook-receiver",
        "resource": "/me/drive/root",
        "expirationDateTime": "2018-01-01T11:23:00.000Z",
        "clientState": "client-specific string"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        return response.json()
    else:
        print(f"Failed to create subscription: {response.status_code}, {response.text}")
        logging.error(f"Failed to create subscription: {response.status_code}, {response.text}")
        return None



def get_changes(access_token, file_name="next_query.txt"):
    has_to_process = None
    if os.path.exists(file_name):
        delta_link = load_delta_link()
    else:
        delta_link = None
    # Delta query URL for the SharePoint site
    url = f"https://graph.microsoft.com/v1.0/sites/{os.getenv('SITE_ID')}/drive/root/delta"
    
    # Headers for authorization
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Initialize variables to hold the file IDs and deleted items
    fileIDs = []
    deleted = []

    # Set the threshold date (3 days ago for example) and make it UTC-aware
    one_day_ago = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=1)

    if delta_link:
        response = requests.get(delta_link, headers=headers)
    else:
        # Start the delta query to get changes
        response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        print("Response status:", response.status_code)
        logging.info(f"Response status: {response.status_code}")
        print("Response headers:", response.headers)
        logging.info(f"Response headers: {response.headers}")
        print("Response content:", response.text)
        logging.info(f"Response content: {response.text}")

        changes = response.json()
   
        # Process the changes in the first page
        for item in changes.get("value", []):
            # Check if the change is for a file and not deleted
            if "file" in item and "deleted" not in item:
                # Convert 'lastModifiedDateTime' to an aware datetime object (UTC)
                last_modified = datetime.fromisoformat(item["lastModifiedDateTime"]).astimezone(timezone.utc)
                
                # Filter by modified date (only include files modified in the last 3 days)
                if last_modified >= one_day_ago:
                    flag = check_file_status(item["createdDateTime"], item["lastModifiedDateTime"])
                    if flag == "added":
                        print(f"File added: {item['name']} (ID: {item['id']})")
                        logging.info(f"File added: {item['name']} (ID: {item['id']})")
                        fileIDs.append(item['id'])  # Add the file ID
                        
                    elif flag == "modified":
                        print(f"File modified: {item['name']} (ID: {item['id']})")
                        logging.info(f"File modified: {item['name']} (ID: {item['id']})")
                        fileIDs.append(item['id'])  # Add the file ID
                    has_to_process = True
            elif "deleted" in item:  # Skip deleted files
                print(f"File deleted (skipped): {item['id']}")
                logging.info(f"File deleted (skipped): {item['id']}")
                deleted.append(item)  # You can track deleted files if needed

        # Store the deltaLink for incremental queries
        delta_link = changes.get("@odata.deltaLink")
        if delta_link:
            print("deltaLink:", delta_link)
            logging.info(f"deltaLink:  {delta_link}")
            save_delta_link(delta_link)
        
        # Handle pagination if there are more changes to fetch
        while "@odata.nextLink" in changes:
            next_url = changes["@odata.nextLink"]
            response = requests.get(next_url, headers=headers)
            if response.status_code != 200:
                print(f"Error fetching next page: {response.status_code}, {response.text}")
                logging.warning(f"Error fetching next page: {response.status_code}, {response.text}")
                return None

            changes = response.json()
            
            # Process the changes in the next page
            for item in changes.get("value", []):
                # Check if the change is for a file and not deleted
                if "file" in item and "deleted" not in item:
                    # Convert 'lastModifiedDateTime' to an aware datetime object (UTC)
                    last_modified = datetime.fromisoformat(item["lastModifiedDateTime"]).astimezone(timezone.utc)
                    
                    # Filter by modified date (only include files modified in the last 3 days)
                    if last_modified >= one_day_ago:
                        flag = check_file_status(item["lastModifiedDatcreatedDateTimeeTime"], item["lastModifiedDateTime"])
                        if flag == "added":
                            print(f"File added: {item['name']} (ID: {item['id']})")
                            logging.info(f"File added: {item['name']} (ID: {item['id']})")
                            fileIDs.append(item['id'])  # Add the file ID
                            
                        elif flag == "modified":
                            print(f"File modified: {item['name']} (ID: {item['id']})")
                            logging.info(f"File modified: {item['name']} (ID: {item['id']})")
                            fileIDs.append(item['id'])  # Add the file ID
                    has_to_process = True
                elif "deleted" in item:
                    print(f"File deleted (skipped): {item['id']}")
                    logging.info(f"File deleted (skipped): {item['id']}")
                    deleted.append(item)  # You can track deleted files if needed

        # Store the deltaLink for incremental queries
        delta_link = changes.get("@odata.deltaLink")
        if delta_link:
            print("deltaLink:", delta_link)
            logging.info(f"deltaLink: {delta_link}")
            save_delta_link(delta_link)

        if has_to_process:
            save_source(changes, has_to_process, [])
        if deleted:
            save_source(changes, False, deleted)
        # Print final list of changed or added file IDs
        print("Changed file IDs:", fileIDs)
        logging.info(f"Changed file IDs: {fileIDs}")
        print("Deleted file IDs:", deleted)
        logging.info(f"Deleted file IDs: {deleted}")
        return fileIDs, deleted  # Return both file IDs and deltaLink
    else:
        print(f"Error fetching changes: {response.status_code}, {response.text}")
        logging.error(f"Error fetching changes: {response.status_code}, {response.text}")
        return None, None


def save_delta_link(delta_link, file_name="next_query.txt"):
    """
    Save the delta link to a file for later use.
    """
    with open(file_name, "w") as file:
        file.write(delta_link)
    print(f"Delta link saved to {file_name}")
    logging.info(f"Delta link saved to {file_name}")

def load_delta_link(file_name="next_query.txt"):
    """
    Load the delta link from the file if it exists.
    """
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            delta_link = file.read().strip()
        print(f"Delta link loaded from {file_name}: {delta_link}")
        logging.info(f"Delta link loaded from {file_name}: {delta_link}")
        return delta_link
    else:
        print(f"No delta link found in {file_name}.")
        logging.warning(f"No delta link found in {file_name}.")
        return None


def save_source(response_data, has_to_process, deleted=[]):
    # File to save the output
    output_file = os.path.join(os.getcwd(), "source_data.txt")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Step 1: Load existing IDs from the file and filenames if deleted is True
    existing_ids = set()
    id_to_filename = {}

    with open(output_file, "r") as file:
        for line in file:
            # Extract the id and filename from lines in the format "filename: ..., id: ..., source: ..."
            parts = line.strip().split(", id: ")
            if len(parts) == 2:
                file_info = parts[0].strip()  # Get "filename: <filename>"
                item_id = parts[1].split(",")[0]  # Get the id value before any trailing commas
                filename = file_info.split(":")[1].strip()  # Extract the filename
                id_to_filename[item_id] = filename  # Store the filename by ID
                existing_ids.add(item_id)
    # If deleted is True, we will map ids to filenames to retrieve them later
    if os.path.exists(output_file) and deleted:                
        # For deleted files, retrieve the filename by ID
        for item_id in deleted:  # `value` contains the list of items
            delete_locally(id_to_filename[item_id.get('id')])
            delete_from_vectorstore(id_to_filename[item_id.get('id')])
            delete_line_by_id(output_file, item_id.get('id'), id_to_filename[item_id.get('id')])
            print(f"File with ID {item_id.get('id')} is deleted. Filename: {id_to_filename[item_id.get('id')]}")
            logging.info(f"File with ID {item_id.get('id')} is deleted. Filename: {id_to_filename[item_id.get('id')]}")
  
    
    # Step 2: Process new items (non-deleted)
    elif os.path.exists(output_file) and has_to_process:
        with open(output_file, "a") as file:  # Append to the file
            for item in response_data.get("value", []):  # `value` contains the list of items
                item_id = item.get("id", "Unknown ID")  # Extract the ID, default to "Unknown ID"
                if item_id in existing_ids:
                    print('File already exists locally')  # Skip if the ID already exists
                    logging.warning('File already exists locally')
                    delete_line_by_id(output_file, item_id, id_to_filename[item_id])
                    delete_locally(id_to_filename[item_id])
                    delete_from_vectorstore(id_to_filename[item_id])
                    print(f"Modified File with ID {item_id} is deleted. Filename: {id_to_filename[item_id]}")
                    logging.info(f"Modified File with ID {item_id} is deleted. Filename: {id_to_filename[item_id]}")
                # Extract other fields
                filename = item.get("name", "Unknown Name")  # Get the filename
                path = item.get("path", "Unknown Path")  # Get the file path
                web_url = item.get("webUrl", "Unknown URL")  # Get the web URL

                # Write the new data to the file
                file.write(f"filename: {filename}, id: {item_id}, source: {web_url}\n")

    print(f"Processed data and updated {output_file}.")
    logging.info(f"Processed data and updated {output_file}.")




def check_file_status(created_datetime: str, last_modified_datetime: str) -> str:
    # Define the format for parsing the date strings (ISO 8601 with UTC 'Z')
    datetime_format = "%Y-%m-%dT%H:%M:%SZ"  # Matches the format "2024-10-13T00:18:04Z"

    # Convert the strings to datetime objects
    created_dt = datetime.strptime(created_datetime, datetime_format)
    modified_dt = datetime.strptime(last_modified_datetime, datetime_format)

    # Compare the full datetime, including time, and check if the difference is <= 1 minute
    time_diff = abs(modified_dt - created_dt)

    if time_diff <= timedelta(minutes=1):  # 1-minute tolerance
        return 'added'
    else:
        return 'modified'

def delete_line_by_id(output_file, target_id, filename):
    # Step 1: Read the file and collect lines that do not contain the target_id
    remaining_lines = []
    
    with open(output_file, "r") as file:
        for line in file:
            # Check if the line contains the target_id
            if target_id not in line or filename not in line:
                remaining_lines.append(line)  # Keep lines that don't contain the target_id
    
    # Step 2: Rewrite the file with the remaining lines
    with open(output_file, "w") as file:
        file.writelines(remaining_lines)
    
    print(f"Line with ID {target_id} or {filename} has been deleted if it existed.")
    logging.info(f"Line with ID {target_id} or {filename} has been deleted if it existed.")

def download_changed_file(access_token, fileIDs=None, output_file=None):
    if fileIDs:
        for fileID in fileIDs:
            url = f"https://graph.microsoft.com/v1.0/sites/{os.getenv('SITE_ID')}/drive/items/{fileID}/content"
            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(url, headers=headers, stream=True)

            if response.status_code == 200:
                # Extract the filename from the Content-Disposition header
                content_disposition = response.headers.get("Content-Disposition", "")
                filename = content_disposition.split("filename=")[-1] if "filename=" in content_disposition else f"{fileID}.file"

                # Clean the filename to remove any invalid characters
                filename = filename.replace('"', '').strip()  # Remove quotes and extra spaces

                # Optional: Further clean the filename by removing or replacing other invalid characters (for Windows systems)
                filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

                # Save the file locally
                if not os.path.exists('files/'):
                    os.makedirs('files/')
                with open(f"files/{filename}", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"File downloaded: {filename}")
                logging.info(f"File downloaded: {filename}")
            else:
                print(f"Error downloading file: {response.status_code}, {response.text}")
                logging.error(f"Error downloading file: {response.status_code}, {response.text}")
    else:
        url = f"https://graph.microsoft.com/v1.0/sites/{os.getenv('SITE_ID')}/drive/root/children"
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(url, headers=headers, stream=True)
        filenames = []
        if response.status_code == 200:
            # Extract the filename from the Content-Disposition header
            data = response.json()
            for item in data.get("value", []):
                filenames.append(item["name"])  # Extract the 'name' field for filenames
                filename = item.get("name", "Unknown Name")  # Get the filename
                path = item.get("path", "Unknown Path")  # Get the file path
                web_url = item.get("webUrl", "Unknown URL")  # Get the web URL
                item_id = item.get("id", "Unknown ID")
                download_url = item.get("@microsoft.graph.downloadUrl")

                response = requests.get(download_url, headers=headers, stream=True)

                if response.status_code == 200:
           
                    filename = filename.replace('"', '').strip()  # Remove quotes and extra spaces

                    # Optional: Further clean the filename by removing or replacing other invalid characters (for Windows systems)
                    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

                    # Save the file locally
                    if not os.path.exists('files/'):
                        os.makedirs('files/')
                    with open(f"files/{filename}", "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"File downloaded: {filename}")
                    logging.info(f"File downloaded: {filename}")

                
                    with open(output_file, "a") as file:  # Append
                        file.write(f"filename: {filename}, id: {item_id}, source: {web_url}\n")
            # Check if there's a next page
            url = data.get("@odata.nextLink")


            while "@odata.nextLink" in data:
                next_url = data["@odata.nextLink"]
                response = requests.get(next_url, headers=headers)
                if response.status_code != 200:
                    print(f"Error fetching next page: {response.status_code}, {response.text}")
                    logging.warning(f"Error fetching next page: {response.status_code}, {response.text}")
                    return None
                else:
                    # Extract the filename from the Content-Disposition header
                    data = response.json()
                    for item in data.get("value", []):
                        filenames.append(item["name"])  # Extract the 'name' field for filenames
                        filename = item.get("name", "Unknown Name")  # Get the filename
                        path = item.get("path", "Unknown Path")  # Get the file path
                        web_url = item.get("webUrl", "Unknown URL")  # Get the web URL
                        item_id = item.get("id", "Unknown ID")
                        download_url = item.get("@microsoft.graph.downloadUrl")

                        response = requests.get(download_url, headers=headers, stream=True)

                        if response.status_code == 200:
                
                            filename = filename.replace('"', '').strip()  # Remove quotes and extra spaces

                            # Optional: Further clean the filename by removing or replacing other invalid characters (for Windows systems)
                            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

                            # Save the file locally
                            if not os.path.exists('files/'):
                                os.makedirs('files/')
                            with open(f"files/{filename}", "wb") as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            print(f"File downloaded: {filename}")
                            logging.info(f"File downloaded: {filename}")

                        
                            with open(output_file, "a") as file:  # Append
                                file.write(f"filename: {filename}, id: {item_id}, source: {web_url}\n")
                    # Check if there's a next page
                    url = data.get("@odata.nextLink")

        else:
            print(f"Error downloading file: {response.status_code}, {response.text}")
            logging.error(f"Error downloading file: {response.status_code}, {response.text}")
        

def delete_from_vectorstore(file_paths):
    processor.delete_from_vectorstore(file_paths)
       

def delete_locally(file_path):
    # Convert file path to corresponding unstructured file path
    file_name = os.path.basename(file_path)  
    file_to_delete = f"{processor.output_path}/{file_name}"
    try:
        os.remove(file_to_delete)
        print(f"Deleted file locally output for {file_path}")
        logging.info(f"Deleted file locally output for {file_path}")
    except FileNotFoundError:
        print(f"File not found locally {file_path}")
        logging.warning(f"File not found locally {file_path}")
    except Exception as e:
        print(f"Error deleting file locally: {e}")
        logging.error(f"Error deleting file locally: {e}")
      
def delete_directory_contents(directory_path):
    # Convert string path to Path object
    path = Path(directory_path)
    
    # Check if it's a valid directory
    if path.is_dir():
        # Iterate over each item in the directory
        for item in path.iterdir():
            try:
                # If the item is a file, delete it
                if item.is_file() or item.is_symlink():
                    item.unlink()  # Removes the file
                # If the item is a directory, delete it recursively
                elif item.is_dir():
                    shutil.rmtree(item)  # Removes the directory and all contents
            except Exception as e:
                print(f"Failed to delete {item}: {e}")
                logging.error(f"Failed to delete {item}: {e}")
    else:
        logging.error(f"The path {directory_path} is not a valid directory.")
        raise ValueError(f"The path {directory_path} is not a valid directory.")


if __name__ == "__main__":
    processor = PreProcessor("./chroma_langchain_db", "amazon.titan-embed-text-v2:0", "anthropic.claude-3-5-sonnet-20240620-v1:0", os.path.join(os.getcwd(), 'files'))
    output_file = os.path.join(os.getcwd(), "source_data.txt")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        access_token = get_access_token()
        if access_token:
            fileIDs, deleted = get_changes(access_token)
            
            if fileIDs:
                print(fileIDs)
                #delete_directory_contents(processor.output_path)
                if os.path.exists('files/') or len(os.listdir('files/')) > 0:
                    download_changed_file(access_token, fileIDs, None)
                    processor.process_directory()
            elif not os.path.exists(output_file) or os.path.getsize(output_file) == 0 or not os.path.exists('files/') or len(os.listdir('files/')) == 0 or len(os.listdir(processor.persist_directory)) <= 1 or not os.path.exists(processor.persist_directory):
                download_changed_file(access_token, None, output_file)
                processor.process_directory()
            else:
                print('Either files deleted or No files changed within last 1 day')
                logging.info('Either files deleted or No files changed within last 1 day')
            '''while True:
            if keyboard.is_pressed('esc'):  # Check if the Esc key is pressed
                print("Stopping observer...")
                break
            time.sleep(1)'''
    except Exception as e:
        print(f"Error: {e.getMessage()}")
        logging.error(f"Error: {e.getMessage()}")
