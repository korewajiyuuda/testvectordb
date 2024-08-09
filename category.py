import os, requests, json
import google.generativeai as genai # type: ignore
import pandas as pd
from dotenv import load_dotenv # type: ignore

load_dotenv()

instance_url = os.getenv('SERVICENOW_INSTANCE_URL')
username = os.getenv('SERVICENOW_USERNAME')
password = os.getenv('SERVICENOW_PASSWORD')


def get_incidents(number: int = 0):
    url = f"{instance_url}/api/now/table/incident"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    if number != 0:
        params = {'sysparm_limit': number}
    else:
        params = {'sysparm_limit': 1, "sys_id": "**************************"}
    print(url)
    response = requests.get(url, auth=(username, password), headers=headers, params=params, verify=False)
    if response.status_code == 200:
        reqs = response.json().get('result')
        print(f'Request URL: {response.request.url}    $$')
        print("KKK")
        print(reqs)
        return reqs
    else:
        print(f"Failed to retrieve incidents: {response.status_code}, {response.text}")
        return None
    
def update_category(df1):
    url = f"{instance_url}/api/now/table/incident"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    for i,row in df1.iterrows():
        r = row["Sys_id"]
        c = row["Category"]
        url = url + f"/{r}"
        print(url)
        data = json.dumps({ "category": c })        
        print(data)
        response = requests.patch(url, auth=(username, password), headers=headers, data=data, verify=False)
        if response.status_code == 200:
            reqs = response.json().get('result')
            print(f'Request URL: {response.request.url}    $$')
            print("ZZZ")
        else:
            print(f"Failed to update incident for incident {r}: {response.status_code}, {response.text}")
            

def genai_cat(df1):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    # print(os.getenv("GOOGLE_API_KEY"))

    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config={"response_mime_type": "application/json"})
    prompt = """
    You are given a support ticket dump and you are to assign them to their appropriate category.
    
    The categories are ["Software", "Hardware", "Network", "Database", "Inquiry / Help"]

    Go through the support ticket text thoroughly and consider the different technical nuances before responding.

    Use the following schema to return the text:

    class Sentiment(TypedDict):
        category: string

    {review}"""
    result, result1, result2 = [], [], []
    for i,row in df1.iterrows():
        r = row["Description"]
        result = json.loads(model.generate_content(prompt.format(review=r)).text)
        print(result)
        result1.append(result["category"])
    df1["Category"] = result1


incidents = get_incidents(5)
if incidents:
    listnum, listdes, listid, p = [], [], [], 0
    for incident in incidents:
        print(incident)
        listnum.append(incident.get("number"))
        listid.append(incident.get("sys_id"))
        listdes.append(incident.get("description"))
    print(listnum,listdes)
    n = pd.Series(listnum)
    d = pd.Series(listdes)
    i = pd.Series(listid)
    df = pd.DataFrame({"Number":n, "Description":d, "Sys_id": i})
    # print(df)
    df1 = df.head(3)
    genai_cat(df1)
    print(df1)
    # update_category(df1)
else:
    print("No incidents retrieved.")
    