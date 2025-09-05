import requests
import json
import os

def create_incident(config, title, description, team):
    """
    Creates an incident in ServiceNow.
    """
    if not config.get('enabled', False):
        print("ServiceNow integration is disabled.")
        return

    url = config['url']
    user = config['user']
    password = config['password']

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "short_description": title,
        "description": description,
        "assignment_group": team,
        "urgency": "3",
        "impact": "3"
    }
    
    try:
        response = requests.post(url, auth=(user, password), headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        incident_number = response.json().get('result', {}).get('number')
        print(f"✅ Created ServiceNow incident: {incident_number}")
        return incident_number
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to create ServiceNow incident: {e}")
        return None

if __name__ == "__main__":
    # Example usage (will not work without a real config)
    mock_config = {
        "enabled": True,
        "url": "https://yourinstance.service-now.com/api/now/table/incident",
        "user": os.environ.get("SN_USER", "test_user"),
        "password": os.environ.get("SN_PASSWORD", "test_password")
    }
    create_incident(mock_config, "Test Certificate Expiration", "A test certificate is expiring.", "Network Operations")
