import os
import requests
import pandas as pd
import paramiko
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, GroupChat, GroupChatManager
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import shutil
import os

# Ensure reports directory exists
os.makedirs("reports", exist_ok=True)


# Load environment variables
load_dotenv()

# Azure Authentication
credential = ClientSecretCredential(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET"),
)

# Fetch Azure cost data
def fetch_azure_cost_data() -> Dict[str, Any]:
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    url = f"https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.Consumption/usageDetails?api-version=2023-03-01"
    params = {
        "$filter": f"properties/usageStart ge '{start_date}' and properties/usageEnd le '{end_date}'"
    }
    
    token = credential.get_token("https://management.azure.com/.default").token
    headers = {"Authorization": f"Bearer {token}"}
    print(f"Fetching cost data from {start_date} to {end_date}")
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Azure API Error: {response.status_code}, {response.text}")
        return {}

# SSH into VM and fetch CPU usage
def get_vm_cpu_usage(vm_ip: str, username: str, private_key_path: str) -> Optional[float]:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    private_key = paramiko.RSAKey.from_private_key_file(private_key_path)
    
    try:
        ssh.connect(vm_ip, username=username, pkey=private_key)
        stdin, stdout, stderr = ssh.exec_command("top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - $1}'")
        cpu_usage = float(stdout.read().decode().strip())
        ssh.close()
        return cpu_usage
    except Exception as e:
        print(f"SSH Error: {e}")
        return None

def get_vm_cpu_usage_azure(resource_id: str) -> float:
    start_time = (datetime.utcnow() - timedelta(days=7)).isoformat() + 'Z'
    end_time = datetime.utcnow().isoformat() + 'Z'
    
    url = f"https://management.azure.com{resource_id}/providers/Microsoft.Insights/metrics"
    params = {
        'metricnames': 'Percentage CPU',
        'api-version': '2018-01-01',
        'metricnames': 'Percentage CPU',
        'timespan': f"{start_time}/{end_time}",
        'interval': 'P1D',
        'aggregation': 'average'
    }
    
    token = credential.get_token("https://management.azure.com/.default").token
    headers = {'Authorization': f'Bearer {token}'}
    
    response = requests.get(url, headers=headers, params=params)
    print(f"Getting CPU metrics for {resource_id}")
    if response.status_code == 200:
        data = response.json()
        values = data.get('value', [])
        if values:
            cpu_values = [v['average'] for v in values[0]['timeseries'][0]['data'] if 'average' in v]
            return sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
    return 0.0

def get_vm_uptime(resource_id: str) -> int:
    # Get VM start time from Azure Metrics
    start_time = (datetime.utcnow() - timedelta(days=30)).isoformat() + 'Z'
    end_time = datetime.utcnow().isoformat() + 'Z'
    
    url = f"https://management.azure.com{resource_id}/providers/Microsoft.Insights/metrics"
    params = {
        'api-version': '2018-01-01',
        'metricnames': 'VM Uptime',
        'timespan': f"{start_time}/{end_time}",
        'interval': 'P1D',
        'aggregation': 'total'
    }
    
    token = credential.get_token("https://management.azure.com/.default").token
    headers = {'Authorization': f'Bearer {token}'}
    
    response = requests.get(url, headers=headers, params=params)
    print(f"Getting uptime for {resource_id}")
    if response.status_code == 200:
        data = response.json()
        # Parse uptime from response
        return int(data.get('value', [{}])[0].get('timeseries', [{}])[0].get('data', [{}])[-1].get('total', 7))
    return 7

# Analyze VM utilization
# Updated analyze_vm_utilization function
# Updated analyze_vm_utilization function
def analyze_vm_utilization(data: List[Dict[str, Any]]) -> List[str]:
    underutilized_vms = []
    for item in data:
        resource_id = item.get("resourceId", "").lower()
        if "/virtualmachines/" in resource_id and "microsoft.compute" in resource_id:
            # Get actual metrics
            cpu_usage = get_vm_cpu_usage_azure(item["resourceId"])
            uptime_days = get_vm_uptime(item["resourceId"])
            
            if cpu_usage < 10 and uptime_days >= 7:
                # Return FULL resource ID
                underutilized_vms.append(item["resourceId"]) 
    
    return underutilized_vms

from azure.mgmt.compute import ComputeManagementClient

def scale_down_vm(vm_id: str) -> str:
    """Improved scaling with validation"""
    try:
        if not vm_id.startswith("/subscriptions/"):
            raise ValueError("Invalid Azure ID format - must start with /subscriptions/")
            
        parts = vm_id.lower().split('/')
        required_parts = ["subscriptions", "resourcegroups", "providers", "microsoft.compute", "virtualmachines"]
        
        if not all(part in parts for part in required_parts):
            raise ValueError("Missing required path components")
            
        # Find resource group index dynamically
        try:
            rg_index = parts.index("resourcegroups") + 1
            vm_index = parts.index("virtualmachines") + 1
            resource_group = parts[rg_index]
            vm_name = parts[vm_index]
        except ValueError:
            raise ValueError("Invalid resource ID structure")

        compute_client = ComputeManagementClient(
            credential, 
            os.getenv("AZURE_SUBSCRIPTION_ID")
        )
        
        poller = compute_client.virtual_machines.begin_deallocate(resource_group, vm_name)
        poller.wait()
        return f"Successfully scaled down VM: {vm_name}"
        
    except Exception as e:
        print(f"Scaling Error: {str(e)}")
        return f"Error scaling VM: {str(e)}"

def generate_cost_report(data: List[Dict[str, Any]]) -> str:
    report_path = "cost_report.csv"
    
    # Create empty file if no data
    if not data:
        pd.DataFrame().to_csv(report_path)
        return report_path
        
    df = pd.DataFrame([{
        'Resource': item.get('resourceName', 'Unknown'),
        'Cost': item.get('cost', 0.0)
    } for item in data])
    
    df.to_csv(report_path)
    return report_path

# Generate cost report
def generate_cost_report_tool() -> str:
    global cost_data_store
    try:
        if not cost_data_store:
            return "No cost data available to generate report"
            
        report_path = generate_cost_report(cost_data_store)
        return f"TEMP_REPORT_PATH: {os.path.abspath(report_path)}"
    except Exception as e:
        return f"Report generation failed: {str(e)}"

# Initialize AutoGen agents
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PHI_API_KEY = os.getenv("PHI_API_KEY")
config_list = [{
    "model": "llama-3.3-70b-versatile",
    "api_key": GROQ_API_KEY,
    "api_type": "groq",
}]
llm_config = {"config_list": config_list}

# Define agents with roles and tools
monitoring_agent = AssistantAgent(
    name="MonitoringAgent",
    system_message="""You MUST:
1. Use fetch_azure_cost_data_tool
2. Use analyze_vm_utilization_tool with the result
3. Output EXACTLY: 'CostOptimizer, scale these VMs: [FULL_RESOURCE_IDS]'
4. If no VMs, say 'ReportingAgent, generate report'""",
    llm_config=llm_config
)

cost_optimizer_agent = AssistantAgent(
    name="CostOptimizer",
    system_message="""You MUST:
1. Use EXACT resource IDs from MonitoringAgent
2. Use scale_down_vm_tool for each ID
3. After scaling, say 'ReportingAgent, generate report with path: [EXACT_PATH]'""",
    llm_config=llm_config
)

reporting_agent = AssistantAgent(
    name="ReportingAgent",
    system_message="""You MUST:
1. Use generate_cost_report_tool FIRST
2. Use save_report_locally_tool WITH THE EXACT OUTPUT FROM STEP 1
3. Final message MUST contain 'REPORT_SAVED_AT:'""",
    llm_config=llm_config
)
# Add validation for tool calls
def validate_tool_calls(messages):
    for msg in messages:
        if "tool_calls" in msg:
            for call in msg["tool_calls"]:
                if "arguments" not in call or "function" not in call:
                    return False
    return True

# Add this new tool to store cost data for reporting
cost_data_store = []


# Register tools with agents
@monitoring_agent.register_for_execution()
@monitoring_agent.register_for_llm(description="Fetch Azure cost data")
def fetch_azure_cost_data_tool() -> List[Dict[str, Any]]:
    global cost_data_store
    raw_data = fetch_azure_cost_data()
    
    seen_ids = set()
    filtered_data = []
    
    for item in raw_data.get("value", []):
        resource_id = item.get("properties", {}).get("resourceId")
        if resource_id and resource_id not in seen_ids:
            seen_ids.add(resource_id)
            filtered_data.append({
                "resourceId": resource_id,
                "resourceName": item.get("properties", {}).get("resourceName"),
                "cost": item.get("properties", {}).get("costInBillingCurrency", {}).get("value", 0.0)
            })
    
    cost_data_store = filtered_data
    return filtered_data

@monitoring_agent.register_for_execution()
@monitoring_agent.register_for_llm(description="Identify underutilized VMs from cost data")
def analyze_vm_utilization_tool(data: List[Dict[str, Any]]) -> List[str]:
    """Analyzes VM utilization metrics to find underutilized resources. Returns list of VM IDs."""
    return analyze_vm_utilization(data)

@cost_optimizer_agent.register_for_execution()
@cost_optimizer_agent.register_for_llm(description="Scale down an underutilized VM")
def scale_down_vm_tool(vm_id: str) -> str:
    """Scales down a specific VM by its ID. Returns status message."""
    return scale_down_vm(vm_id)

# Modified reporting tools to use stored data
@reporting_agent.register_for_execution()
@reporting_agent.register_for_llm(description="Generate cost report from stored data")
def generate_cost_report_tool() -> str:
    global cost_data_store
    try:
        report_path = generate_cost_report(cost_data_store)
        return f"TEMP_REPORT_PATH: {os.path.abspath(report_path)}"
    except Exception as e:
        return f"Report generation failed: {str(e)}"

@reporting_agent.register_for_execution()
@reporting_agent.register_for_llm(description="Save report locally")
def save_report_locally_tool(report_info: str) -> str:
    """Saves report using the generated path"""
    try:
        if not report_info.startswith("TEMP_REPORT_PATH: "):
            return "ERROR: Invalid report path format. Must start with 'TEMP_REPORT_PATH: '"
            
        report_path = report_info.split("TEMP_REPORT_PATH: ")[1]
        final_path = os.path.join("reports", os.path.basename(report_path))
        
        shutil.copy(report_path, final_path)
        return f"REPORT_SAVED_AT: {os.path.abspath(final_path)}"
    
    except Exception as e:
        return f"Error saving report: {str(e)}"

# Set up group chat


def validate_message_structure(msg: dict):
    if msg.get("role") == "assistant":
        if "tool_calls" in msg:
            for call in msg["tool_calls"]:
                if not all(key in call for key in ("function", "arguments", "id")):
                    return False
        elif "content" not in msg:
            return False
    return True


def is_termination_msg(msg):
    content = msg.get("content", "").lower()
    return any([
        "report_saved_at:" in content,
        "error saving report" in content,
        "no vms to scale" in content
    ])
# Modified group chat setup with increased rounds
group_chat = GroupChat(
    agents=[monitoring_agent, cost_optimizer_agent, reporting_agent],
    messages=[],
    max_round=8,  # Increased to allow full workflow
    speaker_selection_method="round_robin",
    allowed_or_disallowed_speaker_transitions={
        monitoring_agent: [cost_optimizer_agent],
        cost_optimizer_agent: [reporting_agent],
        reporting_agent: []
    },
    speaker_transitions_type="allowed"
)
manager = GroupChatManager(
    groupchat=group_chat, 
    llm_config={"config_list": config_list},
    system_message="ENFORCE: Tool calls must have full function/arguments",
    #message_validation=validate_message_structure
)
from tenacity import retry, stop_after_attempt, wait_exponential

user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    is_termination_msg=is_termination_msg
)

# Wrap the initiate_chat call with retry logic
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_initiate_chat():
    user_proxy.initiate_chat(
    manager,
    message="Monitor Azure costs, scale underutilized VMs, and save a cost report. Confirm final report path when done.",
    )

# Call the wrapped function
safe_initiate_chat()
# Start the conversation
