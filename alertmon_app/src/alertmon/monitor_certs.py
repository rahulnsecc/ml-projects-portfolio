import json
import os
import random
from datetime import datetime, timedelta

def monitor_certs(config_file, output_file):
    """
    Monitors certificate expiration and writes status to a JSON file.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in config file: {config_file}")
        return

    certificates = config.get("certificates", [])
    
    certs_status = []
    
    for cert in certificates:
        # Fetch data from config file
        name = cert.get('name')
        cert_type = cert.get('type')
        platform = cert.get('platform')
        sub_platform = cert.get('sub_platform')
        serial_no = cert.get('serial_no')
        issuance_date_str = cert.get('issuance_date')
        expiration_date_str = cert.get('expiration_date')
        subscriber = cert.get('subscriber')
        authorizer = cert.get('authorizer')
        sop_created = cert.get('sop_created')
        is_updated_in_ca_portal = cert.get('is_updated_in_ca_portal')
        spoc_name = cert.get('spoc_name')
        spoc_email = cert.get('spoc_email')
        spoc_phone = cert.get('spoc_phone')
        remedy_queue = cert.get('remedy_queue')
        remarks = cert.get('remarks')
        vendor_name = cert.get('vendor_name')
        signed_by = cert.get('signed_by')

        try:
            expiration_date = datetime.strptime(expiration_date_str, '%Y-%m-%d').date()
            issuance_date = datetime.strptime(issuance_date_str, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            print(f"❌ Invalid date format for certificate: {name}. Skipping.")
            continue

        days_until_expiration = (expiration_date - datetime.now().date()).days
        
        status = "Healthy"
        next_action = "None"
        incident_crq_number = ""

        if days_until_expiration < 0:
            status = "Expired"
            next_action = "URGENT RENEWAL"
            incident_crq_number = f"INC{random.randint(10000, 99999)}"
            print(f"🚨 {name} is EXPIRED. Triggering alerts...")
        elif days_until_expiration <= 30: # 30 days is the warning period
            status = "Expiring Soon"
            next_action = f"Renew before {expiration_date.strftime('%Y-%m-%d')}"
            incident_crq_number = f"CRQ{random.randint(10000, 99999)}"
            print(f"⚠️ {name} is expiring in {days_until_expiration} days. Triggering alerts...")
        
        certs_status.append({
            "type": cert_type,
            "platform": platform,
            "sub_platform": sub_platform,
            "name": name,
            "serial_no": serial_no,
            "issuance_date": issuance_date_str,
            "expiration_date": expiration_date_str,
            "status": status,
            "next_action": next_action,
            "subscriber": subscriber,
            "authorizer": authorizer,
            "sop_created": sop_created,
            "is_updated_in_ca_portal": is_updated_in_ca_portal,
            "spoc_name": spoc_name,
            "spoc_email": spoc_email,
            "spoc_phone": spoc_phone,
            "remedy_queue": remedy_queue,
            "remarks": remarks,
            "vendor_name": vendor_name,
            "signed_by": signed_by,
            "incident_crq_number": incident_crq_number
        })

    with open(output_file, 'w') as f:
        json.dump(certs_status, f, indent=2)
    
    print(f"✅ Wrote {len(certs_status)} cert statuses to {output_file}")


if __name__ == "__main__":
    monitor_certs("certs_config.json", "logs/certs_log.json")
