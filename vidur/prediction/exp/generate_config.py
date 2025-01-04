import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
import json

manifest_path = "vidur/prediction/cl_manifest.xml"
config_output_path = "vidur/prediction/config"


def generate_config(ip_address, predictor_port, backend_port):
    config = {
        "ip_address": ip_address,
        "predictor_port": predictor_port,
        "backend_port": backend_port
    }
    return config


predictor_port = 8100
backend_port = 8000

tree = ET.parse(manifest_path)
num_schedulers = 1
# get root element
nodes = {}
root = tree.getroot()
upload = True

for child in root:
    if "node" in child.tag:
        node_info = {}
        node_name = child.get("client_id")
        nodes[node_name] = node_info
        for subchild in child:
            if "host" in subchild.tag:
                ip_address = subchild.get("ipv4")
                node_info["ip_adresses"] = ip_address
            if "services" in subchild.tag:
                host_name = subchild[0].get("hostname")
                node_info["hostname"] = host_name

nodes = OrderedDict(sorted(nodes.items()))
host_config_files = os.path.join(config_output_path, "host_configs.json")
host_files = os.path.join(config_output_path, "hosts")

host_names = []
with open(host_config_files, "w+") as f, open(host_files, "w+") as n:
    j = 0
    configs = {}
    for node in nodes:
        node_info = nodes[node]
        host_names.append("asdwb@" + node_info["hostname"])
        config = generate_config(node_info["ip_adresses"], predictor_port, backend_port)
        configs[node_info["hostname"]] = config
    json.dump(configs, f, sort_keys=True, indent = 4)
    for host in host_names:
        n.write(host + "\n")
