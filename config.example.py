"""
WiFi and OTA configuration.
Copy this file to config.py and edit with your network credentials.
"""

WIFI_SSID = "YOUR_WIFI_SSID"
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"

# mDNS hostname (will be accessible as hostname.local)
HOSTNAME = "tree"

# OTA server port
OTA_PORT = 8080

# MQTT settings for Home Assistant integration
MQTT_BROKER = "192.168.2.x"  # Your HA/Mosquitto IP
MQTT_PORT = 1883
MQTT_USER = "your_ha_username"
MQTT_PASSWORD = "your_ha_password"
