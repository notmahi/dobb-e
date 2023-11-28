import json
import logging
import os

import requests

file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(file_path)
dotenv_path = os.path.join(directory_path, ".env")


def dotenv_handler(filepath):
    environ = {}
    try:
        with open(filepath, "r") as file:
            for line in file:
                # Ignore empty lines and comments
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    environ[key] = value
    except FileNotFoundError:
        logging.warning(f"File {filepath} not found. Returning empty environment.")
    return environ


DOTENV = dotenv_handler(dotenv_path)
SLACK_WEBHOOK_URL = DOTENV.get("SLACK_WEBHOOK_URL")


def send_slack_message(message, webhook_url=SLACK_WEBHOOK_URL):
    if webhook_url is None:
        logging.error("No Slack webhook URL provided. Cannot send message.")
        return False
    payload = {"text": message}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        logging.info("Message sent to Slack successfully!")
        return True
    except requests.exceptions.HTTPError as errh:
        logging.error("HTTP Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        logging.error("Error connecting to Slack:", errc)
    except requests.exceptions.RequestException as err:
        logging.error("Error sending message to Slack:", err)
    return False


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
