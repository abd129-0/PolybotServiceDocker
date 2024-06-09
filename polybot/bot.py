import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import boto3
import requests
import ast
from collections import Counter


class Bot:
    def __init__(self, token, telegram_chat_url):
        self.telegram_bot_client = telebot.TeleBot(token)
        self._setup_webhook(token, telegram_chat_url)
        self.prev_path = ""
        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def _setup_webhook(self, token, telegram_chat_url):
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60,
                                             certificate=open('YOURPUBLIC.pem', 'r'))

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text, timeout=5)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id, timeout=5)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type "photo" expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = os.path.dirname(file_info.file_path)

        os.makedirs(folder_name, exist_ok=True)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(chat_id, InputFile(img_path), timeout=5)

    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class ObjectDetectionBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')
        usage_msg = 'Please send a photo to start object detection'

        if "text" in msg:
            if msg["text"] == '/start':
                self.send_text(msg['chat']['id'], 'Welcome!!\n'
                                                  f'{usage_msg}')

        if self.is_current_msg_photo(msg):
            photo_path = self.download_user_photo(msg)
            img_name = os.path.basename(photo_path)
            self._upload_to_s3(photo_path, img_name)
            self._process_image_detection(msg, img_name)

    def _upload_to_s3(self, photo_path, img_name):
        s3 = boto3.client('s3')
        images_bucket = os.getenv('BUCKET_NAME')

        try:
            s3.upload_file(photo_path, images_bucket, img_name)
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")

    def _process_image_detection(self, msg, img_name):
        url = 'http://yolo:8081/predict'
        params = {'imgName': img_name}

        try:
            response = requests.post(url, params=params)
            if response.status_code == 200:
                labels_list = self._extract_labels(response.text)
                result_dict = self._count_items(labels_list)
                result_text = '\n'.join(f"{item} = {count}" for item, count in result_dict.items())
                self.send_text(msg['chat']['id'], f'Detected objects:\n{result_text}')
            else:
                self.send_text(msg['chat']['id'], f'Error: {response.status_code}')
        except Exception as e:
            logger.error(f'HTTP request failed: {e}')

    def _extract_labels(self, response_text):
        labels_index = response_text.find("'labels'")
        labels_substr = response_text[labels_index:]
        open_bracket_index = labels_substr.find("[")
        labels_array_substr = labels_substr[open_bracket_index:]
        close_bracket_index = labels_array_substr.find("]")
        labels_array_substr = labels_array_substr[:close_bracket_index + 1]

        return [label['class'] for label in ast.literal_eval(labels_array_substr)]

    def _count_items(self, items_list):
        return dict(Counter(items_list))
