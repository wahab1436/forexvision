import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml
from loguru import logger

class EmailAlert:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.enabled = self.config['alerts']['email_enabled']
        
    def send(self, subject, body):
        if not self.enabled:
            return
            
        msg = MIMEMultipart()
        msg['From'] = self.config['alerts']['sender']
        msg['To'] = self.config['alerts']['recipient']
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(self.config['alerts']['smtp_host'], self.config['alerts']['smtp_port'])
            server.starttls()
            server.login(self.config['alerts']['sender'], self.config['alerts']['password'])
            server.send_message(msg)
            server.quit()
            logger.info("Email alert sent")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
