import smtplib
from email.mime.text import MIMEText

def parse_actions(daily_actions: list) -> list:
    """
    Parse daily stock actions into alert messages
    
    Parameters:
        daily_actions: List of dictionaries with keys "date", "stocks", and "action"

    Returns:
        list: List of alert messages
    """
    alerts = []
    for entry in daily_actions:
        date = entry["date"]
        stocks = entry["stocks"]
        actions = entry["action"]

        signals = []
        for s, a in zip(stocks, actions):
            if a > 0:
                signals.append(f"Suggestion: Buy {a} shares of {s}")
            elif a < 0:
                signals.append(f"Suggestion: Sell {abs(a)} shares of {s}")
        if signals:
            text = f"Date {date}\n\n" + "\n".join(signals) 
            alerts.append(text)

    return alerts

def send_email(subject: str, 
               body: str, 
               to_email: str, 
               from_email: str, 
               app_password: str) -> None:
    """
    Send an email notification
    
    Parameters:
        subject: Subject of the email
        body: Body of the email
        to_email: Recipient's email address
        from_email: Sender's email address
        app_password: App password for the sender's email account
    """
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_email, app_password)
        server.send_message(msg)
