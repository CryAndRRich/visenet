import smtplib
from email.mime.text import MIMEText

def parse_actions(daily_actions: list) -> list:
    """Chuyển đổi danh sách hành động hàng ngày thành các thông báo, gợi ý"""
    alerts = []
    for entry in daily_actions:
        date = entry["date"]
        stocks = entry["stocks"]
        actions = entry["action"]

        signals = []
        for s, a in zip(stocks, actions):
            if a > 0:
                signals.append(f"Gợi ý: Mua {a} cổ phiếu {s}")
            elif a < 0:
                signals.append(f"Gợi ý: Bán {abs(a)} cổ phiếu {s}")
        if signals:
            text = f"Ngày {date}\n\n" + "\n".join(signals)
            alerts.append(text)

    return alerts

def send_email(subject: str, 
               body: str, 
               to_email: str, 
               from_email: str, 
               app_password: str) -> None:
    """Gửi email với tiêu đề và nội dung đã cho"""
    
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_email, app_password)
        server.send_message(msg)
