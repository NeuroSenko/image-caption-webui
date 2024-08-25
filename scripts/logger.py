from datetime import datetime


def log(message, display_full_date=False):
    now = datetime.now()
    time = (
        now.strftime("%Y-%m-%d %H:%M:%S")
        if display_full_date
        else now.strftime("%H:%M:%S")
    )
    print(f"[{time}] {message}")
