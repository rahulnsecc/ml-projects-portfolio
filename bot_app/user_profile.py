class UserProfile:
    def __init__(self, date_range: str = None, details: dict = None):
        self.date_range = date_range
        self.details = details if details else {}
