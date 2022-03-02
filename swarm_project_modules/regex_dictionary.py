import re


class RegexDict(dict):
    def get(self, event):
        return (self[key] for key in self if re.match(key, event))
