import re


class RegexDict(dict):
    def get(self, event):
        for key in self:
            if re.search(key, event):
                return self[key]
