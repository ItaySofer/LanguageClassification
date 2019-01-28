from urllib.parse import urlparse
from abc import abstractmethod
import emoji


class SplitToWords(object):
    def __init__(self):
        pass

    def __call__(self, tweet):
        return tweet.split(" ")


class RemoveTokenBase(object):
    def __init__(self):
        pass
    
    def __call__(self, word_list):
        return [word for word in word_list if self.should_remove(word) is False]

    @abstractmethod
    def should_remove(self, word):
        pass


class RemoveURLs(RemoveTokenBase):
    def should_remove(self, s):
        parseResult = urlparse(s)
        return parseResult.scheme != '' or parseResult.netloc != ''


class RemoveMentions(RemoveTokenBase):
    def should_remove(self, s):
        return s.startswith('@')


class RemoveResponseToken(RemoveTokenBase):
    def should_remove(self, s):
        return s == "RT"


class RemoveHashtags(RemoveTokenBase):
    def should_remove(self, s):
        return s.startswith('#')


class RemoveBlanks(RemoveTokenBase):
    def should_remove(self, s):
        return s == ""


class RemoveEmojis(RemoveTokenBase):
    def should_remove(self, s):
        return all(c in emoji.UNICODE_EMOJI for c in s)
